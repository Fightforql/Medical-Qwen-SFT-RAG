

import os
import gc
import json
import shutil
import logging
from glob import glob
from tqdm import tqdm
from typing import Dict, Any

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from my_lora.dataset.dataset import SFTDataCollator
from datasets import load_dataset
from my_lora.model.lora import apply_lora_to_model
from utils.logger import print_trainable_parameters

logger = logging.getLogger("QwenFinetune")




class FineTuner:
    """封装了整个微调流程的核心类。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_args = config['model_args']
        self.data_args = config['data_args']
        self.training_args = config['training_args']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_dir = self.training_args['output_dir']
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.model, self.tokenizer = self._setup_model_and_tokenizer()
        self.train_dataloader, self.eval_dataloader = self._setup_datasets_and_collator()
        self.optimizer, self.lr_scheduler = self._setup_optimizer_and_scheduler()

    def _setup_model_and_tokenizer(self):
        logging.info(f"Loading base model: {self.model_args['model_name_or_path']}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        ) if self.model_args.get('use_qlora', False) else None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_args['model_name_or_path'],
            quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args['model_name_or_path'], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.gradient_checkpointing_enable()
        model = apply_lora_to_model(
            model=model, rank=self.model_args['lora_rank'],
            lora_alpha=self.model_args['lora_alpha'], lora_dropout=self.model_args['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model.config.use_cache = False
        print_trainable_parameters(model)
        return model, tokenizer

    def _setup_datasets_and_collator(self):
        train_path = os.path.join(self.data_args['processed_data_dir'], self.data_args['train_file'])
        val_path = os.path.join(self.data_args['processed_data_dir'], self.data_args['validation_file'])
        train_dataset = load_dataset('json', data_files=train_path, split='train')
        eval_dataset = load_dataset('json', data_files=val_path, split='train')
        data_collator = SFTDataCollator(self.tokenizer, self.data_args['max_seq_length'])
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_args['per_device_train_batch_size'],
            shuffle=True, collate_fn=data_collator, num_workers=4, pin_memory=True
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.training_args['per_device_train_batch_size'],
            shuffle=False, collate_fn=data_collator, num_workers=4, pin_memory=True
        )
        return train_loader, eval_loader

    def _setup_optimizer_and_scheduler(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = bnb.optim.PagedAdamW32bit(
            trainable_params, lr=self.training_args['learning_rate'],
            weight_decay=self.training_args.get('weight_decay', 0.01)
        )
        num_update_steps_per_epoch = len(self.train_dataloader) // self.training_args['gradient_accumulation_steps']
        self.max_steps = self.training_args['num_train_epochs'] * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=self.training_args['lr_scheduler_type'], optimizer=optimizer,
            num_warmup_steps=int(self.max_steps * self.training_args['warmup_ratio']),
            num_training_steps=self.max_steps,
        )
        return optimizer, lr_scheduler

    def save_checkpoint(self, epoch, step, best_metric):
        ckpt_name = f"checkpoint-step-{step}"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        os.makedirs(ckpt_path, exist_ok=True)
        logging.info(f"Saving checkpoint to {ckpt_path}")
        
        trainable_state_dict = {k: v for k, v in self.model.state_dict().items() if v.requires_grad}
        torch.save(trainable_state_dict, os.path.join(ckpt_path, "lora_adapters.pt"))
        self.tokenizer.save_pretrained(ckpt_path)
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
        
        with open(os.path.join(ckpt_path, "trainer_state.json"), "w") as f:
            json.dump({"epoch": epoch, "step": step, "best_metric": best_metric, "config": self.config}, f, indent=4)
        

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.eval_dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logging.info(f"Evaluation finished. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return {"eval_loss": avg_loss, "eval_perplexity": perplexity}

    def train(self):
        global_step, best_eval_loss = 0, float('inf')
        for epoch in range(self.training_args['num_train_epochs']):
            self.model.train()
            progress_bar = tqdm(total=self.max_steps, initial=global_step, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss / self.training_args['gradient_accumulation_steps']
                loss.backward()
                
                if (step + 1) % self.training_args['gradient_accumulation_steps'] == 0:
                    if self.training_args.get('gradient_clipping'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args['gradient_clipping'])
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item() * self.training_args['gradient_accumulation_steps'], lr=self.lr_scheduler.get_last_lr()[0])

                    if global_step % self.training_args['logging_steps'] == 0:
                        train_loss = loss.item() * self.training_args['gradient_accumulation_steps']
                        lr = self.lr_scheduler.get_last_lr()[0]
                        logging.info(f"Step: {global_step} | Train Loss: {train_loss:.4f} | Learning Rate: {lr:.2e}")
                    
                    if global_step > 0 and global_step % self.training_args['eval_steps'] == 0:
                        eval_metrics = self.evaluate()
                        logging.info(f"Step: {global_step} | Eval Loss: {eval_metrics['eval_loss']:.4f} | Eval Perplexity: {eval_metrics['eval_perplexity']:.2f}")
                        
                        if eval_metrics['eval_loss'] < best_eval_loss:
                            best_eval_loss = eval_metrics['eval_loss']
                            logging.info(f"🎉 New best model found with eval_loss: {best_eval_loss:.4f}. Saving to 'best_model'.")
                            best_model_path = os.path.join(self.output_dir, "best_model")
                            trainable_dict = {k: v for k, v in self.model.state_dict().items() if v.requires_grad}
                            os.makedirs(best_model_path, exist_ok=True)
                            torch.save(trainable_dict, os.path.join(best_model_path, "lora_adapters.pt"))
                            self.tokenizer.save_pretrained(best_model_path)
                    
                    # 保存检查点 (Save Checkpoint)
                    # 您可以在 config.yaml 中将 save_steps 设置为 5000
                    if global_step > 0 and global_step % self.training_args['save_steps'] == 0:
                        self.save_checkpoint(epoch, global_step, best_eval_loss)
            progress_bar.close()

        final_path = os.path.join(self.output_dir, "final_model")
        trainable_dict = {k: v for k, v in self.model.state_dict().items() if v.requires_grad}
        os.makedirs(final_path, exist_ok=True)
        torch.save(trainable_dict, os.path.join(final_path, "lora_adapters.pt"))
        self.tokenizer.save_pretrained(final_path)
        del self.model, self.optimizer, self.lr_scheduler; gc.collect(); torch.cuda.empty_cache()
