
import os
import torch
import logging
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

logger = logging.getLogger("QwenFinetune")

class QwenFineTuner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_args = config['model_args']
        self.data_args = config['data_args']
        self.training_args_dict = config['training_args']
        logger.info("QwenFineTuner initialized with config.")

    def _load_model_and_tokenizer(self):
        """加载模型和分词器，并进行QLoRA配置"""
        logger.info(f"Loading base model: {self.model_args['model_name_or_path']}")
        
        bnb_config = None
        if self.model_args['use_qlora']:
            logger.info("Using QLoRA. Configuring BitsAndBytes...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args['model_name_or_path'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args['model_name_or_path'],
            trust_remote_code=True,
            pad_token='<|endoftext|>' # Qwen推荐使用<|endoftext|>作为pad_token
        )

        logger.info("Setting up PEFT LoraConfig...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 对于Qwen1.5模型，推荐的target_modules通常是这些
        peft_config = LoraConfig(
            r=self.model_args['lora_rank'],
            lora_alpha=self.model_args['lora_alpha'],
            lora_dropout=self.model_args['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.config.use_cache = False
        
        logger.info("Model and Tokenizer loaded and configured.")
        self.model.print_trainable_parameters()

    def _prepare_datasets(self):
        """加载处理好的数据集，并创建正确的格式化函数"""
        logger.info("Loading processed datasets...")
        processed_dir = self.data_args['processed_data_dir']
        train_path = os.path.join(processed_dir, self.data_args['train_file'])
        val_path = os.path.join(processed_dir, self.data_args['validation_file'])

        self.train_dataset = load_dataset('json', data_files=train_path, split='train')
        self.eval_dataset = load_dataset('json', data_files=val_path, split='train')
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.eval_dataset)}")
        
        
        def formatting_prompts_func(examples: Dict[str, List]) -> List[str]:
            output_texts = []
            # SFTTrainer会批量处理数据，所以这里的 'data' 是一个列表
            for i in range(len(examples['data'])):
                qa_pair = examples['data'][i]
                question = qa_pair.get('问')
                answer = qa_pair.get('答')
                
                # 构建一个适合模型学习的完整对话字符串
                # 使用Qwen的ChatML格式，效果更好
                text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
                output_texts.append(text)
            return output_texts
        
        self.formatting_func = formatting_prompts_func

    def train(self):
        """启动训练流程"""
        self._load_model_and_tokenizer()
        self._prepare_datasets()

        logger.info("Initializing Hugging Face Trainer...")
        training_args = TrainingArguments(
            output_dir=self.training_args_dict['output_dir'],
            per_device_train_batch_size=self.training_args_dict['per_device_train_batch_size'],
            gradient_accumulation_steps=self.training_args_dict['gradient_accumulation_steps'],
            learning_rate=self.training_args_dict['learning_rate'],
            num_train_epochs=self.training_args_dict['num_train_epochs'],
            lr_scheduler_type=self.training_args_dict['lr_scheduler_type'],
            warmup_ratio=self.training_args_dict['warmup_ratio'],
            logging_steps=self.training_args_dict['logging_steps'],
            save_steps=self.training_args_dict['save_steps'],
            optim=self.training_args_dict['optim'],
            fp16=True, # 对于40系列显卡，可以设为True
            # bf16=False, # 如果是A100/H100等，可以开启bf16
            logging_dir=f"{self.training_args_dict['output_dir']}/logs",
            #evaluation_strategy="steps",
            eval_steps=2000,
            #save_strategy="steps",
        )

        trainer = SFTTrainer(
            model=self.model,
            #tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            formatting_func=self.formatting_func,
            #max_seq_length=1024, # 根据你的VRAM和数据调整
            #dataset_text_field="text" # 告诉SFTTrainer我们格式化后的文本在哪个字段，如果formatting_func返回字符串列表，则不需要此参数
        )
        
        # 如果formatting_func返回的是字符串列表，就不需要dataset_text_field
        if 'dataset_text_field' in trainer.args.to_dict():
            del trainer.args.dataset_text_field

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished.")

        final_save_path = os.path.join(self.training_args_dict['output_dir'], "final_model")
        trainer.save_model(final_save_path)
        logger.info(f"Final model saved to {final_save_path}")
