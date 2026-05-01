

import os
import torch
import json
import logging
from tqdm import tqdm
from typing import Dict, Any
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import evaluate # Hugging Face官方的评估库
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict

# 推荐安装jieba用于中文分词
try:
    import jieba
    logging.info("jieba found, will use it for tokenization.")
except ImportError:
    logging.warning("jieba not found. ROUGE/BLEU scores will be based on character-level tokenization.")
    jieba = None

logger = logging.getLogger("QwenFinetune")

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型评估器。
        
        Args:
            config (Dict[str, Any]): 从 config.yaml 加载的配置字典。
        """
        self.config = config
        self.model_args = config['model_args']
        self.data_args = config['data_args']
        self.training_args = config['training_args']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        加载基础模型和分词器，并融合训练好的LoRA适配器。
        """
        logger.info(f"Loading base model: {self.model_args['model_name_or_path']}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if self.model_args.get('use_qlora', False) else None

        model = AutoModelForCausalLM.from_pretrained(
            self.model_args['model_name_or_path'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args['model_name_or_path'],
            trust_remote_code=True,
            pad_token='<|endoftext|>'
        )

        # 找到训练输出目录中最佳的检查点
        lora_checkpoint_path = os.path.join(self.training_args['output_dir'])
        checkpoints = [d for d in os.listdir(lora_checkpoint_path) if d.startswith('checkpoint-')]
        if not checkpoints:
            # 如果没有checkpoint，尝试加载final_model
            final_model_path = os.path.join(lora_checkpoint_path, "final_model")
            if os.path.exists(final_model_path):
                 lora_checkpoint_path = final_model_path
            else:
                raise FileNotFoundError(f"No checkpoints or final_model found in {lora_checkpoint_path}")
        else:
            latest_checkpoint = max(checkpoints, key=lambda d: int(d.split('-')[-1]))
            lora_checkpoint_path = os.path.join(lora_checkpoint_path, latest_checkpoint)

        logger.info(f"Loading LoRA adapter from: {lora_checkpoint_path}")
        
    

        # 1. 从配置文件加载LoRA配置
        adapter_path=lora_checkpoint_path
        config = PeftConfig.from_pretrained(adapter_path)

        # 2. 将基础模型转换为PeftModel
        # 注意：此时模型只有LoRA结构，还没有加载权重
        model = get_peft_model(model, config)

        # 3. 手动加载LoRA权重文件
        adapter_weights_path = os.path.join(adapter_path, "lora_adapters.pt")
        adapter_weights = torch.load(adapter_weights_path, map_location=model.device)

        # 4. 将权重加载到PeftModel中
        set_peft_model_state_dict(model, adapter_weights)

        logger.info("LoRA adapter weights loaded manually and successfully.")
        #model = PeftModel.from_pretrained(model, lora_checkpoint_path,local_files_only=True)
        
        model.eval()
        logger.info("Model and tokenizer loaded successfully for evaluation.")
        return model, tokenizer

    def evaluate(self, batch_size=8, max_new_tokens=512):
        """
        在测试集上运行评估并计算指标。
        """
        test_data_path = os.path.join(self.data_args['processed_data_dir'], self.data_args['test_file'])
        logger.info(f"Loading test dataset from: {test_data_path}")
        dataset = load_dataset('json', data_files=test_data_path, split='train')
        
        predictions = []
        references = []

        logger.info("Starting generation on the test set...")
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            
            ## --- 核心修改在这里 ---
            # 直接访问 batch 字典中的 'data' 键，它是一个列表
            batch_data_list = batch['data']
            
            # 现在可以正确地从列表中的每个字典里提取 '问' 和 '答'
            questions = [item['问'] for item in batch_data_list]
            ground_truths = [item['答'] for item in batch_data_list]
            # ---------------------

            prompts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in questions]
            references.extend(ground_truths)

            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            for j, output in enumerate(outputs):
                response_tokens = output[len(inputs.input_ids[j]):]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                predictions.append(response)

        logger.info("Saving predictions and references to a file...")
        results_to_save = {
            "predictions": predictions,
            "references": references
        }
        save_path = os.path.join(self.training_args['output_dir'], "generated_outputs.json")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        logger.info(f"Outputs saved successfully to {save_path}")

        self._calculate_and_log_metrics(predictions, references)

    def _calculate_and_log_metrics(self, predictions, references):
        """计算并打印ROUGE和BLEU分数"""
        logger.info("\nCalculating metrics...")
        
        def tokenize_fn(texts):
            if jieba:
                return [" ".join(jieba.cut(text)) for text in texts]
            return [" ".join(list(text)) for text in texts]

        tokenized_preds = tokenize_fn(predictions)
        tokenized_refs = tokenize_fn(references)

        rouge = evaluate.load('rouge')
        bleu = evaluate.load('bleu')

        rouge_results = rouge.compute(predictions=tokenized_preds, references=tokenized_refs)
        bleu_results = bleu.compute(predictions=tokenized_preds, references=[[ref] for ref in tokenized_refs])

        logger.info("\n--- Evaluation Results ---")
        logger.info("ROUGE Scores:")
        for key, value in rouge_results.items():
            logger.info(f"  {key}: {value*100:.2f}")
            
        logger.info("\nBLEU Score:")
        logger.info(f"  BLEU: {bleu_results['bleu']*100:.2f}")
        
        results = {"rouge": rouge_results, "bleu": bleu_results}
        results_path = os.path.join(self.training_args['output_dir'], "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"\nResults have been saved to {results_path}")
