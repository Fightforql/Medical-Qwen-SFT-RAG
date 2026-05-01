# inference/chat.py

import torch
import logging
from typing import Dict, Any
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)
import os
logger = logging.getLogger("QwenFinetune")

class ChatBot:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_args = config['model_args']
        self.training_args = config['training_args']
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _load_model_and_tokenizer(self):
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
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载训练好的LoRA适配器
        adapter_path = os.path.join(self.training_args['output_dir'], "final_model")
        if not os.path.exists(adapter_path):
             # 如果final_model不存在，尝试加载best_model
             adapter_path = os.path.join(self.training_args['output_dir'], "best_model")
             if not os.path.exists(adapter_path):
                 raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
        logger.info("Model and tokenizer loaded for chat.")
        return model, tokenizer

    def start_chat(self):
        print("Starting interactive chat. Type 'exit' or 'quit' to end.")
        history = []
        while True:
            try:
                query = input("\nUser: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                # 构建完整的对话历史
                messages = []
                for turn in history:
                    messages.append({"role": "user", "content": turn["query"]})
                    messages.append({"role": "assistant", "content": turn["response"]})
                messages.append({"role": "user", "content": query})

                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                print("Assistant: ", end="")
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=1024,
                    streamer=self.streamer
                )
                
                # 解码以保存历史记录
                response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                history.append({"query": query, "response": response})

            except KeyboardInterrupt:
                print("\nExiting chat.")
                break
