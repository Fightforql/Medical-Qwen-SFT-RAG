# inference/api_server.py

import torch
import logging
import os
from typing import Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# 假设主入口会设置好日志
logger = logging.getLogger("QwenFinetune")

# --- 全局变量，用于缓存模型，避免每次请求都重新加载 ---
model_cache = {}

class ChatRequest(BaseModel):
    """定义API请求的数据结构"""
    query: str
    history: List[Dict[str, str]] = [] # 例如 [{"role": "user", "content": "你好"}]
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8

def load_model_for_api(config: Dict[str, Any]):
    """
    加载模型和分词器，并放入全局缓存。
    这个函数只在服务启动时执行一次。
    """
    global model_cache
    if "model" in model_cache and "tokenizer" in model_cache:
        logger.info("Model and tokenizer already loaded in cache.")
        return model_cache["model"], model_cache["tokenizer"]

    model_args = config['model_args']
    training_args = config['training_args']
    
    logger.info(f"Loading base model for API: {model_args['model_name_or_path']}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if model_args.get('use_qlora', False) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_args['model_name_or_path'],
        quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args['model_name_or_path'], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 自动寻找最佳或最终模型
    output_dir = training_args['output_dir']
    adapter_path = os.path.join(output_dir, "best_model")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(output_dir, "final_model")
        if not os.path.exists(adapter_path):
             raise FileNotFoundError(f"Neither 'best_model' nor 'final_model' found in {output_dir}")

    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    logger.info("Model and tokenizer loaded successfully for API server.")
    
    model_cache["model"] = model
    model_cache["tokenizer"] = tokenizer
    
    return model, tokenizer

def serve_api(config: Dict[str, Any]):
    """启动FastAPI服务"""
    app = FastAPI(
        title="Qwen Fine-tuned Model API",
        description="一个用于与微调后的Qwen模型交互的API服务。",
        version="1.0.0"
    )

    @app.on_event("startup")
    async def startup_event():
        """在服务启动时加载模型"""
        logger.info("Server startup: loading model into memory...")
        load_model_for_api(config)

    @app.post("/v1/chat/completions")
    async def chat_endpoint(request: ChatRequest):
        """
        接收聊天请求并返回模型生成的完整回复。
        路径模仿OpenAI API格式，方便集成。
        """
        model = model_cache["model"]
        tokenizer = model_cache["tokenizer"]
        
        messages = request.history + [{"role": "user", "content": request.query}]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id
        )
        
        response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        return {"response": response, "history": messages + [{"role": "assistant", "content": response}]}

    import uvicorn
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    # 0.0.0.0 使其可以被外部访问
    uvicorn.run(app, host="0.0.0.0", port=8000)

