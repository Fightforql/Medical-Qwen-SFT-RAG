# inference/chat_rag.py

import torch
import logging
import os
from typing import Dict, Any,List
from threading import Thread

import gradio as gr
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# 从我们自己的模块导入RAG Pipeline
from rag.rag_pipeline import RAGPipeline
from rag.my_rag_pipeline import MyRAGPipeline

logger = logging.getLogger("QwenFinetune")

class RAGChatBot:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_args = config['model_args']
        self.training_args = config['training_args']
        self.inference_args = config['inference_args']
        
        # 1. 初始化RAG Pipeline
        self.rag_pipeline = RAGPipeline(config)
        self.use_qlora=self.inference_args['use_qlora']
        # 2. 加载微调好的LLM
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading base model for RAG chat: {self.model_args['model_name_or_path']}")
        
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

        if self.use_qlora:
                adapter_path = os.path.join(self.training_args['output_dir'], self.inference_args['adapter_path'])
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
        logger.info("Fine-tuned model and tokenizer loaded for RAG chat.")
        return model, tokenizer

    def _build_rag_prompt(self, query: str) -> str:
        """构建包含检索上下文的Prompt"""
        # 1. 检索
        retrieved_context = self.rag_pipeline.retrieve(query)
        
        # 2. 构建Prompt
        prompt = (
            "You are a professional medical assistant. Please provide a helpful and accurate answer based on the following retrieved context and your own knowledge. "
            "If the context is not relevant, ignore it and rely on your internal knowledge.\n\n"
            f"--- CONTEXT ---\n{retrieved_context}\n"
            f"--- QUESTION ---\n{query}\n\n"
            "--- ANSWER ---\n"
        )
        return prompt

    def stream_chat(self, message: str, history: List[List[str]]):
        # 1. 构建RAG增强后的Prompt
        rag_prompt = self._build_rag_prompt(message)
        
        # 2. 准备模型输入
        messages = [{"role": "user", "content": rag_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=self.inference_args['max_new_tokens'],
            do_sample=True,
            top_p=self.inference_args['top_p'],
            temperature=self.inference_args['temperature'],
            pad_token_id=self.tokenizer.pad_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            # 返回的是历史对话，加上当前用户消息和模型回复
            yield history + [[message, partial_text]]


    def launch_ui(self):
        """启动Gradio Web UI"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Qwen 微调模型 + RAG 知识库问答")
            chatbot = gr.Chatbot(
                [], elem_id="chatbot", bubble_full_width=False,height=600
            )
            with gr.Row():
                txt = gr.Textbox(
                    scale=4, show_label=False,
                    placeholder="请输入您的医疗问题...", container=False,
                )
                submit_btn = gr.Button("发送", variant="primary")

            txt.submit(self.stream_chat, [txt, chatbot], chatbot)
            submit_btn.click(self.stream_chat, [txt, chatbot], chatbot)
            txt.submit(lambda: "", [], txt)
            submit_btn.click(lambda: "", [], txt)

        logger.info("Launching RAG Chat UI...")
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

