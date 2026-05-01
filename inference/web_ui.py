# inference/web_ui.py

import torch
import logging
import os
from typing import Dict, Any, List, Generator
from threading import Thread

import gradio as gr
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

logger = logging.getLogger("QwenFinetune")

class WebUI:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Web UI 服务。
        
        Args:
            config (Dict[str, Any]): 从 config.yaml 加载的配置字典。
        """
        self.config = config
        self.model_args = config['model_args']
        self.training_args = config['training_args']
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
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 寻找并加载训练好的LoRA适配器
        
        adapter_path = os.path.join(self.training_args['output_dir'], "best_model")
        
        if not os.path.exists(adapter_path):
             adapter_path = os.path.join(self.training_args['output_dir'], "final_model")
             if not os.path.exists(adapter_path):
                 # 如果都没有，则寻找最新的checkpoint
                checkpoints = sorted(
                    [d for d in os.listdir(self.training_args['output_dir']) if d.startswith('checkpoint-')],
                    key=lambda d: int(d.split('-')[-1])
                )
                if checkpoints:
                    adapter_path = os.path.join(self.training_args['output_dir'], checkpoints[-1])
                else:
                    raise FileNotFoundError(f"No adapter found in {self.training_args['output_dir']}")

        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        model.eval()
        logger.info("Model and tokenizer loaded for Web UI.")
        return model, tokenizer

    def _format_history(self, history: List[List[str]]) -> List[Dict[str, str]]:
        """将Gradio的history格式转换为Qwen的ChatML格式"""
        messages = []
        for turn in history:
            user_msg, bot_msg = turn
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        return messages

    def _predict_stream(self, message: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        # history 是 [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
        messages = history + [{"role": "user", "content": message}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            yield [{"role": "user", "content": message}, 
            {"role": "assistant",               "content":partial_text}]


    def launch(self):
        """
        启动Gradio Web UI。
        """
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Qwen 1.8B 微调模型聊天界面")
            
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
                type="messages",
                height=700,
            )

            with gr.Row():
                txt = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="请输入您的问题...",
                    container=False,
                )
                submit_btn = gr.Button("发送", variant="primary")

            # 绑定事件
            txt.submit(self._predict_stream, [txt, chatbot], chatbot)
            submit_btn.click(self._predict_stream, [txt, chatbot], chatbot)
            
            # 清空输入框
            txt.submit(lambda: "", [], txt)
            submit_btn.click(lambda: "", [], txt)

        logger.info("Launching Gradio Web UI...")
        # share=True 会生成一个公开链接
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
