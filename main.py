
import argparse
import logging
import os

# 假设您的工具函数和数据处理模块位于这些路径
# 您需要确保这些import在您的项目中是有效的
from utils.logger import setup_logger
from utils.file_io import load_yaml_config
from hf_lora.data_pipeline.dataset_manager import DatasetManager

# 框架版 Trainer
from hf_lora.training.trainer import QwenFineTuner 
# 手写版 Trainer
from my_lora.training.my_trainer import FineTuner

from inference.chat import ChatBot
from rag.chat_RAG import RAGChatBot
from inference.web_ui import WebUI

from evaluation.test import ModelEvaluator
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def main():
    parser = argparse.ArgumentParser(description="Qwen Fine-tuning Project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 数据处理子命令
    parser_data = subparsers.add_parser("data", help="Run the data processing pipeline")
    parser_data.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    # 训练子命令
    parser_train = subparsers.add_parser("train", help="Run model training")
    parser_train.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    ## --- 核心修改 2: 添加 --impl 参数用于选择实现 ---
    parser_train.add_argument(
        '--impl', 
        type=str, 
        default='hf', 
        choices=['hf', 'custom'], 
        help="Choose the trainer implementation: 'hf' for Hugging Face SFTTrainer, 'custom' for the hand-rolled version."
    )
    
    # 评估子命令
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate the fine-tuned model")
    parser_eval.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    
    # 推理子命令
    parser_chat=subparsers.add_parser("chat", help="Start interactive command-line chat")
    parser_chat.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    parser_chat.add_argument('--type', type=str, default='finetune', help='Path to the config file')
    
    parser_api=subparsers.add_parser("serve_api", help="Serve the model via FastAPI")
    parser_api.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    parser_ui=subparsers.add_parser("serve_ui", help="Serve the model via Gradio Web UI")
    parser_ui.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()

    # 配置日志
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)
    logger = logging.getLogger("QwenFinetune")
    
    logger.info(f"Executing command: {args.command}")

    if hasattr(args, 'config_path'):
        config = load_yaml_config(args.config_path)

    if args.command == "data":
        manager = DatasetManager(config)
        manager.prepare_dataset()
    elif args.command == "train":
        ## 根据 --impl 参数的值选择不同的训练器 ---
        if args.impl == 'hf':
            logger.info("Using Hugging Face framework-based trainer (QwenFineTuner).")
            finetuner = QwenFineTuner(config)
        elif args.impl == 'custom':
            logger.info("Using custom hand-rolled trainer (FineTuner).")
            finetuner = FineTuner(config)
            
        else:
            # 理论上 argparse 的 choices 会阻止这种情况，但作为保险
            logger.error(f"Unknown implementation choice: {args.impl}")
            return
            
        finetuner.train()

    elif args.command == "evaluate":
        logger.info("Starting evaluating...")
        evaluator=ModelEvaluator(config)
        evaluator.evaluate()
    
    elif args.command == "chat":
        logger.info("Starting command-line chat bot...")
        if args.type=='finetune':
            chatbot = ChatBot(config)
            chatbot.start_chat()
        elif args.type=='RAG':
            chatbot=RAGChatBot(config)
            chatbot.launch_ui()
        
    elif args.command == "serve_ui":
        logger.info("Starting Gradio Web UI...")
        ui = WebUI(config)
        ui.launch()

if __name__ == "__main__":
    main()
