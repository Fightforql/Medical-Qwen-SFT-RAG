
import os
import logging
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from .utils import download_data,clean_text_file,format_to_sft
from utils.file_io import write_jsonl

logger = logging.getLogger("QwenFinetune")

class DatasetManager:
    def __init__(self, config: Dict[str, Any]):
        self.data_args = config['data_args']
        self.raw_dir = self.data_args['raw_data_dir']
        self.processed_dir = self.data_args['processed_data_dir']
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def prepare_dataset(self):
        """
        完整的数据准备流水线：下载 -> 清洗 -> 格式化 -> 切分
        """
        logger.info("Starting dataset preparation pipeline...")

        # 1. 下载数据 (TODO: 实现具体的下载逻辑)
        logger.info("Step 1: Downloading data...")
        raw_file_path = download_data(self.data_args['dataset_name'], self.raw_dir)
        logger.info(f"Raw data saved to {raw_file_path}")
        
        
        # 2. 清洗数据
        logger.info("Step 2: Cleaning data...")
        cleaned_file_path = os.path.join(self.raw_dir, "cleaned_data.jsonl")
        clean_text_file(raw_file_path, cleaned_file_path)
        logger.info(f"Cleaned data saved to {cleaned_file_path}")

        # 3. 格式化数据
        logger.info("Step 3: Formatting data to SFT format...")
        formatted_data = format_to_sft(cleaned_file_path)
        logger.info(f"Formatted {len(formatted_data)} samples.")

        # 4. 切分数据集
        logger.info("Step 4: Splitting dataset...")
        train_val_data, test_data = train_test_split(
            formatted_data, 
            test_size=self.data_args['test_size'], 
            random_state=42
        )
        
        val_size_adjusted = self.data_args['validation_size'] / (1 - self.data_args['test_size'])
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=42
        )

        # 5. 保存切分后的数据
        write_jsonl(train_data, os.path.join(self.processed_dir, self.data_args['train_file']))
        write_jsonl(val_data, os.path.join(self.processed_dir, self.data_args['validation_file']))
        write_jsonl(test_data, os.path.join(self.processed_dir, self.data_args['test_file']))
        
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        logger.info("Dataset preparation finished successfully.")
        