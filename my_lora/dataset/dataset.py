# src/dataset.py

import logging
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

class SFTDataset(Dataset):
    """
    一个可选的、用于有监督微调（SFT）的数据集封装类。
    它的主要作用是封装 Hugging Face `datasets` 库加载的数据对象，
    使其在形式上更像一个标准的 PyTorch Dataset。
    在您的 trainer.py 代码中，由于 DataLoader 可以直接处理
    `load_dataset` 返回的对象，所以这个封装类被省略了，这是更简洁的做法。
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): .jsonl 格式的数据文件路径。
        """
        # 使用Hugging Face的load_dataset来加载数据
        self.data = load_dataset('json', data_files=file_path, split='train')
        logging.info(f"Loaded dataset from {file_path}. Size: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 直接返回数据集中的一行（一个字典）
        return self.data[idx]

class SFTDataCollator:
    """
    数据整理器 (Data Collator)，这是 DataLoader 的核心组件。
    它负责将从数据集中取出的一批“散装”样本（a batch of samples）
    转换为模型能够直接处理的、格式规整的张量。
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        """
        Args:
            tokenizer (PreTrainedTokenizer): 用于编码文本的分词器。
            max_seq_length (int): 模型的最大序列长度。
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        这个方法会在DataLoader每次打包数据时被调用。

        Args:
            batch (List[Dict[str, Any]]): 从数据集中获取的一批样本，
                                         例如 [{'data': {'问': '...', '答': '...'}}, ...]。

        Returns:
            Dict[str, torch.Tensor]: 包含 input_ids, attention_mask 和 labels 的字典。
        """
        # 1. 从批次数据中提取问答对，并构建 ChatML 格式的文本
        formatted_texts = []
        for item in batch:
            qa_pair = item.get('data', {})
            question = qa_pair.get('问', '')
            answer = qa_pair.get('答', '')
            
            # 使用Qwen的ChatML格式，这对于Qwen模型至关重要
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}{self.tokenizer.eos_token}"
            formatted_texts.append(text)
        
        # 2. 使用分词器对格式化后的文本进行编码
        tokenized_output = self.tokenizer(
            formatted_texts,
            max_length=self.max_seq_length,
            padding="max_length", # 将所有序列填充到最大长度
            truncation=True,      # 如果序列超过最大长度，则进行截断
            return_tensors="pt"   # 返回PyTorch张量
        )

        # 3. 创建 labels 用于计算损失
        # 在语言模型微调中，labels 通常是 input_ids 的一个副本。
        # 我们需要将 padding token 的位置设置为 -100，这样损失函数在计算时会忽略它们。
        labels = tokenized_output["input_ids"].clone()
        labels[tokenized_output["input_ids"] == self.tokenizer.pad_token_id] = -100
        tokenized_output["labels"] = labels

        return tokenized_output
