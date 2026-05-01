
import os
import logging
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict

# 获取在项目主入口配置好的logger
logger = logging.getLogger("QwenFinetune")

def download_data(source: str, dest_dir: str) -> str:
    """
    从Hugging Face Hub加载数据集，并将其保存为本地的JSONL文件。

    Args:
        source (str): Hugging Face上的数据集名称, 例如 "FreedomIntelligence/HuatuoGPT-sft-data"。
        dest_dir (str): 用于存放下载和处理后数据的目录。

    Returns:
        str: 保存到本地的原始数据文件的完整路径。
    """
    logger.info(f"Attempting to load dataset '{source}' from Hugging Face Hub...")
    
    # 定义本地文件的保存路径
    # 我们将其保存为jsonl格式，因为这对于SFT（监督微调）数据非常通用
    local_file_path = os.path.join(dest_dir, "huatuo_raw_data.jsonl")

    # 检查文件是否已存在，如果存在则可以跳过下载，节约时间
    if os.path.exists(local_file_path):
        logger.info(f"Raw data file already exists at {local_file_path}. Skipping download.")
        return local_file_path

    try:
        # 1. 加载数据集
        # 我们通常使用 'train' 分割来进行微调
        # streaming=True 可以避免一次性将整个数据集加载到内存，对大数据集更友好
        dataset = load_dataset(source, split="train", streaming=False) # 改为False以获取总数用于tqdm
        
        logger.info(f"Dataset '{source}' loaded successfully. It has {len(dataset)} samples.")
        logger.info(f"Saving dataset to local file: {local_file_path}")
        
        # 2. 迭代数据集并保存到文件
        with open(local_file_path, 'w', encoding='utf-8') as f:
            # 使用tqdm来显示保存进度
            for item in tqdm(dataset, desc="Saving dataset to file"):
                # 将每个样本（它本身是一个字典）转换为JSON字符串并写入文件
                # ensure_ascii=False 保证中文字符能被正确写入
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Successfully saved dataset to local file: {local_file_path}")
        
        return local_file_path

    except Exception as e:
        logger.error(f"Failed to load or save dataset '{source}'. Error: {e}")
        # 抛出异常
        raise e




def clean_text_file(input_path: str, output_path: str):
    """
    清洗从Hugging Face下载的JSONL格式的华佗数据集。
    1. 验证数据格式是否为 {"data": ["问：...", "答：..."]}。
    2. 去掉空的或无效的问答对。
    3. 将数据转换为更标准的 {"data": {"问": "...", "答": "..."}} 格式并保存。
    """
    logger.info(f"Starting to clean and reformat dataset file: {input_path}")
    
    valid_records_count = 0
    invalid_records_count = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        try:
            total_lines = sum(1 for line in open(input_path, 'r', encoding='utf-8'))
            fin.seek(0)
        except Exception:
            total_lines = None

        for line in tqdm(fin, total=total_lines, desc="Cleaning and Reformatting"):
            try:
                record = json.loads(line)

                # --- 核心修改逻辑开始 ---

                # 1. 验证数据格式：'data' 键存在，其值为一个包含2个元素的列表
                if 'data' not in record or not isinstance(record.get('data'), list) or len(record['data']) != 2:
                    invalid_records_count += 1
                    continue
                
                qa_list = record['data']
                q_str = qa_list[0]
                a_str = qa_list[1]

                # 2. 验证并提取问答内容
                if not isinstance(q_str, str) or not q_str.startswith("问：") or \
                   not isinstance(a_str, str) or not a_str.startswith("答："):
                    invalid_records_count += 1
                    continue
                
                # 去掉 "问：" 和 "答：" 前缀，并去除首尾空白
                question = q_str.replace("问：", "", 1).strip()
                answer = a_str.replace("答：", "", 1).strip()

                # 3. 确保清洗后的问答内容不为空
                if not question or not answer:
                    invalid_records_count += 1
                    continue

                # 4. 构建新的、标准化的数据结构
                new_record = {
                    "data": {
                        "问": question,
                        "答": answer
                    }
                }

                # 5. 将新的JSON对象写入文件
                fout.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                valid_records_count += 1
                
                
            except (json.JSONDecodeError, TypeError):
                invalid_records_count += 1
                continue

    logger.info("Cleaning and reformatting finished.")
    logger.info(f"Total valid records kept: {valid_records_count}")
    logger.info(f"Total invalid or malformed records removed: {invalid_records_count}")



def format_to_sft(input_path: str) -> List[Dict]:
    """
    加载已经清洗过的JSONL文件。
    由于数据已经是我们需要的格式，这个函数主要负责读取文件并将其加载到内存中，
    以便后续的数据切分。

    Args:
        input_path (str): 清洗后的JSONL文件路径。

    Returns:
        List[Dict]: 包含所有数据记录的列表。
    """
    logger.info(f"Loading cleaned data from {input_path} into memory...")
    
    formatted_data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading formatted data"):
                try:
                    # 将每一行的JSON字符串解析成一个Python字典
                    record = json.loads(line)
                    formatted_data.append(record)
                except json.JSONDecodeError:
                    # 如果有损坏的行，记录一个警告并跳过
                    logger.warning(f"Skipping a malformed line in {input_path}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Please ensure the cleaning step ran successfully.")
        raise

    logger.info(f"Successfully loaded {len(formatted_data)} records into memory.")
    return formatted_data