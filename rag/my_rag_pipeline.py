# src/rag_pipeline.py

import os
import json
import logging
import numpy as np
import torch
import sys
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, List

logger = logging.getLogger("QwenFinetune")


class ManualEmbeddingModel:
    """
    一个“手搓”的类，用于直接使用 Hugging Face transformers 库生成句子嵌入。
    它负责处理分词、模型推理和均值池化（mean pooling）。
    """
    def __init__(self, model_name: str, device: str):
        logger.info(f"手动加载分词器和模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_length = self.model.config.max_position_embeddings # 获取模型最大长度
        self.device = device
        logger.info("ManualEmbeddingModel 初始化成功。")

    def _mean_pooling(self, model_output, attention_mask):
        """对词元嵌入执行均值池化（mean pooling）。"""
        token_embeddings = model_output[0]  # model_output 的第一个元素包含所有词元的嵌入
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 32, convert_to_numpy: bool = True) -> np.ndarray:
        """
        将句子列表编码为嵌入向量。
        """
        self.model.eval()
        all_embeddings = []
        
        # 手动实现的进度条
        total_batches = (len(sentences) - 1) // batch_size + 1
        logger.info(f"正在编码 {len(sentences)} 个句子，共 {total_batches} 个批次...")

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, # 强制截断到模型最大长度
                return_tensors='pt'
            ).to(self.device)
            
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 对嵌入进行归一化
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            all_embeddings.append(sentence_embeddings.cpu())

            # 手动更新进度条
            progress = (i // batch_size + 1) / total_batches * 100
            sys.stdout.write(f"\r -> 编码进度: {progress:.2f}%")
            sys.stdout.flush()

        sys.stdout.write("\n") # 进度条结束后换行
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return embeddings.numpy()
        return embeddings


class SimpleVectorIndex:
    """
    一个“手搓”的向量索引，用于存储嵌入向量并执行
    暴力的 L2（欧氏）距离搜索。
    """
    def __init__(self, dimension: int = 0):
        self.dimension = dimension
        self.embeddings = None
        self.is_trained = False
        logger.info("SimpleVectorIndex 初始化成功。")

    def add(self, vectors: np.ndarray):
        """向索引中添加一批向量。"""
        if self.embeddings is None:
            self.dimension = vectors.shape[1]
            self.embeddings = vectors.copy()
        else:
            if vectors.shape[1] != self.dimension:
                raise ValueError("向量维度不匹配。")
            self.embeddings = np.vstack((self.embeddings, vectors))
        self.is_trained = True
        logger.info(f"已添加 {len(vectors)} 个向量。索引中总向量数: {len(self.embeddings)}")

    def search(self, query_vectors: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
        """
        为每个查询向量搜索 k 个最近邻。
        返回距离和索引。
        """
        if not self.is_trained:
            raise RuntimeError("索引尚未训练。请在搜索前添加向量。")
        
        num_queries = query_vectors.shape[0]
        distances = np.zeros((num_queries, k), dtype=np.float32)
        indices = np.zeros((num_queries, k), dtype=np.int64)

        for i in range(num_queries):
            query_vec = query_vectors[i]
            # 手动计算 L2 距离（欧氏距离的平方）
            diff = self.embeddings - query_vec
            dist_sq = np.sum(diff ** 2, axis=1)
            
            # 通过对距离排序来获取前 k 个索引
            sorted_indices = np.argsort(dist_sq)
            top_k_indices = sorted_indices[:k]
            
            indices[i] = top_k_indices
            distances[i] = np.sqrt(dist_sq[top_k_indices])
            
        return distances, indices

    def write_index(self, path: str):
        """将嵌入向量保存到 numpy 文件。"""
        logger.info(f"正在将自定义索引的嵌入向量保存到 {path}...")
        np.save(path, self.embeddings)

    @classmethod
    def read_index(cls, path: str):
        """从 numpy 文件加载嵌入向量并创建索引。"""
        logger.info(f"正在从 {path} 加载自定义索引的嵌入向量...")
        embeddings = np.load(path)
        dimension = embeddings.shape[1]
        index = cls(dimension)
        index.add(embeddings)
        return index


class MyRAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_args = config['rag_args']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("使用自定义组件初始化 RAG 流水线...")
        # 使用我们自定义的 ManualEmbeddingModel
        self.embedding_model = ManualEmbeddingModel(
            model_name=self.rag_args['embedding_model'], 
            device=self.device
        )
        
        self.documents = []
        self.index = None
        
        self._build_or_load_index()

    def _load_documents(self) -> List[str]:
        knowledge_file = self.rag_args['knowledge_base_file']
        docs_path = os.path.splitext(self.rag_args['faiss_index_path'])[0] + "_docs.json"

        # 如果存在预先保存的文档文件，则直接加载
        if os.path.exists(docs_path):
             logger.info(f"从预存文件 {docs_path} 加载文档...")
             with open(docs_path, 'r', encoding='utf-8') as f:
                 return json.load(f)

        logger.info(f"从原始知识库 {knowledge_file} 加载文档...")
        docs = []
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                question = data.get('data', {}).get('问', '')
                answer = data.get('data', {}).get('答', '')
                if question and answer:
                    docs.append(f"问：{question}\n答：{answer}")
        
        # 保存处理后的文档，以便下次快速加载
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=4)
            
        return docs

    def _build_or_load_index(self):
        # 为我们的自定义索引起使用 .npy 扩展名
        index_path = os.path.splitext(self.rag_args['faiss_index_path'])[0] + ".npy"
        
        if os.path.exists(index_path):
            logger.info(f"从 {index_path} 加载已存在的自定义索引...")
            # 使用我们自定义的 SimpleVectorIndex 来加载数据
            self.index = SimpleVectorIndex.read_index(index_path)
            self.documents = self._load_documents()
            logger.info("自定义索引和文档加载成功。")
        else:
            logger.info("未找到已存在的自定义索引，正在构建新索引...")
            self.documents = self._load_documents()
            
            logger.info(f"为知识库编码 {len(self.documents)} 个文档...")
            embeddings = self.embedding_model.encode(
                self.documents, 
                batch_size=self.rag_args.get('batch_size', 128) # 从配置中获取 batch_size
            )
            
            dimension = embeddings.shape[1]
            # 使用我们自定义的 SimpleVectorIndex
            self.index = SimpleVectorIndex(dimension)
            self.index.add(np.array(embeddings, dtype=np.float32))
            
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            # 使用我们自定义索引的方法进行保存
            self.index.write_index(index_path)
            logger.info(f"自定义索引构建完成并保存至 {index_path}")

    def retrieve(self, query: str) -> str:
        logger.info(f"正在为查询检索相关文档: '{query}'")
        query_embedding = self.embedding_model.encode([query])
        
        top_k = self.rag_args['top_k']
        # 使用我们自定义的索引进行搜索
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        context = ""
        for i, idx in enumerate(indices[0]):
            context += f"[Reference Document {i+1} | Distance: {distances[0][i]:.4f}]\n{self.documents[idx]}\n\n"
        
        return context