# src/rag_pipeline.py

import os
import json
import logging
import faiss
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import torch
logger = logging.getLogger("QwenFinetune")

class RAGPipeline:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_args = config['rag_args']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("Initializing RAG pipeline...")
        self.embedding_model = SentenceTransformer(self.rag_args['embedding_model'], device=self.device)
        
        self.documents = []
        self.index = None
        
        self._build_or_load_index()

    def _load_documents(self) -> List[str]:
        """从jsonl文件中加载文档内容"""
        knowledge_file = self.rag_args['knowledge_base_file']
        logger.info(f"Loading documents from {knowledge_file}...")
        docs = []
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                question = data.get('data', {}).get('问', '')
                answer = data.get('data', {}).get('答', '')
                if question and answer:
                    docs.append(f"问：{question}\n答：{answer}")
        return docs

    def _build_or_load_index(self):
       
        index_path = self.rag_args['faiss_index_path']
        
        if os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            # 我们还需要加载原始文档
            self.documents = self._load_documents()
            logger.info("FAISS index and documents loaded successfully.")
        else:
            logger.info("No existing FAISS index found. Building a new one...")
            self.documents = self._load_documents()
            
            logger.info(f"Encoding {len(self.documents)} documents for the knowledge base...")
            embeddings = self.embedding_model.encode(
                self.documents, 
                show_progress_bar=True, 
                batch_size=128,
                convert_to_numpy=True
            )
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings, dtype=np.float32))
            
            # 保存索引以备后用
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS index built and saved to {index_path}")

    def retrieve(self, query: str) -> str:
       
        logger.info(f"Retrieving documents for query: '{query}'")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        top_k = self.rag_args['top_k']
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        context = ""
        for i, idx in enumerate(indices[0]):
            context += f"[Reference Document {i+1}]\n{self.documents[idx]}\n\n"
        
        return context

