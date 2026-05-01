import math
import logging
from collections import Counter
from typing import List, Dict

# 新增的依赖，用于BERTScore计算
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("QwenFinetune")

class CustomMetricsCalculator:
    """
    一个从零开始实现的、用于计算 ROUGE, BLEU 和 BERTScore 分数的类。
    """
    def __init__(self, device: str = None):
        logger.info("Initialized CustomMetricsCalculator.")
        # --- 新增: 初始化BERTScore所需的模型 ---
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading SentenceTransformer model for BERTScore on device: {self.device}...")
        # 我们使用一个轻量级的、支持中文的多语言模型
        # 第一次运行时会自动下载
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        logger.info("SentenceTransformer model loaded.")
        # ------------------------------------

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """从一个token列表中提取n-grams"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)

    def _calculate_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
        """计算 ROUGE-N (N=1, 2) 的 F1, Precision, Recall"""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        overlap_count = 0
        for ngram, count in pred_ngrams.items():
            overlap_count += min(count, ref_ngrams.get(ngram, 0))
            
        ref_total_count = sum(ref_ngrams.values())
        pred_total_count = sum(pred_ngrams.values())
        
        precision = overlap_count / pred_total_count if pred_total_count > 0 else 0.0
        recall = overlap_count / ref_total_count if ref_total_count > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"f": f1, "p": precision, "r": recall}

    def _calculate_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        """使用动态规划计算最长公共子序列 (LCS) 来计算 ROUGE-L"""
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        precision = lcs_length / m if m > 0 else 0.0
        recall = lcs_length / n if n > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"f": f1, "p": precision, "r": recall}
    
    def _calculate_bleu(self, predictions: List[List[str]], references: List[List[List[str]]], max_n: int = 4) -> float:
        """计算 BLEU 分数"""
        # ... (此部分代码不变) ...
        score = 0.0
        epsilon = 1e-12
        weights = [1.0 / max_n] * max_n
        
        for pred_tokens, ref_list_tokens in zip(predictions, references):
            len_pred = len(pred_tokens)
            len_ref = min([len(ref) for ref in ref_list_tokens], key=lambda x: abs(x - len_pred))
            
            if len_pred > len_ref:
                bp = 1.0
            else:
                bp = math.exp(1 - len_ref / (len_pred + epsilon))

            s = 0.0
            for n in range(1, max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                clipped_count = 0
                total_count = 0
                for ngram, count in pred_ngrams.items():
                    max_ref_count = 0
                    for ref_tokens in ref_list_tokens:
                        max_ref_count = max(max_ref_count, self._get_ngrams(ref_tokens, n).get(ngram, 0))
                    clipped_count += min(count, max_ref_count)
                    total_count += count
                
                pn = clipped_count / (total_count + epsilon)
                s += weights[n-1] * math.log(pn + epsilon)
            
            score += bp * math.exp(s)
            
        return score / len(predictions) if predictions else 0.0

    # --- 新增: BERTScore 计算方法 ---
    def _calculate_bertscore_single_pair(self, pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        """为单个句子对计算BERTScore"""
        # 1. 获取每个token的向量 (注意：这里为了教学目的简化了，直接用句子向量的思路来编码)
        # 官方实现会获取每个token的上下文向量，更复杂
        pred_embeddings = self.st_model.encode(pred_tokens, convert_to_tensor=True, device=self.device)
        ref_embeddings = self.st_model.encode(ref_tokens, convert_to_tensor=True, device=self.device)

        # 2. 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(pred_embeddings.cpu(), ref_embeddings.cpu())

        # 3. 计算 Precision, Recall, F1
        # Precision: 对于每个pred_token，在ref_tokens中找到最相似的token
        precision_scores = similarity_matrix.max(axis=1)
        avg_precision = precision_scores.mean()

        # Recall: 对于每个ref_token，在pred_tokens中找到最相似的token
        recall_scores = similarity_matrix.max(axis=0)
        avg_recall = recall_scores.mean()
        
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {"f": f1, "p": avg_precision, "r": avg_recall}
    # ---------------------------

    def compute(self, predictions: List[str], references: List[str]):
        """
        主计算函数，接收字符串列表并返回所有指标。
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must be the same.")

        pred_tokens_list = [p.split() for p in predictions]
        ref_tokens_list = [r.split() for r in references]

        # --- 计算 ROUGE ---
        rouge1_scores = [self._calculate_rouge_n(p, r, 1) for p, r in zip(pred_tokens_list, ref_tokens_list)]
        rouge2_scores = [self._calculate_rouge_n(p, r, 2) for p, r in zip(pred_tokens_list, ref_tokens_list)]
        rougel_scores = [self._calculate_rouge_l(p, r) for p, r in zip(pred_tokens_list, ref_tokens_list)]
        
        avg_rouge1 = sum(s['f'] for s in rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(s['f'] for s in rouge2_scores) / len(rouge2_scores)
        avg_rougel = sum(s['f'] for s in rougel_scores) / len(rougel_scores)

        # --- 计算 BLEU ---
        bleu_ref_tokens_list = [[r] for r in ref_tokens_list]
        bleu_score = self._calculate_bleu(pred_tokens_list, bleu_ref_tokens_list)

        # --- 新增: 计算 BERTScore ---
        bertscore_scores = [self._calculate_bertscore_single_pair(p, r) for p, r in zip(pred_tokens_list, ref_tokens_list)]
        avg_bertscore_f1 = sum(s['f'] for s in bertscore_scores) / len(bertscore_scores)
        # ---------------------------

        return {
            "rouge": {
                "rouge1": avg_rouge1,
                "rouge2": avg_rouge2,
                "rougeL": avg_rougel
            },
            "bleu": {
                "bleu": bleu_score
            },
            "bertscore": {
                "f1": avg_bertscore_f1
            }
            # ------------------------
        }