"""BM25 关键词检索服务 - 基于 pymilvus BM25EmbeddingFunction"""

from pathlib import Path
from typing import List

from loguru import logger
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

from app.config import config


class BM25Encoder:
    """BM25 编码器，封装 BM25EmbeddingFunction 的训练、编码、持久化"""

    def __init__(self) -> None:
        self._analyzer = build_default_analyzer(language="zh")
        self._bm25: BM25EmbeddingFunction | None = None

    @property
    def is_fitted(self) -> bool:
        return self._bm25 is not None

    def fit(self, corpus: List[str]) -> None:
        """在语料上训练 BM25 模型"""
        if not corpus:
            logger.warning("BM25 fit: corpus 为空，跳过训练")
            return
        self._bm25 = BM25EmbeddingFunction(self._analyzer)
        self._bm25.fit(corpus)
        logger.info(f"BM25 模型训练完成，语料大小: {len(corpus)}")

    def encode_documents(self, texts: List[str]) -> List[dict]:
        """将文档编码为稀疏向量，返回 [{dim_idx: value}, ...]"""
        if self._bm25 is None:
            raise RuntimeError("BM25 模型未训练，请先调用 fit()")
        if not texts:
            return []
        sparse_mat = self._bm25.encode_documents(texts)
        return [_sparse_row_to_dict(sparse_mat, i) for i in range(sparse_mat.shape[0])]

    def encode_query(self, query: str) -> dict:
        """将查询编码为稀疏向量，返回 {dim_idx: value}"""
        if self._bm25 is None:
            raise RuntimeError("BM25 模型未训练，请先调用 fit()")
        sparse_mat = self._bm25.encode_queries([query])
        return _sparse_row_to_dict(sparse_mat, 0)

    def save(self, path: str) -> None:
        if self._bm25 is None:
            logger.warning("BM25 模型未训练，跳过保存")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._bm25.save(path)
        logger.info(f"BM25 模型已保存: {path}")

    def load(self, path: str) -> None:
        self._bm25 = BM25EmbeddingFunction(self._analyzer)
        self._bm25.load(path)
        logger.info(f"BM25 模型已加载: {path}")


def _sparse_row_to_dict(sparse_mat, row_idx: int) -> dict:
    """将 scipy 稀疏矩阵的一行转换为 {dim_index: value} 字典"""
    row = sparse_mat[row_idx].tocoo()
    return {int(col): float(val) for col, val in zip(row.col, row.data)}


# 全局单例
keyword_search_service = BM25Encoder()
