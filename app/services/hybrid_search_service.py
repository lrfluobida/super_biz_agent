"""混合检索服务 - Dense Vector + BM25 Sparse 二路召回 + RRF 合并

Python 侧执行 RRF 合并，支持追踪每条结果的召回路径（dense/keyword/both）。
"""

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from loguru import logger
from pymilvus import Collection

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.keyword_search_service import keyword_search_service
from app.services.vector_embedding_service import vector_embedding_service


def hybrid_search(
    query: str,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
) -> Tuple[List[Document], Dict]:
    """执行二路混合检索 (Dense + Sparse)，Python 侧 RRF 合并。

    Args:
        query: 查询文本
        top_k: 最终返回数量，默认使用配置的 rag_top_k
        use_hybrid: 显式控制是否启用混合检索，None 时使用配置值

    Returns:
        (docs, recall_meta): docs 为 LangChain Document 列表，
        recall_meta 包含两路召回详情用于 trace
    """
    effective_top_k = top_k if top_k and top_k > 0 else config.rag_top_k
    per_ranker_limit = config.hybrid_per_ranker_limit
    enable_hybrid = use_hybrid if use_hybrid is not None else config.hybrid_search_enabled
    recall_meta: Dict = {"mode": "dense_only", "dense_ids": [], "keyword_ids": []}

    if not enable_hybrid:
        docs = _dense_search(query, effective_top_k)
        recall_meta["dense_ids"] = [d.metadata.get("_milvus_id", "") for d in docs]
        for doc in docs:
            doc.metadata["_recall_path"] = "dense"
        return docs, recall_meta

    if not keyword_search_service.is_fitted:
        logger.warning("BM25 模型未训练，回退到纯向量检索")
        docs = _dense_search(query, effective_top_k)
        recall_meta["dense_ids"] = [d.metadata.get("_milvus_id", "") for d in docs]
        for doc in docs:
            doc.metadata["_recall_path"] = "dense"
        return docs, recall_meta

    try:
        # 1. 并行执行两路独立检索
        dense_docs = _dense_search(query, per_ranker_limit)
        sparse_docs = _sparse_search(query, per_ranker_limit)

        recall_meta["dense_ids"] = [d.metadata.get("_milvus_id", "") for d in dense_docs]
        recall_meta["keyword_ids"] = [d.metadata.get("_milvus_id", "") for d in sparse_docs]

        # 2. Python 侧 RRF 合并
        merged_docs = _rrf_merge(
            dense_docs,
            sparse_docs,
            k=config.hybrid_rrf_k,
            top_k=effective_top_k,
        )

        recall_meta["mode"] = "hybrid"
        recall_meta["merged_count"] = len(merged_docs)
        recall_meta["dense_count"] = len(dense_docs)
        recall_meta["keyword_count"] = len(sparse_docs)

        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"dense={len(dense_docs)}, keyword={len(sparse_docs)}, "
            f"merged={len(merged_docs)}"
        )
        return merged_docs, recall_meta

    except Exception as e:
        logger.error(f"混合检索失败，回退到纯向量检索: {e}")
        docs = _dense_search(query, effective_top_k)
        recall_meta["mode"] = "dense_fallback"
        recall_meta["dense_ids"] = [d.metadata.get("_milvus_id", "") for d in docs]
        for doc in docs:
            doc.metadata["_recall_path"] = "dense"
        return docs, recall_meta


def _dense_search(query: str, limit: int) -> List[Document]:
    """纯向量检索"""
    query_vector = vector_embedding_service.embed_query(query)
    collection: Collection = milvus_manager.get_collection()
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=["id", "content", "metadata"],
    )

    docs: List[Document] = []
    for hits in results:
        for hit in hits:
            metadata = hit.entity.get("metadata", {}) or {}
            metadata["_milvus_id"] = hit.entity.get("id", "")
            metadata["_dense_score"] = hit.distance
            docs.append(
                Document(
                    page_content=hit.entity.get("content", ""),
                    metadata=metadata,
                )
            )
    return docs


def _sparse_search(query: str, limit: int) -> List[Document]:
    """BM25 稀疏向量检索"""
    sparse_vec = keyword_search_service.encode_query(query)
    if not sparse_vec:
        return []

    collection: Collection = milvus_manager.get_collection()
    results = collection.search(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        limit=limit,
        output_fields=["id", "content", "metadata"],
    )

    docs: List[Document] = []
    for hits in results:
        for hit in hits:
            metadata = hit.entity.get("metadata", {}) or {}
            metadata["_milvus_id"] = hit.entity.get("id", "")
            metadata["_sparse_score"] = hit.distance
            docs.append(
                Document(
                    page_content=hit.entity.get("content", ""),
                    metadata=metadata,
                )
            )
    return docs


def _rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int,
    top_k: int,
) -> List[Document]:
    """RRF 合并两路结果，标记每条文档的召回路径"""
    scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Document] = {}
    dense_ids: set[str] = set()
    sparse_ids: set[str] = set()

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = doc.metadata.get("_milvus_id", "")
        if not doc_id:
            continue
        dense_ids.add(doc_id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs_by_id[doc_id] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        doc_id = doc.metadata.get("_milvus_id", "")
        if not doc_id:
            continue
        sparse_ids.add(doc_id)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = doc

    # 按 RRF score 降序取 top_k
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    result: List[Document] = []
    for doc_id in sorted_ids:
        doc = docs_by_id[doc_id]
        # 标记召回路径
        in_dense = doc_id in dense_ids
        in_sparse = doc_id in sparse_ids
        if in_dense and in_sparse:
            doc.metadata["_recall_path"] = "both"
        elif in_dense:
            doc.metadata["_recall_path"] = "dense"
        else:
            doc.metadata["_recall_path"] = "keyword"
        doc.metadata["_rrf_score"] = round(scores[doc_id], 6)
        result.append(doc)

    return result
