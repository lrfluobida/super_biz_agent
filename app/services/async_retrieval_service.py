"""异步检索服务 — 并行多 query 检索 + RRF 合并

用于 decompose 场景：多个子查询并行检索，结果 RRF 合并去重。
"""

import asyncio
from typing import Optional

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.hybrid_search_service import hybrid_search


async def parallel_decomposed_retrieval(
    sub_queries: list[str],
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    max_workers: int | None = None,
) -> list[Document]:
    """并行检索多个子查询，RRF 合并去重。

    Args:
        sub_queries: 分解后的子查询列表
        top_k: 最终返回文档数量
        use_hybrid: 是否启用混合检索
        max_workers: 最大并行数

    Returns:
        RRF 合并后的文档列表
    """
    if not sub_queries:
        return []

    if len(sub_queries) == 1:
        docs, _ = hybrid_search(sub_queries[0], top_k=top_k, use_hybrid=use_hybrid)
        return docs

    effective_top_k = top_k if top_k and top_k > 0 else config.rag_top_k
    effective_max_workers = max_workers or config.rewrite_parallel_max_workers

    if config.rewrite_parallel_retrieval_enabled and len(sub_queries) > 1:
        logger.info(f"[AsyncRetrieval] 并行检索 {len(sub_queries)} 个子查询, max_workers={effective_max_workers}")
        semaphore = asyncio.Semaphore(effective_max_workers)

        async def _retrieve_one(query: str) -> tuple[str, list[Document]]:
            async with semaphore:
                docs, _ = await asyncio.to_thread(
                    hybrid_search, query, effective_top_k * 2, use_hybrid
                )
                return query, docs

        results = await asyncio.gather(*[_retrieve_one(q) for q in sub_queries])
    else:
        logger.info(f"[AsyncRetrieval] 串行检索 {len(sub_queries)} 个子查询")
        results = []
        for q in sub_queries:
            docs, _ = hybrid_search(q, top_k=effective_top_k * 2, use_hybrid=use_hybrid)
            results.append((q, docs))

    # RRF 合并所有子查询结果
    merged = _rrf_merge_multi(results, k=config.hybrid_rrf_k, top_k=effective_top_k)
    logger.info(f"[AsyncRetrieval] 合并完成: {len(sub_queries)} 子查询 → {len(merged)} 文档")
    return merged


def _rrf_merge_multi(
    query_results: list[tuple[str, list[Document]]],
    k: int,
    top_k: int,
) -> list[Document]:
    """多路 RRF 合并（超过 2 路）

    每路结果独立贡献 RRF 分数，按总分降序取 top_k。
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    for _, docs in query_results:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata.get("_milvus_id", "")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    result: list[Document] = []
    for doc_id in sorted_ids:
        doc = docs_by_id[doc_id]
        doc.metadata["_rrf_score"] = round(scores[doc_id], 6)
        result.append(doc)

    return result
