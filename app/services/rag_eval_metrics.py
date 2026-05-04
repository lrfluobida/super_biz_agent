"""RAG 评测辅助指标计算。"""

from __future__ import annotations

from typing import Iterable


def normalize_doc_source(source: str) -> str:
    """将文档来源统一归一化，便于评测集与检索结果对齐。"""
    normalized = source.replace("\\", "/").strip().lower()
    marker = "aiops-docs/"
    marker_index = normalized.rfind(marker)
    if marker_index != -1:
        return normalized[marker_index:]
    return normalized


def doc_source_variants(source: str) -> set[str]:
    """为同一文档来源生成可匹配的多个归一化形式。"""
    normalized = normalize_doc_source(source)
    variants = {normalized}
    if normalized:
        variants.add(normalized.rsplit("/", 1)[-1])
    return variants


def find_first_relevant_rank(gold_docs: Iterable[str], retrieved_docs: Iterable[dict]) -> int | None:
    """返回首个命中文档的排名，未命中则返回 None。"""
    normalized_gold_docs = set()
    for item in gold_docs:
        normalized_gold_docs.update(doc_source_variants(item))

    for index, doc in enumerate(retrieved_docs, start=1):
        doc_variants = set()
        doc_variants.update(doc_source_variants(str(doc.get("source", ""))))
        doc_variants.update(doc_source_variants(str(doc.get("file_name", ""))))
        if normalized_gold_docs & doc_variants:
            return index
    return None


def hit_at_k(rank: int | None, k: int) -> int:
    """命中则返回 1，否则返回 0。"""
    return int(rank is not None and rank <= k)


def reciprocal_rank(rank: int | None) -> float:
    """计算 reciprocal rank。"""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank
