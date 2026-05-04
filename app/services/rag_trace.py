"""RAG 检索链路的轻量可观测性支持。"""

from __future__ import annotations

from contextvars import ContextVar, Token
from copy import deepcopy
from typing import Any

from langchain_core.documents import Document
from loguru import logger

_trace_state: ContextVar[dict[str, Any] | None] = ContextVar("rag_trace_state", default=None)


def begin_rag_trace(top_k: int | None = None) -> Token:
    """开启当前请求的 RAG trace。"""
    state = {
        "enabled": True,
        "requested_top_k": top_k,
        "retrieval_calls": [],
    }
    token = _trace_state.set(state)
    logger.debug(f"RAG trace 开启, token={token}, state_id={id(state)}")
    return token


def end_rag_trace(token: Token) -> None:
    """关闭当前请求的 RAG trace。"""
    _trace_state.reset(token)
    logger.debug(f"RAG trace 关闭, token={token}")


def get_effective_top_k(default_top_k: int) -> int:
    """返回当前 trace 请求覆盖的 top_k；如果未启用 trace，则回落到默认值。"""
    state = _trace_state.get()
    if not state:
        return default_top_k

    requested_top_k = state.get("requested_top_k")
    if isinstance(requested_top_k, int) and requested_top_k > 0:
        return requested_top_k
    return default_top_k


def record_retrieval(
    query: str,
    docs: list[Document],
    recall_meta: dict | None = None,
) -> None:
    """记录一次检索调用结果。未开启 trace 时静默忽略。"""
    state = _trace_state.get()
    if not state:
        logger.warning(f"record_retrieval: trace state 为空, query='{query[:60]}...'")
        return

    logger.info(f"record_retrieval: 记录 {len(docs)} 条检索结果, query='{query[:60]}...'")
    normalized_results = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = str(metadata.get("_source") or metadata.get("_file_name") or "未知来源")
        file_name = str(metadata.get("_file_name") or source.rsplit("/", 1)[-1])
        headers = [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
        preview = doc.page_content.replace("\r", " ").replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:180] + "..."

        normalized_results.append(
            {
                "rank": rank,
                "source": source,
                "file_name": file_name,
                "headers": headers,
                "preview": preview,
                "recall_path": metadata.get("_recall_path", "unknown"),
            }
        )

    call_record = {
        "query": query,
        "results": normalized_results,
    }
    if recall_meta:
        call_record["recall_meta"] = recall_meta

    state["retrieval_calls"].append(call_record)


def get_rag_trace_payload() -> dict[str, Any]:
    """获取当前请求的 trace 快照。"""
    state = _trace_state.get()
    if not state:
        return {
            "enabled": False,
            "requested_top_k": None,
            "retrieval_calls": [],
            "retrieved_docs": [],
        }

    retrieval_calls = deepcopy(state.get("retrieval_calls", []))
    aggregated_docs: dict[str, dict[str, Any]] = {}

    recall_summary: dict[str, Any] = {}

    for call in retrieval_calls:
        query = call.get("query", "")
        for item in call.get("results", []):
            source = item.get("source", "")
            existing = aggregated_docs.get(source)
            if existing is None:
                aggregated_docs[source] = {
                    **item,
                    "queries": [query] if query else [],
                }
                continue

            if item.get("rank", 9999) < existing.get("rank", 9999):
                existing["rank"] = item["rank"]
            if query and query not in existing["queries"]:
                existing["queries"].append(query)

        call_meta = call.get("recall_meta")
        if call_meta:
            recall_summary = call_meta

    retrieved_docs = sorted(aggregated_docs.values(), key=lambda item: item["rank"])

    path_counts: dict[str, int] = {}
    for doc in retrieved_docs:
        path = doc.get("recall_path", "unknown")
        path_counts[path] = path_counts.get(path, 0) + 1

    return {
        "enabled": True,
        "requested_top_k": state.get("requested_top_k"),
        "retrieval_calls": retrieval_calls,
        "retrieved_docs": retrieved_docs,
        "recall_meta": recall_summary,
        "recall_path_stats": path_counts,
    }
