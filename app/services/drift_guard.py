"""查询漂移防护 — 嵌入相似度检测

检测改写后 query 是否偏离原始语义：
- >= threshold（默认 0.65）: 改写可用
- moderate ~ threshold: 中度漂移，双 query 混合
- < moderate: 严重漂移，丢弃改写
"""

import math
from typing import Optional

from loguru import logger

from app.config import config
from app.models.rewrite import DriftCheckResult, DriftSeverity
from app.services.vector_embedding_service import vector_embedding_service


class DriftGuard:
    """基于余弦相似度的查询漂移检测"""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._threshold = config.rewrite_drift_threshold
        self._moderate_threshold = config.rewrite_drift_moderate_threshold

    async def check(
        self,
        original_query: str,
        rewritten_query: str,
    ) -> DriftCheckResult:
        """检测改写查询是否语义漂移

        Args:
            original_query: 用户原始查询
            rewritten_query: 改写后的查询

        Returns:
            DriftCheckResult: 漂移检测结果
        """
        if not self._enabled or not original_query or not rewritten_query:
            return DriftCheckResult(
                similarity=1.0,
                severity=DriftSeverity.NONE,
                rewritten_usable=True,
                action="use_rewritten",
            )

        if original_query == rewritten_query:
            return DriftCheckResult(
                similarity=1.0,
                severity=DriftSeverity.NONE,
                rewritten_usable=True,
                action="use_rewritten",
            )

        try:
            query_pair = [original_query, rewritten_query]
            embeddings = vector_embedding_service.embed_documents(query_pair)
            similarity = _cosine_similarity(embeddings[0], embeddings[1])

            severity, rewritten_usable, action = self._classify(similarity, original_query, rewritten_query)

            logger.info(
                f"[DriftGuard] similarity={similarity:.3f}, "
                f"severity={severity.value}, action={action}"
            )
            return DriftCheckResult(
                similarity=round(similarity, 4),
                severity=severity,
                rewritten_usable=rewritten_usable,
                action=action,
            )
        except Exception as e:
            logger.warning(f"[DriftGuard] 嵌入计算失败: {e}，信任改写结果")
            return DriftCheckResult(
                similarity=0.0,
                severity=DriftSeverity.LOW,
                rewritten_usable=True,
                action="use_rewritten",
            )

    async def check_batch(
        self,
        original_query: str,
        rewritten_queries: list[str],
    ) -> list[DriftCheckResult]:
        """批量检测多个改写查询的漂移（用于 decompose 场景）

        使用批量嵌入一次 API 调用完成所有计算。
        """
        if not self._enabled or not rewritten_queries:
            return [
                DriftCheckResult(
                    similarity=1.0,
                    severity=DriftSeverity.NONE,
                    rewritten_usable=True,
                    action="use_rewritten",
                )
                for _ in rewritten_queries
            ]

        try:
            all_queries = [original_query] + rewritten_queries
            embeddings = vector_embedding_service.embed_documents(all_queries)
            original_emb = embeddings[0]

            results: list[DriftCheckResult] = []
            for i, emb in enumerate(embeddings[1:], start=1):
                similarity = _cosine_similarity(original_emb, emb)
                severity, rewritten_usable, action = self._classify(
                    similarity, original_query, rewritten_queries[i - 1]
                )
                results.append(DriftCheckResult(
                    similarity=round(similarity, 4),
                    severity=severity,
                    rewritten_usable=rewritten_usable,
                    action=action,
                ))
                logger.debug(
                    f"[DriftGuard] sub_query[{i - 1}] similarity={similarity:.3f}, action={action}"
                )
            return results
        except Exception as e:
            logger.warning(f"[DriftGuard] 批量嵌入失败: {e}，信任改写结果")
            return [
                DriftCheckResult(
                    similarity=0.0,
                    severity=DriftSeverity.LOW,
                    rewritten_usable=True,
                    action="use_rewritten",
                )
                for _ in rewritten_queries
            ]

    def _classify(
        self,
        similarity: float,
        original_query: str,
        rewritten_query: str,
    ) -> tuple[DriftSeverity, bool, str]:
        if similarity >= self._threshold:
            return DriftSeverity.LOW, True, "use_rewritten"
        elif similarity >= self._moderate_threshold:
            return DriftSeverity.MODERATE, True, "use_both"
        else:
            logger.warning(
                f"[DriftGuard] 严重漂移 detected: "
                f"'{original_query[:80]}' → '{rewritten_query[:80]}' (sim={similarity:.3f})"
            )
            return DriftSeverity.HIGH, False, "use_original"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


drift_guard = DriftGuard(enabled=config.rewrite_enabled)
