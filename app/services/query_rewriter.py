"""查询改写器 — Decompose / Step-Back / Contextualize"""

import json
from typing import Optional

from loguru import logger

from app.models.rewrite import (
    ContextualizeResult,
    DecomposeResult,
    StepBackResult,
)
from app.services.rewrite_model_service import rewrite_model_service

DECOMPOSE_PROMPT = """将以下复杂问题拆分为2-4个独立的检索子问题。
规则：
1. 每个子问题聚焦单一主题，使用具体关键词
2. 子问题之间无语义重叠
3. 按重要性排序
4. 只拆分纯检索主题，综合/对比等任务由后续步骤完成

输出严格JSON：{"sub_queries": ["...", "..."], "original_topic": "..."}"""

STEP_BACK_PROMPT = """将以下过于具体的问题泛化为通用检索查询。
规则：
1. 去除过于具体的数值/条件/特定名称
2. 提取问题核心类别/领域
3. 保持领域语义不变

输出严格JSON：{"step_back_query": "...", "retained_specifics": "..."}"""

CONTEXTUALIZE_PROMPT = """你是一个查询改写工具。你的唯一任务是将对话消息转换为独立完整的检索句子。你只输出JSON，不说其他话。
对话历史：用户之前问了关于<<HISTORY_TOPIC>>的问题。
当前用户消息：<<CURRENT_QUERY>>
规则：将代词替换为具体实体，补全省略的主语/宾语。
输出JSON（只输出这个，不要解释）：{"standalone_query": "补全后的完整检索句"}"""


class QueryRewriter:
    """查询改写器集合"""

    async def decompose(self, query: str) -> DecomposeResult:
        """将复合问题拆分为独立子查询"""
        try:
            result = await rewrite_model_service.rewrite(DECOMPOSE_PROMPT, query)
            sub_queries = result.get("sub_queries", [])
            original_topic = result.get("original_topic", "")

            if not isinstance(sub_queries, list) or len(sub_queries) < 2:
                logger.warning(f"[Decompose] 拆分结果不足，返回原始查询")
                return DecomposeResult(
                    sub_queries=[query],
                    original_topic=query,
                )

            # 过滤空子查询，保留最多 4 个
            sub_queries = [q.strip() for q in sub_queries if q and q.strip()][:4]
            if len(sub_queries) < 2:
                sub_queries = [query]

            logger.info(f"[Decompose] 拆分为 {len(sub_queries)} 个子查询: {sub_queries}")
            return DecomposeResult(
                sub_queries=sub_queries,
                original_topic=original_topic or query,
            )
        except Exception as e:
            logger.warning(f"[Decompose] 改写失败: {e}")
            return DecomposeResult(sub_queries=[query], original_topic=query)

    async def step_back(self, query: str) -> StepBackResult:
        """泛化过度具体的问题"""
        try:
            result = await rewrite_model_service.rewrite(STEP_BACK_PROMPT, query)
            step_back_query = result.get("step_back_query", "")
            retained_specifics = result.get("retained_specifics", "")

            if not step_back_query or not step_back_query.strip():
                logger.warning("[StepBack] 泛化结果为空，返回原始查询")
                return StepBackResult(step_back_query=query, retained_specifics="")

            logger.info(f"[StepBack] 泛化: '{query}' → '{step_back_query}'")
            return StepBackResult(
                step_back_query=step_back_query.strip(),
                retained_specifics=retained_specifics.strip() if retained_specifics else "",
            )
        except Exception as e:
            logger.warning(f"[StepBack] 改写失败: {e}")
            return StepBackResult(step_back_query=query)

    async def contextualize(
        self,
        current_query: str,
        history_topic: str,
    ) -> ContextualizeResult:
        """将多轮对话中的省略/指代查询补全为独立完整语句

        Args:
            current_query: 用户当前消息
            history_topic: 对话历史主题（来自最近的 user/assistant 消息摘要）

        Returns:
            ContextualizeResult: 补全后的独立查询
        """
        try:
            result = await rewrite_model_service.rewrite(
                CONTEXTUALIZE_PROMPT,
                current_query,
                history_topic=history_topic,
                current_query=current_query,
            )
            standalone_query = result.get("standalone_query", "")

            if not standalone_query or not standalone_query.strip():
                logger.warning("[Contextualize] 补全结果为空，返回原始查询")
                return ContextualizeResult(standalone_query=current_query)

            logger.info(f"[Contextualize] 补全: '{current_query}' → '{standalone_query}'")
            return ContextualizeResult(standalone_query=standalone_query.strip())
        except Exception as e:
            logger.warning(f"[Contextualize] 改写失败: {e}")
            return ContextualizeResult(standalone_query=current_query)


query_rewriter = QueryRewriter()
