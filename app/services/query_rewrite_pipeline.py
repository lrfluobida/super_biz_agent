"""查询改写编排器 — route → rewrite → drift → agent_question

Normal Mode 与 Eval Mode 共用改写管线，差异仅在最后的输出动作。
"""

from textwrap import dedent
from typing import Optional

from loguru import logger

from app.config import config
from app.models.rewrite import (
    DriftCheckResult,
    DriftSeverity,
    Intent,
    RewritePipelineResult,
    RouterResult,
    RouterSource,
)
from app.services.drift_guard import drift_guard
from app.services.query_rewriter import query_rewriter
from app.services.query_router import query_router


class QueryRewritePipeline:
    """查询改写编排器"""

    def __init__(self) -> None:
        self._enabled = config.rewrite_enabled

    async def process(
        self,
        query: str,
        session_id: str,
        memory_manager,
    ) -> RewritePipelineResult:
        """执行改写管线

        Args:
            query: 用户原始查询
            session_id: 会话 ID
            memory_manager: MemoryManager 实例，用于获取会话轮次和历史主题

        Returns:
            RewritePipelineResult: 包含改写后问题和元信息
        """
        if not self._enabled or not query or not query.strip():
            return RewritePipelineResult(
                agent_question=query,
                original_query=query,
                intent=Intent.DIRECT,
                rewritten=False,
            )

        # 获取会话轮次（用于路由决策）
        session_round = self._get_session_round(session_id, memory_manager)

        # Step 1: 路由
        router_result = await query_router.route(query, session_round)
        intent = router_result.intent

        # Step 2: 改写
        rewrite_meta: dict = {"intent": intent.value, "source": router_result.source.value}
        drift_result: Optional[DriftCheckResult] = None
        extra_system_prompt = ""
        agent_question = query
        rewritten = False

        if intent == Intent.DIRECT:
            # 直接透传，无需改写
            pass

        elif intent == Intent.CONTEXTUALIZE:
            history_topic = self._get_history_topic(session_id, memory_manager)
            ctx_result = await query_rewriter.contextualize(query, history_topic)
            agent_question = ctx_result.standalone_query
            rewritten = agent_question != query
            rewrite_meta["contextualize"] = ctx_result.model_dump()

            if rewritten:
                drift_result = await drift_guard.check(query, agent_question)
                if drift_result.severity == DriftSeverity.HIGH:
                    agent_question = query
                    rewritten = False
                extra_system_prompt = dedent("""
                    [系统提示] 用户使用了代词或省略表达，系统已自动将其补全为完整检索语句。
                    当前检索语句已经完整，可直接用于检索。
                """).strip()

        elif intent == Intent.DECOMPOSE:
            decomp_result = await query_rewriter.decompose(query)
            sub_queries = decomp_result.sub_queries
            rewrite_meta["decompose"] = decomp_result.model_dump()

            if len(sub_queries) > 1:
                # 批量 drift check
                drift_results = await drift_guard.check_batch(query, sub_queries)
                # 过滤严重漂移 + 伪分解（与原问题相似度 > 0.95 = 换个说法而已）
                _pseudo_threshold = 0.95
                valid_sub_queries: list[str] = []
                for sq, dr in zip(sub_queries, drift_results):
                    if dr.severity == DriftSeverity.HIGH:
                        logger.info(f"[Pipeline] 子查询漂移剔除: '{sq[:50]}' (sim={dr.similarity:.3f})")
                        continue
                    if dr.similarity >= _pseudo_threshold:
                        logger.info(f"[Pipeline] 伪分解剔除: '{sq[:50]}' (sim={dr.similarity:.3f} >= {_pseudo_threshold})")
                        continue
                    valid_sub_queries.append(sq)
                # 如果全部漂移/伪分解，回退到原始查询
                if not valid_sub_queries:
                    logger.warning("[Pipeline] 所有子查询被漂移/伪分解过滤，回退 direct")
                    agent_question = query
                    rewritten = False
                else:
                    rewritten = True
                    agent_question = _build_decompose_question(query, valid_sub_queries)
                    extra_system_prompt = _build_decompose_system_prompt(len(valid_sub_queries))
                    drift_result = drift_results[0] if drift_results else None
                    rewrite_meta["valid_sub_queries"] = valid_sub_queries
                    rewrite_meta["drift_results"] = [d.model_dump() for d in drift_results]
            else:
                # 只有一个子查询（模型可能没拆分成功），降级为 direct
                agent_question = sub_queries[0] if sub_queries else query

        elif intent == Intent.STEP_BACK:
            sb_result = await query_rewriter.step_back(query)
            step_back_query = sb_result.step_back_query
            rewrite_meta["step_back"] = sb_result.model_dump()

            if step_back_query and step_back_query != query:
                drift_result = await drift_guard.check(query, step_back_query)
                if drift_result.severity == DriftSeverity.HIGH:
                    agent_question = query
                else:
                    rewritten = True
                    agent_question = _build_stepback_question(query, step_back_query)
                    extra_system_prompt = dedent("""
                        [系统提示] 当前问题提供了泛化和具体两个视角。
                        请从通用知识 + 具体方案两个层面检索和回答。
                    """).strip()
            else:
                agent_question = query

        # 如果 agent_question 为空，回退
        if not agent_question or not agent_question.strip():
            agent_question = query
            rewritten = False

        logger.info(
            f"[Pipeline] 改写完成: intent={intent.value}, "
            f"rewritten={rewritten}, "
            f"agent_question='{agent_question[:80]}{'...' if len(agent_question) > 80 else ''}'"
        )

        return RewritePipelineResult(
            agent_question=agent_question,
            original_query=query,
            intent=intent,
            router_result=router_result,
            rewritten=rewritten,
            rewrite_meta=rewrite_meta,
            drift_check=drift_result,
            extra_system_prompt=extra_system_prompt,
        )

    def _get_session_round(self, session_id: str, memory_manager) -> int:
        """获取当前会话轮次（user 消息计数 + 1 表示新一轮）"""
        try:
            count = memory_manager.get_message_count(session_id)
            # 每条消息对应一轮 user+assistant，所以轮次 = user 数 + 1
            return (count // 2) + 1
        except Exception:
            return 1

    def _get_history_topic(self, session_id: str, memory_manager) -> str:
        """从会话记忆中提取最近的对话主题"""
        try:
            history = memory_manager.get_session_history(session_id)
            if not history:
                return ""

            # 取最近的 user 和 assistant 消息
            recent_user = ""
            recent_assistant = ""
            for msg in reversed(history):
                if msg["role"] == "user" and not recent_user:
                    recent_user = msg["content"]
                elif msg["role"] == "assistant" and not recent_assistant:
                    recent_assistant = msg["content"]
                if recent_user and recent_assistant:
                    break

            topic = recent_user or ""
            # 限制长度，避免 prompt 过长
            if len(topic) > 200:
                topic = topic[:200]
            return topic
        except Exception:
            return ""


def _build_decompose_question(original_query: str, sub_queries: list[str]) -> str:
    sub_lines = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_queries, start=1))
    return dedent(f"""
        原始问题：{original_query}

        你需要分别检索以下子主题并综合对比：
        {sub_lines}
    """).strip()


def _build_stepback_question(original_query: str, step_back_query: str) -> str:
    return dedent(f"""
        原始问题：{original_query}

        请从以下两个角度检索：
        1. 通用角度：{step_back_query}
        2. 具体角度：{original_query}
    """).strip()


def _build_decompose_system_prompt(num_sub_queries: int) -> str:
    return dedent(f"""
        [系统指令 - 必须执行]
        检测到你的问题是复合问题，已被拆分为 {num_sub_queries} 个子主题。
        你必须严格遵守以下步骤：
        STEP 1: 对每个子主题分别调用一次 retrieve_knowledge 工具（至少 {num_sub_queries} 次）
        STEP 2: 确认所有子主题检索完成后，综合所有检索结果进行对比分析
        STEP 3: 基于综合结论生成答案

        严格禁止以下行为：
        - 仅检索一次就生成答案
        - 跳过任何子主题
        - 在未完成全部检索前就开始回答
    """).strip()


query_rewrite_pipeline = QueryRewritePipeline()
