"""QueryRewritePipeline 集成测试 — 端到端改写管线"""

from unittest.mock import MagicMock, patch

import pytest

from app.models.rewrite import (
    DecomposeResult,
    Intent,
    RewritePipelineResult,
    RouterResult,
    RouterSource,
    StepBackResult,
    ContextualizeResult,
    DriftCheckResult,
    DriftSeverity,
)
from app.services.query_rewrite_pipeline import (
    QueryRewritePipeline,
    _build_decompose_question,
    _build_stepback_question,
    _build_decompose_system_prompt,
)


class TestAgentQuestionBuilders:
    """agent_question 构造函数测试"""

    def test_decompose_question_structure(self):
        q = _build_decompose_question(
            "CPU和内存有什么区别",
            ["CPU排查流程", "内存排查流程"],
        )
        assert "CPU和内存有什么区别" in q
        assert "1. CPU排查流程" in q
        assert "2. 内存排查流程" in q

    def test_stepback_question_structure(self):
        q = _build_stepback_question(
            "针对level=70的阈值建议",
            "告警阈值配置优化",
        )
        assert "针对level=70的阈值建议" in q
        assert "告警阈值配置优化" in q
        assert "通用角度" in q
        assert "具体角度" in q

    def test_decompose_system_prompt(self):
        prompt = _build_decompose_system_prompt(3)
        assert "3 个子主题" in prompt
        assert "STEP 1" in prompt
        assert "STEP 2" in prompt
        assert "STEP 3" in prompt
        assert "至少 3 次" in prompt
        assert "严格禁止" in prompt


class TestPipelineDisabled:
    """管线禁用/空查询测试"""

    @pytest.mark.asyncio
    async def test_disabled_returns_passthrough(self):
        pipeline = QueryRewritePipeline()
        pipeline._enabled = False
        mock_memory = MagicMock()
        result = await pipeline.process("test query", "session-1", mock_memory)
        assert isinstance(result, RewritePipelineResult)
        assert result.agent_question == "test query"
        assert result.original_query == "test query"
        assert result.rewritten is False
        assert result.intent == Intent.DIRECT

    @pytest.mark.asyncio
    async def test_empty_query_returns_passthrough(self):
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        result = await pipeline.process("", "session-1", mock_memory)
        assert result.agent_question == ""
        assert result.rewritten is False


class TestPipelineDirect:
    """direct 意图测试"""

    @pytest.mark.asyncio
    async def test_direct_rule_hit_passthrough(self):
        """规则命中 direct → 原样透传"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        mock_memory.get_message_count.return_value = 0
        mock_memory.get_session_history.return_value = []

        result = await pipeline.process("CPU过高怎么排查", "session-1", mock_memory)
        assert result.intent == Intent.DIRECT
        assert result.agent_question == "CPU过高怎么排查"
        assert result.rewritten is False


class TestPipelineContextualize:
    """contextualize 意图测试"""

    @pytest.mark.asyncio
    async def test_contextualize_with_drift_ok(self):
        """指代补全 + 低漂移 → 改写生效"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        # 模拟第二轮的会话
        mock_memory.get_message_count.return_value = 2  # 1 user + 1 assistant = round 2
        mock_memory.get_session_history.return_value = [
            {"role": "user", "content": "CPU使用率过高排查方法", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "排查方法包括...", "timestamp": "2026-01-01T00:00:01"},
        ]

        with patch(
            "app.services.query_rewrite_pipeline.query_rewriter.contextualize",
            return_value=ContextualizeResult(standalone_query="内存使用率过高排查流程"),
        ) as mock_contextualize:
            with patch(
                "app.services.query_rewrite_pipeline.drift_guard.check",
                return_value=DriftCheckResult(
                    similarity=0.85,
                    severity=DriftSeverity.LOW,
                    rewritten_usable=True,
                    action="use_rewritten",
                ),
            ):
                result = await pipeline.process("那内存呢", "session-1", mock_memory)

        assert result.intent == Intent.CONTEXTUALIZE
        assert result.agent_question == "内存使用率过高排查流程"
        assert result.rewritten is True
        assert result.extra_system_prompt != ""

    @pytest.mark.asyncio
    async def test_contextualize_high_drift_fallback(self):
        """指代补全但严重漂移 → 回退原始查询"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        mock_memory.get_message_count.return_value = 2
        mock_memory.get_session_history.return_value = [
            {"role": "user", "content": "CPU排查", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "ok", "timestamp": "2026-01-01T00:00:01"},
        ]

        with patch(
            "app.services.query_rewrite_pipeline.query_rewriter.contextualize",
            return_value=ContextualizeResult(standalone_query="磁盘完全无关的内容"),
        ):
            with patch(
                "app.services.query_rewrite_pipeline.drift_guard.check",
                return_value=DriftCheckResult(
                    similarity=0.25,
                    severity=DriftSeverity.HIGH,
                    rewritten_usable=False,
                    action="use_original",
                ),
            ):
                result = await pipeline.process("那内存呢", "session-1", mock_memory)

        assert result.agent_question == "那内存呢"  # 回退到原始
        assert result.rewritten is False


class TestPipelineDecompose:
    """decompose 意图测试"""

    @pytest.mark.asyncio
    async def test_decompose_with_drift_filtering(self):
        """分解 + 部分子查询漂移 → 只保留有效子查询"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        mock_memory.get_message_count.return_value = 0

        with patch(
            "app.services.query_rewrite_pipeline.query_rewriter.decompose",
            return_value=DecomposeResult(
                sub_queries=["CPU排查", "内存排查", "完全无关"],
                original_topic="CPU与内存对比",
            ),
        ):
            with patch(
                "app.services.query_rewrite_pipeline.drift_guard.check_batch",
                return_value=[
                    DriftCheckResult(similarity=0.80, severity=DriftSeverity.LOW, rewritten_usable=True, action="use_rewritten"),
                    DriftCheckResult(similarity=0.75, severity=DriftSeverity.LOW, rewritten_usable=True, action="use_rewritten"),
                    DriftCheckResult(similarity=0.15, severity=DriftSeverity.HIGH, rewritten_usable=False, action="use_original"),
                ],
            ):
                result = await pipeline.process("CPU和内存有什么区别", "session-1", mock_memory)

        assert result.intent == Intent.DECOMPOSE
        assert result.rewritten is True
        # 第 3 个子查询被 drift 过滤
        assert "CPU排查" in result.agent_question
        assert "内存排查" in result.agent_question
        assert "完全无关" not in result.agent_question
        # 有 system prompt 增强
        assert result.extra_system_prompt != ""
        assert "STEP 1" in result.extra_system_prompt


class TestPipelineStepBack:
    """step_back 意图测试"""

    @pytest.mark.asyncio
    async def test_stepback_with_low_drift(self):
        """泛化 + 低漂移 → 改写生效"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        mock_memory.get_message_count.return_value = 0

        with patch(
            "app.services.query_rewrite_pipeline.query_rewriter.step_back",
            return_value=StepBackResult(
                step_back_query="告警阈值配置优化方法",
                retained_specifics="level=70",
            ),
        ):
            with patch(
                "app.services.query_rewrite_pipeline.drift_guard.check",
                return_value=DriftCheckResult(
                    similarity=0.70,
                    severity=DriftSeverity.LOW,
                    rewritten_usable=True,
                    action="use_rewritten",
                ),
            ):
                result = await pipeline.process("针对level=70的阈值设置建议", "session-1", mock_memory)

        assert result.intent == Intent.STEP_BACK
        assert result.rewritten is True
        assert "告警阈值配置优化方法" in result.agent_question
        assert "level=70" in result.agent_question
        assert result.extra_system_prompt != ""


class TestPipelineDegradation:
    """异常降级测试"""

    @pytest.mark.asyncio
    async def test_pipeline_exception_falls_back_to_direct(self):
        """整个 pipeline 异常 → fallback direct"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()

        # query_router.route 内部会调用 _rule_route，让 memory 抛异常
        mock_memory.get_message_count.side_effect = Exception("DB error")

        result = await pipeline.process("CPU高怎么排查", "session-1", mock_memory)
        assert result.intent == Intent.DIRECT
        assert result.agent_question == "CPU高怎么排查"
        assert result.rewritten is False

    @pytest.mark.asyncio
    async def test_empty_agent_question_fallback(self):
        """改写后 agent_question 为空 → 回退原始查询"""
        pipeline = QueryRewritePipeline()
        mock_memory = MagicMock()
        mock_memory.get_message_count.return_value = 2
        mock_memory.get_session_history.return_value = [
            {"role": "user", "content": "CPU排查", "timestamp": "x"},
            {"role": "assistant", "content": "ok", "timestamp": "y"},
        ]

        with patch(
            "app.services.query_rewrite_pipeline.query_rewriter.contextualize",
            return_value=ContextualizeResult(standalone_query=""),
        ):
            with patch(
                "app.services.query_rewrite_pipeline.drift_guard.check",
                return_value=DriftCheckResult(similarity=0.85, severity=DriftSeverity.LOW, rewritten_usable=True, action="use_rewritten"),
            ):
                result = await pipeline.process("那内存呢", "session-1", mock_memory)

        assert result.agent_question == "那内存呢"
        assert result.rewritten is False
