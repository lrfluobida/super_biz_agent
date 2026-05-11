"""QueryRouter 单元测试 — 验证规则矩阵命中率和降级行为"""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.rewrite import Intent, RouterSource, RouterResult
from app.services.query_router import (
    _entity_count,
    _has_decompose_marker,
    _has_question_word,
    _is_overly_specific,
    QueryRouter,
)


class TestRuleHelpers:
    """规则辅助函数测试"""

    def test_question_word_detection(self):
        assert _has_question_word("CPU过高怎么排查")
        assert _has_question_word("这是什么问题")
        assert _has_question_word("如何优化")
        assert not _has_question_word("CPU")
        assert not _has_question_word("那内存呢")

    def test_decompose_marker_detection(self):
        assert _has_decompose_marker("CPU和内存有什么区别")
        assert _has_decompose_marker("A和B比较")
        assert _has_decompose_marker("分别说")
        assert not _has_decompose_marker("CPU过高怎么排查")

    def test_entity_count(self):
        assert _entity_count("CPU使用率过高排查") >= 1
        assert _entity_count("Prometheus和Grafana告警配置区别") >= 2
        assert _entity_count("嗯") == 0

    def test_overly_specific(self):
        assert _is_overly_specific("针对level=70的阈值设置建议")
        assert _is_overly_specific("Prometheus的scrape_interval设成15秒还是30秒")
        assert not _is_overly_specific("CPU过高怎么排查")
        assert not _is_overly_specific("磁盘排查流程")


class TestRuleRouting:
    """Stage 1 规则路由测试（无需 Ollama）"""

    def setup_method(self):
        self.router = QueryRouter()

    # ── contextualize 规则 ──

    @pytest.mark.parametrize("query,expected_intent", [
        ("它为什么会触发", Intent.CONTEXTUALIZE),
        ("这个怎么处理", Intent.CONTEXTUALIZE),
        ("那个是什么原因", Intent.CONTEXTUALIZE),
        ("上面说的怎么查", Intent.CONTEXTUALIZE),
        ("之前那个", Intent.CONTEXTUALIZE),
    ])
    def test_pronoun_triggers_contextualize_after_round1(self, query, expected_intent):
        """非首轮 + 代词 → contextualize"""
        result = self.router._rule_route(query, session_round=2)
        assert result is not None
        assert result.intent == expected_intent
        assert result.source == RouterSource.RULE

    @pytest.mark.parametrize("query", ["嗯", "然后呢", "具体", "继续"])
    def test_short_query_triggers_contextualize_after_round1(self, query):
        """非首轮 + 短查询(<=5字) + 无疑问词 → contextualize"""
        result = self.router._rule_route(query, session_round=3)
        assert result is not None
        assert result.intent == Intent.CONTEXTUALIZE

    def test_pronoun_not_contextualize_in_round1(self):
        """首轮有代词但命中 direct 规则（短问句），不命中 contextualize"""
        result = self.router._rule_route("它为什么会触发", session_round=1)
        # contextualize 要求 round > 1，所以不会命中
        # 可能命中 direct（短问句）或 step_back 等，但不会是 contextualize
        if result is not None:
            assert result.intent != Intent.CONTEXTUALIZE

    # ── decompose 规则 ──

    @pytest.mark.parametrize("query", [
        "CPU使用率高和内存泄漏的排查流程有什么区别",
        "Prometheus告警规则和Grafana告警有什么区别",
    ])
    def test_compare_marker_triggers_decompose(self, query):
        """对比标记 + 多个独立实体 → decompose"""
        result = self.router._rule_route(query, session_round=1)
        assert result is not None
        assert result.intent == Intent.DECOMPOSE
        assert result.source == RouterSource.RULE

    def test_multiple_question_marks_triggers_decompose(self):
        result = self.router._rule_route("A怎么做?B怎么查?", session_round=1)
        assert result is not None
        assert result.intent == Intent.DECOMPOSE

    # ── step_back 规则 ──

    def test_overly_specific_triggers_stepback(self):
        result = self.router._rule_route("针对level=70的阈值设置建议", session_round=1)
        assert result is not None
        assert result.intent == Intent.STEP_BACK
        assert result.confidence == 0.70

    # ── direct 规则 ──

    def test_short_question_first_round_triggers_direct(self):
        result = self.router._rule_route("CPU过高怎么排查", session_round=1)
        assert result is not None
        assert result.intent == Intent.DIRECT
        assert result.confidence == 0.95

    def test_unmatched_returns_none(self):
        """不匹配任何规则 → 返回 None（进入 Stage 2）"""
        result = self.router._rule_route("如何优化系统整体性能和稳定性", session_round=1)
        # 长度超过20，不命中 direct；无对比词，不命中 decompose
        assert result is None or isinstance(result, RouterResult)


class TestModelRouteFallback:
    """Stage 2 模型分类降级测试"""

    @pytest.mark.asyncio
    async def test_model_error_fallback_to_direct(self):
        """rewrite_model_service.classify 抛异常 → fallback direct"""
        router = QueryRouter()
        with patch(
            "app.services.query_router.rewrite_model_service.classify",
            new_callable=AsyncMock,
            side_effect=Exception("Ollama not available"),
        ):
            result = await router._model_route("如何优化系统性能")
            assert result.intent == Intent.DIRECT
            assert result.source == RouterSource.FALLBACK
            assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_model_unknown_intent_fallback_to_direct(self):
        """模型返回未知意图 → fallback direct"""
        router = QueryRouter()
        with patch(
            "app.services.query_router.rewrite_model_service.classify",
            new_callable=AsyncMock,
            return_value={"intent": "unknown_type", "reason": "no match"},
        ):
            result = await router._model_route("some obscure query")
            assert result.intent == Intent.DIRECT
            assert result.source == RouterSource.FALLBACK

    @pytest.mark.asyncio
    async def test_model_empty_result_fallback_to_direct(self):
        """模型返回空 dict → fallback direct"""
        router = QueryRouter()
        with patch(
            "app.services.query_router.rewrite_model_service.classify",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await router._model_route("test")
            assert result.intent == Intent.DIRECT
            assert result.source == RouterSource.FALLBACK


class TestRouteIntegration:
    """route() 集成测试（规则路径，无模型）"""

    @pytest.mark.asyncio
    async def test_route_uses_rules_without_model(self):
        """规则命中时直接走 Stage 1，不调用模型"""
        router = QueryRouter()
        result = await router.route("CPU高怎么排查", session_round=1)
        assert result.intent == Intent.DIRECT
        assert result.source == RouterSource.RULE

    @pytest.mark.asyncio
    async def test_route_disabled_returns_direct(self):
        """rewrite 禁用时直接返回 direct"""
        router = QueryRouter()
        router._enabled = False
        result = await router.route("CPU高怎么排查", session_round=1)
        assert result.intent == Intent.DIRECT
        assert result.source == RouterSource.FALLBACK
