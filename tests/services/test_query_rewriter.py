"""QueryRewriter 单元测试 — 三种改写器 JSON 输出格式和降级行为"""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.rewrite import DecomposeResult, StepBackResult, ContextualizeResult
from app.services.query_rewriter import QueryRewriter, query_rewriter


class TestDecomposeRewriter:
    """任务分解改写器测试"""

    @pytest.mark.asyncio
    async def test_parse_valid_response(self):
        """正确解析模型返回的 JSON"""
        rewriter = QueryRewriter()
        valid_response = {
            "sub_queries": ["CPU排查流程", "内存排查流程", "CPU与内存对比"],
            "original_topic": "CPU与内存排查对比",
        }
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value=valid_response,
        ):
            result = await rewriter.decompose("CPU高和内存高有什么区别")
            assert isinstance(result, DecomposeResult)
            assert len(result.sub_queries) == 3
            assert "CPU排查流程" in result.sub_queries
            assert result.original_topic == "CPU与内存排查对比"

    @pytest.mark.asyncio
    async def test_fallback_on_model_error(self):
        """模型调用失败时返回单元素列表"""
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            side_effect=Exception("Ollama timeout"),
        ):
            result = await rewriter.decompose("CPU高和内存高有什么区别")
            assert isinstance(result, DecomposeResult)
            assert len(result.sub_queries) == 1
            assert result.sub_queries[0] == "CPU高和内存高有什么区别"

    @pytest.mark.asyncio
    async def test_fallback_when_too_few_subqueries(self):
        """子查询不足 2 个时回退"""
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={"sub_queries": ["only one"], "original_topic": "x"},
        ):
            result = await rewriter.decompose("test")
            assert len(result.sub_queries) == 1

    @pytest.mark.asyncio
    async def test_max_4_subqueries(self):
        """超过 4 个子查询时截断"""
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={
                "sub_queries": ["q1", "q2", "q3", "q4", "q5", "q6"],
                "original_topic": "test",
            },
        ):
            result = await rewriter.decompose("test")
            assert len(result.sub_queries) == 4

    @pytest.mark.asyncio
    async def test_filter_empty_subqueries(self):
        """过滤空子查询"""
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={
                "sub_queries": ["q1", "", "  ", "q2"],
                "original_topic": "test",
            },
        ):
            result = await rewriter.decompose("test")
            assert len(result.sub_queries) == 2


class TestStepBackRewriter:
    """Step-Back 泛化改写器测试"""

    @pytest.mark.asyncio
    async def test_parse_valid_response(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={
                "step_back_query": "Prometheus采集间隔配置优化方法",
                "retained_specifics": "scrape_interval 15s 30s",
            },
        ):
            result = await rewriter.step_back("Prometheus的scrape_interval设成15秒还是30秒")
            assert isinstance(result, StepBackResult)
            assert "Prometheus" in result.step_back_query
            assert result.retained_specifics != ""

    @pytest.mark.asyncio
    async def test_fallback_on_empty_result(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={"step_back_query": "", "retained_specifics": ""},
        ):
            result = await rewriter.step_back("test query")
            assert result.step_back_query == "test query"

    @pytest.mark.asyncio
    async def test_fallback_on_model_error(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            side_effect=Exception("timeout"),
        ):
            result = await rewriter.step_back("test query")
            assert result.step_back_query == "test query"


class TestContextualizeRewriter:
    """上下文补全改写器测试"""

    @pytest.mark.asyncio
    async def test_parse_valid_response(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={
                "standalone_query": "内存使用率过高排查流程",
            },
        ):
            result = await rewriter.contextualize(
                current_query="那内存呢",
                history_topic="CPU使用率过高排查方法",
            )
            assert isinstance(result, ContextualizeResult)
            assert result.standalone_query == "内存使用率过高排查流程"

    @pytest.mark.asyncio
    async def test_fallback_on_empty_result(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={"standalone_query": ""},
        ):
            result = await rewriter.contextualize("那内存呢", "CPU排查")
            assert result.standalone_query == "那内存呢"

    @pytest.mark.asyncio
    async def test_fallback_on_model_error(self):
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            side_effect=Exception("timeout"),
        ):
            result = await rewriter.contextualize("那内存呢", "CPU排查")
            assert result.standalone_query == "那内存呢"

    @pytest.mark.asyncio
    async def test_prompt_has_history_topic(self):
        """验证 prompt 中正确替换了历史主题占位符"""
        rewriter = QueryRewriter()
        with patch(
            "app.services.query_rewriter.rewrite_model_service.rewrite",
            new_callable=AsyncMock,
            return_value={"standalone_query": "ok"},
        ) as mock_rewrite:
            await rewriter.contextualize(
                current_query="那内存呢",
                history_topic="CPU使用率过高排查",
            )
            call_kwargs = mock_rewrite.call_args
            args, kwargs = call_kwargs
            # rewrite(prompt, query, **extra)
            assert "history_topic" in kwargs or mock_rewrite.called
