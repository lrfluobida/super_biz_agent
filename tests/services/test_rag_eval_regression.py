import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

from loguru import logger as loguru_logger
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

os.environ["DEBUG"] = "false"

_original_logger_add = loguru_logger.add


def _patched_logger_add(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("logs/"):
        kwargs["enqueue"] = False
    return _original_logger_add(*args, **kwargs)


loguru_logger.add = _patched_logger_add

fake_tools_module = types.ModuleType("app.tools")
fake_tools_module.get_current_time = object()
fake_tools_module.retrieve_knowledge = object()
sys.modules["app.tools"] = fake_tools_module

fake_knowledge_tool_module = types.ModuleType("app.tools.knowledge_tool")
fake_knowledge_tool_module.retrieve_knowledge_documents = lambda query, top_k=None: []
fake_knowledge_tool_module.format_docs = lambda docs: "\n".join(
    f"【参考资料 {index}】\n来源: {doc.metadata.get('_file_name', '未知来源')}\n内容:\n{doc.page_content}\n"
    for index, doc in enumerate(docs, start=1)
)
sys.modules["app.tools.knowledge_tool"] = fake_knowledge_tool_module

from app.services.rag_agent_service import RagAgentService
from app.services.rag_trace import record_retrieval


class RagEvalRegressionTestCase(unittest.IsolatedAsyncioTestCase):
    def test_detect_eval_question_style_handles_compare_and_scenario(self):
        service = RagAgentService(streaming=False)

        self.assertEqual(
            service._detect_eval_question_style("CPU 使用率过高和服务响应变慢在排查重点上有什么区别？"),
            "compare",
        )
        self.assertEqual(
            service._detect_eval_question_style("如果缓存命中率突然下降、数据库查询量激增、响应时间同步变长，这更像什么问题？应该怎么处理？"),
            "scenario",
        )
        self.assertEqual(
            service._detect_eval_question_style("SlowResponse 告警通常在什么条件下触发，会造成哪些影响？"),
            "default",
        )

    async def test_eval_mode_uses_retrieval_before_answer_generation_for_compare_questions(self):
        service = RagAgentService(streaming=False)
        fake_docs = [
            Document(
                page_content="CPU 高更关注进程消耗；慢响应更关注慢查询和外部依赖。",
                metadata={"_file_name": "cpu_high_usage.md", "h1": "CPU使用率过高告警处理方案"},
            )
        ]

        def fake_retrieve(query, top_k=None, **kwargs):
            record_retrieval(query, fake_docs)
            return fake_docs

        with patch(
            "app.services.rag_agent_service.retrieve_knowledge_documents",
            side_effect=fake_retrieve,
        ) as mock_retrieve:
            with patch.object(
                service.memory_manager,
                "build_messages",
                new=AsyncMock(return_value=[]),
            ) as mock_build_messages:
                with patch.object(
                    service.memory_manager,
                    "complete_turn",
                    new=AsyncMock(),
                ):
                    with patch.object(
                        service,
                        "_ainvoke_eval_model",
                        new=AsyncMock(return_value=AIMessage(content="A 更关注进程，B 更关注慢查询。")),
                    ):
                        with patch.object(
                            service,
                            "_initialize_agent",
                            new=AsyncMock(side_effect=AssertionError("eval mode should not initialize agent")),
                        ):
                            result = await service.query_with_evaluation(
                                "CPU 使用率过高和服务响应变慢在排查重点上有什么区别？",
                                session_id="eval-compare",
                                eval_mode=True,
                                eval_top_k=5,
                            )

        self.assertEqual(result["answer"], "A 更关注进程，B 更关注慢查询。")
        self.assertTrue(result["evaluation"]["enabled"])
        self.assertEqual(len(result["evaluation"]["retrieved_docs"]), 1)
        mock_retrieve.assert_called_once_with(
            "CPU 使用率过高和服务响应变慢在排查重点上有什么区别？",
            top_k=5,
            use_hybrid=None,
        )

        build_kwargs = mock_build_messages.await_args.kwargs
        self.assertIn("对照", build_kwargs["system_prompt"])
        self.assertIn("【参考资料 1】", build_kwargs["question"])
        self.assertIn("A vs B", build_kwargs["question"])

    async def test_eval_mode_builds_scenario_guidance_for_scenario_questions(self):
        service = RagAgentService(streaming=False)
        fake_docs = [
            Document(
                page_content="缓存失效会导致数据库查询量激增和响应时间上升。",
                metadata={"_file_name": "slow_response.md", "h1": "服务响应时间过长告警处理方案"},
            )
        ]

        def fake_retrieve(query, top_k=None, **kwargs):
            record_retrieval(query, fake_docs)
            return fake_docs

        with patch(
            "app.services.rag_agent_service.retrieve_knowledge_documents",
            side_effect=fake_retrieve,
        ):
            with patch.object(
                service.memory_manager,
                "build_messages",
                new=AsyncMock(return_value=[]),
            ) as mock_build_messages:
                with patch.object(
                    service.memory_manager,
                    "complete_turn",
                    new=AsyncMock(),
                ):
                    with patch.object(
                        service,
                        "_ainvoke_eval_model",
                        new=AsyncMock(return_value=AIMessage(content="更像缓存失效问题。")),
                    ):
                        with patch.object(
                            service,
                            "_initialize_agent",
                            new=AsyncMock(side_effect=AssertionError("eval mode should not initialize agent")),
                        ):
                            await service.query_with_evaluation(
                                "如果缓存命中率突然下降、数据库查询量激增、响应时间同步变长，这更像什么问题？应该怎么处理？",
                                session_id="eval-scenario",
                                eval_mode=True,
                                eval_top_k=5,
                            )

        build_kwargs = mock_build_messages.await_args.kwargs
        self.assertIn("判断", build_kwargs["system_prompt"])
        self.assertIn("优先动作", build_kwargs["question"])


if __name__ == "__main__":
    unittest.main()
