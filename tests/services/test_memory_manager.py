import os
import unittest
from unittest.mock import patch
from loguru import logger as loguru_logger
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["DEBUG"] = "false"

_original_logger_add = loguru_logger.add


def _patched_logger_add(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("logs/"):
        kwargs["enqueue"] = False
    return _original_logger_add(*args, **kwargs)


loguru_logger.add = _patched_logger_add

from app.models.memory import MemorySummary
from app.config import config
from app.services.memory_manager import MemoryManager


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    async def __call__(self, summary: MemorySummary, overflow_messages):
        self.calls.append((summary, overflow_messages))
        facts = [message["content"] for message in overflow_messages if message["role"] == "user"]
        return MemorySummary(
            current_goal="继续当前任务",
            important_facts=facts,
            constraints=["保持兼容"],
            resolved_items=[],
            open_items=["等待后续问题"],
            user_preferences=["中文回复"],
        )


class MemoryManagerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = os.path.join("volumes", "test-memory")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.db_path = os.path.join(self.temp_dir, f"{self.id().split('.')[-1]}.sqlite3")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def asyncTearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def test_complete_turn_persists_full_history_and_summarizes_prompt_memory(self):
        summarizer = FakeSummarizer()
        manager = MemoryManager(
            db_path=self.db_path,
            window_turns=2,
            prompt_budget_tokens=1000,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=128,
            summary_updater=summarizer,
        )

        await manager.complete_turn("session-1", "问题1", "回答1")
        await manager.complete_turn("session-1", "问题2", "回答2")
        await manager.complete_turn("session-1", "问题3", "回答3")

        history = manager.get_session_history("session-1")

        self.assertEqual(
            [item["content"] for item in history],
            ["问题1", "回答1", "问题2", "回答2", "问题3", "回答3"],
        )
        self.assertEqual(len(summarizer.calls), 1)

    async def test_build_messages_compacts_when_prompt_exceeds_hard_threshold(self):
        summarizer = FakeSummarizer()
        manager = MemoryManager(
            db_path=self.db_path,
            window_turns=5,
            prompt_budget_tokens=120,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=16,
            summary_updater=summarizer,
        )

        await manager.complete_turn("session-2", "这是第一轮特别长的问题" * 8, "这是第一轮特别长的回答" * 8)
        await manager.complete_turn("session-2", "这是第二轮特别长的问题" * 8, "这是第二轮特别长的回答" * 8)

        messages = await manager.build_messages("session-2", "系统提示", "继续处理这个问题")

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertTrue(
            any(
                isinstance(message, SystemMessage) and "结构化摘要" in message.content
                for message in messages
            )
        )
        self.assertTrue(
            any(
                isinstance(message, HumanMessage) and message.content == "继续处理这个问题"
                for message in messages
            )
        )
        self.assertGreaterEqual(len(summarizer.calls), 1)

    async def test_new_manager_instance_recovers_session_from_sqlite(self):
        summarizer = FakeSummarizer()
        manager = MemoryManager(
            db_path=self.db_path,
            window_turns=2,
            prompt_budget_tokens=200,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=32,
            summary_updater=summarizer,
        )

        await manager.complete_turn("session-reload", "问题1", "回答1")
        await manager.complete_turn("session-reload", "问题2", "回答2")
        await manager.complete_turn("session-reload", "问题3", "回答3")

        reloaded = MemoryManager(
            db_path=self.db_path,
            window_turns=2,
            prompt_budget_tokens=200,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=32,
            summary_updater=FakeSummarizer(),
        )

        history = reloaded.get_session_history("session-reload")
        messages = await reloaded.build_messages("session-reload", "系统提示", "继续")

        self.assertEqual(len(history), 6)
        self.assertTrue(
            any(
                isinstance(message, SystemMessage) and "结构化摘要" in message.content
                for message in messages
            )
        )
        self.assertTrue(any(message.content == "问题2" for message in messages[1:]))
        self.assertTrue(any(message.content == "回答3" for message in messages[1:]))

    async def test_clear_session_resets_sqlite_and_memory_cache(self):
        manager = MemoryManager(
            db_path=self.db_path,
            window_turns=2,
            prompt_budget_tokens=1000,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=128,
            summary_updater=FakeSummarizer(),
        )

        await manager.complete_turn("session-3", "问题", "回答")

        self.assertTrue(manager.clear_session("session-3"))
        self.assertEqual(manager.get_session_history("session-3"), [])

    async def test_build_messages_logs_prompt_preview_when_debug_enabled(self):
        manager = MemoryManager(
            db_path=self.db_path,
            window_turns=2,
            prompt_budget_tokens=1000,
            summary_soft_ratio=0.6,
            summary_hard_ratio=0.8,
            reserved_output_tokens=128,
            summary_updater=FakeSummarizer(),
        )

        await manager.complete_turn("session-debug", "问题1", "回答1")

        original_debug = config.debug
        config.debug = True
        try:
            with patch("app.services.memory_manager.logger.info") as mock_info:
                await manager.build_messages("session-debug", "系统提示", "继续追问")

            mock_info.assert_called()
            logged_text = mock_info.call_args[0][0]
            self.assertIn("最终 prompt messages", logged_text)
            self.assertIn("system", logged_text.lower())
            self.assertIn("继续追问", logged_text)
        finally:
            config.debug = original_debug
