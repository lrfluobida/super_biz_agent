"""对话记忆管理器

负责：
1. 原始会话持久化到 SQLite
2. 滑动窗口记忆
3. 结构化摘要更新
4. Token 预算控制
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.models.memory import MemorySummary

SessionMessage = dict[str, str]
SummaryUpdater = Callable[[MemorySummary, list[SessionMessage]], Awaitable[MemorySummary]]


@dataclass
class SessionState:
    """单个会话的运行态记忆"""

    summary: MemorySummary = field(default_factory=MemorySummary)
    recent_messages: list[SessionMessage] = field(default_factory=list)


class MemoryManager:
    """管理会话记忆与持久化"""

    def __init__(
        self,
        db_path: str | None = None,
        window_turns: int | None = None,
        prompt_budget_tokens: int | None = None,
        summary_soft_ratio: float | None = None,
        summary_hard_ratio: float | None = None,
        reserved_output_tokens: int | None = None,
        summary_updater: SummaryUpdater | None = None,
    ):
        self.db_path = Path(db_path or config.memory_sqlite_path)
        self.window_turns = window_turns or config.memory_window_turns
        self.prompt_budget_tokens = prompt_budget_tokens or config.memory_prompt_budget_tokens
        self.summary_soft_ratio = summary_soft_ratio or config.memory_summary_soft_ratio
        self.summary_hard_ratio = summary_hard_ratio or config.memory_summary_hard_ratio
        self.reserved_output_tokens = reserved_output_tokens or config.memory_reserved_output_tokens
        self._sessions: dict[str, SessionState] = {}
        self._token_encoder = self._build_token_encoder()
        self._summary_model = None
        self._summary_updater = summary_updater

        self._ensure_database()

    async def build_messages(
        self,
        session_id: str,
        system_prompt: str,
        question: str,
    ) -> list[BaseMessage]:
        """构建发送给主模型的消息列表"""
        state = self._get_or_load_session(session_id)
        await self._compact_for_budget(session_id, state, question)

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        if state.summary.has_content():
            messages.append(SystemMessage(content=self._format_summary_for_prompt(state.summary)))

        for message in state.recent_messages:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            else:
                messages.append(AIMessage(content=message["content"]))

        messages.append(HumanMessage(content=question))
        self._log_prompt_preview(session_id, messages)
        return messages

    async def complete_turn(self, session_id: str, question: str, answer: str) -> None:
        """完成一轮对话并更新记忆"""
        state = self._get_or_load_session(session_id)
        timestamp = self._now()
        messages = [
            {"role": "user", "content": question, "timestamp": timestamp},
            {"role": "assistant", "content": answer, "timestamp": timestamp},
        ]

        self._persist_messages(session_id, messages)
        state.recent_messages.extend(messages)
        self._save_summary(session_id, state.summary)

        await self._compact_to_window(session_id, state)

    def get_session_history(self, session_id: str) -> list[SessionMessage]:
        """获取完整历史，用于前端展示"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            return [
                {"role": row[0], "content": row[1], "timestamp": row[2]}
                for row in rows
            ]
        finally:
            conn.close()

    def clear_session(self, session_id: str) -> bool:
        """清理会话缓存和 SQLite 数据"""
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
            self._sessions.pop(session_id, None)
            return True
        except Exception as exc:
            logger.error(f"清理会话失败: {session_id}, 错误: {exc}")
            return False
        finally:
            conn.close()

    def get_message_count(self, session_id: str) -> int:
        """获取完整历史消息数"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    async def _compact_to_window(self, session_id: str, state: SessionState) -> None:
        max_recent_messages = self.window_turns * 2
        if len(state.recent_messages) <= max_recent_messages:
            return

        overflow_count = len(state.recent_messages) - max_recent_messages
        overflow_messages = state.recent_messages[:overflow_count]
        state.recent_messages = state.recent_messages[overflow_count:]
        state.summary = await self._update_summary(state.summary, overflow_messages)
        self._save_summary(session_id, state.summary)

    async def _compact_for_budget(
        self,
        session_id: str,
        state: SessionState,
        question: str,
    ) -> None:
        hard_limit = int(self.prompt_budget_tokens * self.summary_hard_ratio)
        soft_limit = int(self.prompt_budget_tokens * self.summary_soft_ratio)
        estimated = self._estimate_prompt_tokens(state, question)

        while estimated > hard_limit and len(state.recent_messages) > 2:
            overflow_messages = state.recent_messages[:2]
            state.recent_messages = state.recent_messages[2:]
            state.summary = await self._update_summary(state.summary, overflow_messages)
            self._save_summary(session_id, state.summary)
            estimated = self._estimate_prompt_tokens(state, question)

        if estimated > soft_limit and len(state.recent_messages) > 4:
            overflow_messages = state.recent_messages[:2]
            state.recent_messages = state.recent_messages[2:]
            state.summary = await self._update_summary(state.summary, overflow_messages)
            self._save_summary(session_id, state.summary)

    def _estimate_prompt_tokens(self, state: SessionState, question: str) -> int:
        summary_text = self._format_summary_for_prompt(state.summary) if state.summary.has_content() else ""
        parts = [summary_text, question]
        parts.extend(message["content"] for message in state.recent_messages)

        estimated = 0
        for part in parts:
            estimated += self._estimate_text_tokens(part)

        estimated += max(0, len(state.recent_messages) * 4)
        return estimated + self.reserved_output_tokens

    def _log_prompt_preview(self, session_id: str, messages: list[BaseMessage]) -> None:
        """在调试模式下打印最终 prompt 的消息预览"""
        if not config.debug:
            return

        previews = []
        for index, message in enumerate(messages, start=1):
            role = self._get_message_role(message)
            content = self._message_to_text(message)
            preview = content.replace("\r", " ").replace("\n", " ").strip()
            if len(preview) > 60:
                preview = f"{preview[:60]}..."
            previews.append(f"{index}. {role}: {preview}")

        logger.info(
            f"[会话 {session_id}] 最终 prompt messages ({len(messages)} 条):\n"
            + "\n".join(previews)
        )

    async def _update_summary(
        self,
        current_summary: MemorySummary,
        overflow_messages: list[SessionMessage],
    ) -> MemorySummary:
        if not overflow_messages:
            return current_summary

        updater = self._summary_updater or self._default_summary_updater
        return await updater(current_summary, overflow_messages)

    async def _default_summary_updater(
        self,
        current_summary: MemorySummary,
        overflow_messages: list[SessionMessage],
    ) -> MemorySummary:
        if self._summary_model is None:
            self._summary_model = ChatQwen(
                model=config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=0.2,
                streaming=False,
            )

        prompt = self._build_summary_prompt(current_summary, overflow_messages)
        response = await self._summary_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是对话记忆压缩助手。"
                        "请只基于给定旧摘要和对话内容更新结构化摘要，不要编造信息。"
                        "输出必须是 JSON 对象。"
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        content = self._message_to_text(response)
        summary_data = self._extract_json_object(content)

        try:
            return MemorySummary.model_validate(summary_data)
        except Exception as exc:
            logger.warning(f"摘要解析失败，回退到启发式摘要: {exc}")
            return self._fallback_summary(current_summary, overflow_messages)

    def _fallback_summary(
        self,
        current_summary: MemorySummary,
        overflow_messages: list[SessionMessage],
    ) -> MemorySummary:
        user_messages = [item["content"] for item in overflow_messages if item["role"] == "user"]
        assistant_messages = [
            item["content"] for item in overflow_messages if item["role"] == "assistant"
        ]

        important_facts = list(current_summary.important_facts)
        important_facts.extend(user_messages[-2:])

        resolved_items = list(current_summary.resolved_items)
        if assistant_messages:
            resolved_items.append(assistant_messages[-1][:120])

        return MemorySummary(
            current_goal=current_summary.current_goal or (user_messages[-1] if user_messages else ""),
            important_facts=self._deduplicate_items(important_facts),
            constraints=current_summary.constraints,
            resolved_items=self._deduplicate_items(resolved_items),
            open_items=current_summary.open_items,
            user_preferences=current_summary.user_preferences,
        )

    def _build_summary_prompt(
        self,
        current_summary: MemorySummary,
        overflow_messages: list[SessionMessage],
    ) -> str:
        payload = {
            "old_summary": current_summary.model_dump(),
            "new_messages": overflow_messages,
            "target_schema": {
                "current_goal": "string",
                "important_facts": ["string"],
                "constraints": ["string"],
                "resolved_items": ["string"],
                "open_items": ["string"],
                "user_preferences": ["string"],
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _format_summary_for_prompt(self, summary: MemorySummary) -> str:
        lines = ["以下是之前对话的结构化摘要，请在回答时继续沿用这些上下文："]
        if summary.current_goal:
            lines.append(f"- 当前目标: {summary.current_goal}")
        if summary.important_facts:
            lines.append(f"- 重要事实: {'；'.join(summary.important_facts)}")
        if summary.constraints:
            lines.append(f"- 约束条件: {'；'.join(summary.constraints)}")
        if summary.resolved_items:
            lines.append(f"- 已解决事项: {'；'.join(summary.resolved_items)}")
        if summary.open_items:
            lines.append(f"- 待处理事项: {'；'.join(summary.open_items)}")
        if summary.user_preferences:
            lines.append(f"- 用户偏好: {'；'.join(summary.user_preferences)}")
        return "\n".join(lines)

    def _get_or_load_session(self, session_id: str) -> SessionState:
        if session_id in self._sessions:
            return self._sessions[session_id]

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT summary_json
                FROM conversation_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            summary = MemorySummary()
            if row and row[0]:
                try:
                    summary = MemorySummary.model_validate_json(row[0])
                except Exception as exc:
                    logger.warning(f"读取会话摘要失败，使用空摘要: {exc}")

            recent_limit = self.window_turns * 2
            cursor = conn.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, recent_limit),
            )
            recent_rows = list(reversed(cursor.fetchall()))
            recent_messages = [
                {"role": row[0], "content": row[1], "timestamp": row[2]}
                for row in recent_rows
            ]
        finally:
            conn.close()

        state = SessionState(summary=summary, recent_messages=recent_messages)
        self._sessions[session_id] = state
        return state

    def _persist_messages(self, session_id: str, messages: list[SessionMessage]) -> None:
        now = self._now()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_sessions(session_id, summary_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (session_id, "{}", now, now),
                )
                conn.executemany(
                    """
                    INSERT INTO conversation_messages(session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (session_id, message["role"], message["content"], message["timestamp"])
                        for message in messages
                    ],
                )
                conn.execute(
                    """
                    UPDATE conversation_sessions
                    SET updated_at = ?
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
        finally:
            conn.close()

    def _save_summary(self, session_id: str, summary: MemorySummary) -> None:
        now = self._now()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_sessions(session_id, summary_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        summary_json = excluded.summary_json,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, summary.model_dump_json(), now, now),
                )
        finally:
            conn.close()

    def _ensure_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        summary_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id
                    ON conversation_messages(session_id, id)
                    """
                )
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _build_token_encoder(self) -> Any | None:
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0

        if self._token_encoder is not None:
            return len(self._token_encoder.encode(text))

        # 回退：按中文/英文混合文本做保守估计
        return max(1, len(text) // 2)

    def _message_to_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _get_message_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, SystemMessage):
            return "system"
        return type(message).__name__.lower()

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("响应中未找到 JSON 对象")
        return json.loads(match.group(0))

    def _deduplicate_items(self, items: list[str]) -> list[str]:
        results: list[str] = []
        for item in items:
            normalized = item.strip()
            if normalized and normalized not in results:
                results.append(normalized)
        return results[-10:]

    def _now(self) -> str:
        return datetime.now().isoformat()
