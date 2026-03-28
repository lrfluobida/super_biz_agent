# Memory Manager Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conversation memory system with SQLite-backed raw-session persistence, a sliding window, structured summaries, and token-based compaction thresholds.

**Architecture:** Introduce a `MemoryManager` that persists raw turns in SQLite while keeping only recent turns and a structured summary in active prompt memory. Keep the existing chat API shape compatible by continuing to return the full raw history to the front end, while the model consumes `system prompt + summary + recent turns + current question`.

**Tech Stack:** FastAPI, LangChain, LangGraph agent runtime, ChatQwen, pytest

---

### Task 1: Add memory models and failing unit tests

**Files:**
- Create: `app/models/memory.py`
- Create: `tests/services/test_memory_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_complete_turn_summarizes_messages_beyond_window():
    manager = MemoryManager(window_turns=2, ...)
    await manager.complete_turn("s1", "u1", "a1")
    await manager.complete_turn("s1", "u2", "a2")
    await manager.complete_turn("s1", "u3", "a3")
    history = manager.get_session_history("s1")
    assert history[0]["content"].startswith("[历史摘要]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: FAIL because `MemoryManager` does not exist yet

- [ ] **Step 3: Add minimal data models**

```python
class MemorySummary(BaseModel):
    current_goal: str = ""
    important_facts: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it still fails for missing behavior**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: FAIL because compaction and prompt building are not implemented yet

### Task 2: Implement in-memory memory manager

**Files:**
- Create: `app/services/memory_manager.py`
- Modify: `app/config.py`

- [ ] **Step 1: Write the failing tests for token budgeting**

```python
async def test_build_messages_compacts_when_prompt_exceeds_hard_threshold():
    manager = MemoryManager(prompt_budget_tokens=120, hard_ratio=0.8, ...)
    messages = await manager.build_messages("s1", "system", "follow-up")
    assert any("结构化摘要" in msg.content for msg in messages)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: FAIL because token-aware compaction is not implemented

- [ ] **Step 3: Implement minimal manager**

```python
class MemoryManager:
    async def complete_turn(...):
        ...

    async def build_messages(...):
        ...
```

- [ ] **Step 4: Add configuration defaults**

```python
memory_window_turns = 5
memory_prompt_budget_tokens = 24000
memory_summary_soft_ratio = 0.6
memory_summary_hard_ratio = 0.8
memory_reserved_output_tokens = 4096
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: PASS

### Task 3: Integrate memory manager into chat service

**Files:**
- Modify: `app/services/rag_agent_service.py`

- [ ] **Step 1: Write the failing integration-style tests or assertions around service behavior**

```python
async def test_query_uses_memory_manager_messages():
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: FAIL or missing integration coverage

- [ ] **Step 3: Update the service**

```python
messages = await self.memory_manager.build_messages(...)
result = await self.agent.ainvoke({"messages": messages}, ...)
await self.memory_manager.complete_turn(...)
```

- [ ] **Step 4: Ensure streaming collects the final answer before persisting**

```python
buffer.append(text_content)
...
await self.memory_manager.complete_turn(session_id, question, "".join(buffer))
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: PASS

### Task 4: Keep session-history API compatible

**Files:**
- Modify: `app/api/chat.py`
- Modify: `app/models/response.py`

- [ ] **Step 1: Confirm current response shape**

Run: `rg -n "SessionInfoResponse|history" app static -S`
Expected: identify consumers of `history`

- [ ] **Step 2: Return summary + recent turns through existing `history` field**

```python
history = rag_agent_service.get_session_history(session_id)
message_count = len(history)
```

- [ ] **Step 3: Keep the response contract unchanged**

Expected: front-end still renders history as a normal message list

### Task 5: Verify end-to-end behavior

**Files:**
- Test: `tests/services/test_memory_manager.py`

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/services/test_memory_manager.py -q`
Expected: PASS

- [ ] **Step 2: Run a broader smoke test if available**

Run: `pytest -q`
Expected: PASS or note unrelated failures

- [ ] **Step 3: Re-open edited files and verify UTF-8 / Chinese text**

Expected: no mojibake, existing Chinese comments remain intact
