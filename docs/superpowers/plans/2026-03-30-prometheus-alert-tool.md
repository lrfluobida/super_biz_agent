# Prometheus Alert Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real `query_prometheus_alerts` MCP tool that fetches active Prometheus alerts and makes AIOps start diagnosis from alert data.

**Architecture:** Keep the MCP integration shape unchanged by adding the new tool to the existing Monitor MCP server. Isolate the HTTP fetch and JSON parsing into a small service module so the parsing logic can be unit-tested without booting the MCP server, then call that service from the new MCP tool and update the AIOps diagnosis prompt to prioritize the tool.

**Tech Stack:** FastMCP, FastAPI, httpx, LangGraph, pytest

---

### Task 1: Add failing tests for Prometheus alert fetch and parsing

**Files:**
- Create: `tests/services/test_prometheus_alert_service.py`
- Create: `app/services/prometheus_alert_service.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_fetch_alerts_extracts_required_fields():
    ...
    assert result["alerts"] == [
        {
            "alertname": "ServiceDown",
            "description": "广告微服务下线",
            "activeAt": "2025-10-29T08:48:42Z",
        }
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: FAIL because the service does not exist yet

- [ ] **Step 3: Add the minimal service implementation**

```python
async def fetch_prometheus_alerts(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: PASS

### Task 2: Cover error handling and missing field behavior

**Files:**
- Modify: `tests/services/test_prometheus_alert_service.py`
- Modify: `app/services/prometheus_alert_service.py`

- [ ] **Step 1: Write failing tests for error paths**

```python
async def test_fetch_alerts_returns_none_description_when_missing():
    ...

async def test_fetch_alerts_raises_when_prometheus_status_is_not_success():
    ...
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: FAIL because the missing-field and error-path behavior is not implemented yet

- [ ] **Step 3: Implement the minimal parsing and validation changes**

```python
description = annotations.get("description")
if payload.get("status") != "success":
    raise PrometheusAlertError(...)
```

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: PASS

### Task 3: Expose the service through the Monitor MCP server and config

**Files:**
- Modify: `mcp_servers/monitor_server.py`
- Modify: `app/config.py`

- [ ] **Step 1: Write the failing integration-style test or import-level assertion**

```python
def test_query_prometheus_alerts_tool_returns_structured_payload():
    ...
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: FAIL because the MCP tool/config wiring does not exist yet

- [ ] **Step 3: Add config defaults and the MCP tool**

```python
prometheus_base_url = "http://localhost:9090"
prometheus_timeout_seconds = 5.0
```

```python
@mcp.tool()
async def query_prometheus_alerts() -> Dict[str, Any]:
    ...
```

- [ ] **Step 4: Run targeted tests again**

Run: `pytest tests/services/test_prometheus_alert_service.py -q`
Expected: PASS

### Task 4: Make AIOps explicitly start from Prometheus alerts

**Files:**
- Modify: `app/services/aiops_service.py`
- Create: `tests/services/test_aiops_prompt.py`

- [ ] **Step 1: Write the failing prompt test**

```python
def test_aiops_task_prompt_requires_query_prometheus_alerts_first():
    prompt = build_aiops_task_prompt()
    assert "query_prometheus_alerts" in prompt
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/services/test_aiops_prompt.py -q`
Expected: FAIL because the prompt builder does not exist yet

- [ ] **Step 3: Extract the task prompt builder and update the prompt text**

```python
def build_aiops_task_prompt() -> str:
    ...
```

- [ ] **Step 4: Run the prompt test**

Run: `pytest tests/services/test_aiops_prompt.py -q`
Expected: PASS

### Task 5: Update docs and run final verification

**Files:**
- Modify: `mcp_servers/README.md`

- [ ] **Step 1: Document the new tool and Prometheus Docker setup**

Expected: README explains `query_prometheus_alerts`, required env/config, and a minimal Docker startup example

- [ ] **Step 2: Run targeted verification**

Run: `pytest tests/services/test_prometheus_alert_service.py tests/services/test_aiops_prompt.py -q`
Expected: PASS

- [ ] **Step 3: Run a broader smoke test if feasible**

Run: `pytest -q`
Expected: PASS or report unrelated existing failures

- [ ] **Step 4: Re-open edited files and verify UTF-8 / Chinese text**

Expected: no mojibake, existing Chinese text still renders correctly
