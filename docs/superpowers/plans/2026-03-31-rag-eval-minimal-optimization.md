# RAG Eval Minimal Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不大改架构的前提下，优先提升 `scenario/compare` 题型的回答命中率，并减少“明明召回到了但答案没贴 gold points”的问题。

**Architecture:** 这次不引入 GraphRAG、hybrid retrieval 或大规模重构，只做三类最小改动：让评测模式更稳定地走检索、让生成输出更贴近题型结构、补一个小型回归集避免改完后退化。核心思路是先把当前通用 Agent 调成“更像评测型 RAG”，等这轮收益吃满后再考虑 rerank / hybrid retrieval。

**Tech Stack:** FastAPI, LangChain/LangGraph, ChatQwen, Milvus, pytest

---

## File Map

- Modify: `app/services/rag_agent_service.py`
  - 评测模式使用更稳的生成配置
  - 为评测模式增加更强的回答约束
  - 尽量确保知识题优先调用 `retrieve_knowledge`
- Modify: `app/tools/knowledge_tool.py`
  - 可选地补一个轻量 query rewrite / query normalization 钩子
  - 不引入新存储结构
- Create: `tests/services/test_rag_eval_regression.py`
  - 为当前最差的 `scenario/compare` 样本建回归测试
- Modify: `scripts/run_rag_eval.py`
  - 可选：补一份更易比较的输出摘要
- Optional Modify: `app/services/rag_eval_metrics.py`
  - 只在确认“评测低估语义正确”明显时，再微调匹配规则

---

### Task 1: 让评测模式更稳定，不再“聊嗨了”

**Files:**
- Modify: `app/services/rag_agent_service.py`
- Test: `tests/services/test_rag_eval_regression.py`

- [ ] **Step 1: 写一个失败回归测试，覆盖“已召回但答偏”的典型题**

选 3-4 个代表题作为最小回归集：
- `svc_11`
- `disk_11`
- `slow_12`
- `mem_12`

测试目标：
- 断言回答中出现题干要求的核心对比维度
- 断言不再只是泛化建议

- [ ] **Step 2: 运行测试，确认当前实现无法稳定通过**

Run: `pytest tests/services/test_rag_eval_regression.py -v`
Expected: 至少部分 case FAIL，证明当前回答结构不稳定。

- [ ] **Step 3: 在评测模式下单独降低温度**

在 `app/services/rag_agent_service.py` 中：
- 保持线上普通对话链路不动
- 仅在 `eval_mode=True` 时使用更确定的生成参数，例如 `temperature=0` 或 `0.1`

原因：
- 当前主链路 `temperature=0.7` 更适合聊天，不适合评测
- 评测需要稳定、克制、少扩写

- [ ] **Step 4: 在评测模式下增加更强的系统约束**

增加一段仅用于评测模式的附加提示：
- 只能基于检索证据回答
- 不要泛化扩写
- 尽量按 3-6 条 bullet 回答
- 如果是 compare 题，必须用“A vs B”对照结构
- 如果是 scenario 题，先判断“更像什么问题”，再给处理动作

- [ ] **Step 5: 运行回归测试**

Run: `pytest tests/services/test_rag_eval_regression.py -v`
Expected: 至少结构性问题减少，compare/scenario 输出更聚焦。

---

### Task 2: 让知识题优先走检索，而不是靠模型先验硬答

**Files:**
- Modify: `app/services/rag_agent_service.py`
- Modify: `app/tools/knowledge_tool.py`
- Test: `tests/services/test_rag_eval_regression.py`

- [ ] **Step 1: 写一个失败测试，覆盖“漏召但其实应该查知识库”的题**

优先覆盖：
- `cpu_09`
- `cpu_10`
- `slow_10`

测试目标：
- 评测模式下，这类题应该稳定触发 `retrieve_knowledge`

- [ ] **Step 2: 在评测模式下优先触发 `retrieve_knowledge`**

最小改动方案：
- 不改整体 Agent 架构
- 在 `query_with_evaluation()` 中，对评测请求先走一次 `retrieve_knowledge_documents(question, top_k=eval_top_k)`
- 将检索到的上下文作为显式证据插入消息，再让模型生成答案

这样做的好处：
- 不依赖 `create_agent()` 自己决定要不要调工具
- 直接把“是否调用检索”从不确定行为变成确定行为

- [ ] **Step 3: 可选地做轻量 query normalization**

只做最小增强，不引入复杂 rewrite：
- 去掉冗余口语
- 保留告警名、错误码、核心对象词
- compare/scenario 题保留两个实体关键词

如果实现很轻，可以放在 `app/tools/knowledge_tool.py` 内部。

- [ ] **Step 4: 跑回归测试**

Run: `pytest tests/services/test_rag_eval_regression.py -v`
Expected: 漏召样本减少，至少不再因为“没调检索工具”导致空召回。

---

### Task 3: 针对 compare / scenario 题做最小回答模板化

**Files:**
- Modify: `app/services/rag_agent_service.py`
- Test: `tests/services/test_rag_eval_regression.py`

- [ ] **Step 1: 增加题型识别的最小规则**

不要做复杂分类器，只做规则判断：
- question 包含 `有什么区别 / 分别 / 相比 / 关系` -> `compare`
- question 包含 `如果 / 某次 / 同时 / 更像 / 应该怎么处理` -> `scenario`
- 其他保持原样

- [ ] **Step 2: 为 compare 题增加固定输出骨架**

输出要求：
- 先一句总结差异
- 再按 `A / B / 处理重点` 分点

目标是让答案更贴近 gold points 的结构，而不是输出一大段泛化分析。

- [ ] **Step 3: 为 scenario 题增加固定输出骨架**

输出要求：
- 先判断“更像哪类问题”
- 再给“为什么”
- 再给“优先动作”

- [ ] **Step 4: 跑回归测试**

Run: `pytest tests/services/test_rag_eval_regression.py -v`
Expected: `compare/scenario` 的结构更稳定，point hit rate 有明显提升空间。

---

### Task 4: 建一个小型失败样本回归集，避免每次都靠全量评测

**Files:**
- Create: `tests/services/test_rag_eval_regression.py`
- Modify: `docs/rag_eval_results_readable.md`（只读参考，不要求改）

- [ ] **Step 1: 从当前 25 个低分样本里挑 8 个代表题**

建议覆盖：
- 3 个漏召题
- 3 个 compare 题
- 2 个 scenario 题

- [ ] **Step 2: 设计最小断言**

不要一开始就断言整段答案完全一致，只断言：
- 至少覆盖 2-3 个关键点
- compare 题必须出现对照结构
- scenario 题必须出现“判断 + 动作”

- [ ] **Step 3: 跑测试并固化为回归门**

Run: `pytest tests/services/test_rag_eval_regression.py -v`
Expected: 改动后这批高价值样本稳定通过。

---

### Task 5: 最后才考虑评测规则微调（可选，不是首要）

**Files:**
- Optional Modify: `scripts/run_rag_eval.py`
- Optional Modify: `app/services/rag_eval_metrics.py`
- Test: `tests/services/test_rag_eval_metrics.py`

- [ ] **Step 1: 先确认是否真有“语义正确但评分过低”的样本**

只要人工抽样发现 3-5 个明确案例，再动 matcher。

- [ ] **Step 2: 最小化调整 `point_matched()`**

可以考虑：
- 对比类关键词做同义词归一
- 放宽 bigram overlap 阈值一点点
- 或增加人工白名单规则

注意：
- 不要先改评测规则来“刷分”
- 必须先把生成质量问题解决掉

- [ ] **Step 3: 跑现有指标测试**

Run: `pytest tests/services/test_rag_eval_metrics.py -v`
Expected: 原有指标逻辑不被破坏，只减少明显误判。

---

## 推荐执行顺序

先做这 3 件，ROI 最高：

1. `Task 1`：评测模式降温 + 更强回答约束
2. `Task 2`：评测模式强制先检索，再生成
3. `Task 3`：给 compare / scenario 题加最小模板

这三步做完，再重新跑：

Run: `python scripts/run_rag_eval.py --base-url http://127.0.0.1:9900 --retrieval-k 5`

预期收益：
- 漏召数下降
- `compare/scenario` 题显著改善
- 平均要点命中率比现在 `50%` 更容易往上拉

## 不建议这轮就做的事

这轮先别做：
- GraphRAG
- 大规模重构成全新 RAG chain
- 全量 hybrid retrieval
- 引入复杂 reranker 服务
- 重写评测集

原因：
- 这些改动大，验证成本高
- 当前最明显的瓶颈还没吃满

## 成功判定标准

本轮最小优化完成后，建议以这几个目标作为 done 标准：

- 漏召样本从 `8` 降到 `<= 4`
- `compare` 类平均命中率从 `0.15` 提升到 `>= 0.40`
- `scenario` 类平均命中率从 `0.35` 提升到 `>= 0.50`
- 总体平均要点命中率从 `0.50` 提升到 `>= 0.60`

---

Plan complete and saved to `docs/superpowers/plans/2026-03-31-rag-eval-minimal-optimization.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - 我按任务逐个实现并复核，适合这种小步快跑优化

**2. Inline Execution** - 我直接在这个会话里连续实现这 3 个高 ROI 任务
