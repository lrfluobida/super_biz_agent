# Query Rewrite 实现报告

**日期**：2026-05-05  
**作者**：LRF + Claude Code  
**版本**：v1

---

## 一、优化出发点

### 当前 RAG 管线存在的问题

| 问题 | 场景举例 | 影响 |
|------|----------|------|
| 复合问题无法拆分 | "CPU高和内存高的排查流程有什么区别" → 作为单一问题检索，召回偏向某一侧 | 对比类问题答案不全面 |
| 过度具体化问题检索效果差 | "针对level=70的阈值设置建议" → 检索不到精确匹配的文档 | MRR 降低，要点丢失 |
| 多轮代词指代丢失 | 上一轮问"CPU排查"，追问"那内存呢" → 检索关键词变成"那内存呢" | 上下文断裂，召回失败 |

### 设计目标

1. 快速路径优先：简单问题规则命中直接透传（<1ms），不调用模型
2. 模型兜底：复杂问题调用本地大模型分类+改写（~1-3s）
3. 安全兜底：每层可降级，最坏情况等同于当前行为（不改写）
4. 不破坏现有流程：改写发生在 Agent 上游——改写输入的问题文本，不绕过 Agent 操控检索

---

## 二、架构

```
用户原始 Query
      │
      ▼
┌──────────────────────┐
│  Step 1: Route       │  ← Stage1 规则（<1ms）+ Stage2 本地模型（~1-3s）
│     → direct         │     规则覆盖 6 种模式，未命中进 Stage2
│     → decompose      │
│     → step_back      │
│     → contextualize  │
├──────────────────────┤
│  Step 2: Rewrite     │  ← Ollama qwen2.5:7b
│     → 原样透传       │     decompose: 拆分为 2-4 个独立子查询
│     → 拆分子查询     │     step_back: 泛化查询 + 保留具体参数
│     → 泛化+保留具体  │     contextualize: 代词补全为具体实体
│     → 代词补全       │
├──────────────────────┤
│  Step 3: Drift Guard │  ← DashScope text-embedding-v4 批量嵌入
│     ≥0.95 伪分解剔除 │     新增伪分解过滤
│     ≥0.65 可用       │
│     0.40-0.65 双路   │
│     <0.40 丢弃       │
├──────────────────────┤
│  Step 4: 构造        │
│     agent_question   │  ← 喂给 Agent 的改写后问题
│     extra_system_    │  ← 追加到 System Prompt 的引导指令
│        prompt        │
└──────────────────────┘
```

### 降级链路（6 层）

```
规则命中 → 模型分类 → JSON 解析 → 重试 → Fallback direct
   ↓          ↓          ↓         ↓         ↓
  可用       可用       可用      可用     最终兜底（不改写）
```

---

## 三、文件变更

### 3.1 新建文件（7 个）

| 文件 | 职责 |
|------|------|
| `app/models/rewrite.py` | Pydantic 数据模型：Intent、RouterResult、DecomposeResult、StepBackResult、ContextualizeResult、DriftCheckResult、RewritePipelineResult |
| `app/services/rewrite_model_service.py` | Ollama ChatOpenAI 封装，max_tokens=256，三层 JSON 提取 |
| `app/services/query_router.py` | Stage1 规则矩阵 + Stage2 模型分类，6 条规则覆盖 4 种意图 |
| `app/services/query_rewriter.py` | 三种改写器：decompose/step_back/contextualize |
| `app/services/drift_guard.py` | 余弦相似度漂移检测，check() 单条 + check_batch() 批量 |
| `app/services/async_retrieval_service.py` | parallel_decomposed_retrieval() 并行多 query 检索 + RRF 合并（当前未在 decompose 中使用，保留供后续） |
| `app/services/query_rewrite_pipeline.py` | 编排器：route → rewrite → drift → agent_question + extra_system_prompt |

### 3.2 修改文件（4 个）

| 文件 | 变更 |
|------|------|
| `app/config.py` | +12 项 rewrite 配置（模型 URL/名称/温度/超时、router 开关、drift 阈值、并行检索） |
| `app/services/rag_agent_service.py` | _apply_rewrite() 方法，Normal/Eval/Stream 三种模式全部接入；_build_system_prompt() 支持动态追加；complete_turn() 存储原始 query |
| `app/services/rag_trace.py` | record_rewrite() 函数，trace 输出包含 rewrite 字段（intent、agent_question、drift、extra_system_prompt） |
| `app/api/chat.py` | X-RAG-Rewrite-Enabled HTTP Header 支持 |

### 3.3 测试文件（4 个新建）

| 文件 | 用例数 | 覆盖范围 |
|------|:----:|----------|
| `tests/services/test_query_router.py` | 25 | 规则辅助函数、Stage1 规则矩阵、Stage2 降级（3 种异常→fallback）、route 集成 |
| `tests/services/test_query_rewriter.py` | 12 | 三种改写器正确解析、空结果/异常回退、子查询截断过滤 |
| `tests/services/test_drift_guard.py` | 15 | 余弦相似度计算（6 边界）、阈值分类（5 档）、集成（3） |
| `tests/services/test_query_rewrite_pipeline.py` | 12 | agent_question 构造、直接透传、contextualize 漂移回退、decompose 伪分解过滤、异常降级 |

**64 个用例全部通过。**

### 3.4 评测相关

| 文件 | 变更 |
|------|------|
| `docs/rag_eval_dataset.json` | 新增 rewrite_expected 字段，标注 17 条预期改写意图 |
| `scripts/run_rag_eval.py` | 新增 --rewrite/--no-rewrite/--compare-rewrite 参数，支持 A/B 对比 |

---

## 四、关键设计决策

### 4.1 Agent 输入改写（非检索改写）

改写发生在 Agent 的上游——改写喂给 Agent 的问题文本（agent_question），Agent 仍然自己决定何时检索、检索什么。不破坏 Agent 的自主推理闭环。

### 4.2 Memory 存储原始 Query

memory_manager.complete_turn() 存入用户原始消息（"那内存呢"），而非改写后的消息（"内存使用率过高排查流程"）。保证多轮对话的上下文自然。

### 4.3 System Prompt 动态增强

| 意图 | 追加指令 |
|------|---------|
| decompose | "必须对每个子主题分别调 retrieve_knowledge（至少 N 次），确认全部完成后再综合对比。严格禁止仅检索一次就生成答案。" |
| step_back | "从通用知识 + 具体方案两个层面检索和回答。" |
| contextualize | "系统已自动将代词替换为具体内容，当前检索语句已经完整。" |

### 4.4 伪分解过滤

子查询与原始 query 相似度 ≥ 0.95 时判定为"换个说法而已"，直接剔除。防止模型输出跟原问题几乎相同的"伪"子查询。

### 4.5 降级全面性

| 失败场景 | 降级行为 | 用户影响 |
|----------|----------|:--:|
| Ollama 超时 | 跳过改写，用原始 query | 零 |
| Ollama 不可用 | Router→rule-only，Rewriter→skip | 仅失模型分类 |
| 模型返回非 JSON | 3 层提取 + 重试 → fallback {} → direct | 极小 |
| 嵌入 API 报错 | 跳过 drift guard，信任改写 | 小 |
| 所有子查询漂移/伪分解 | 丢弃改写，回退原始 query | 零 |
| 整个 Pipeline 异常 | 原始 query → 正常检索 | 零 |

---

## 五、评估结果

### 5.1 环境

- 改写模型：Ollama `qwen2.5:7b`（4.7GB，Q4_K_M 量化）
- 评测模式：normal（Agent 完整推理流程）
- 检索：Hybrid（Dense + BM25 + RRF），Top-K=5
- 评测集：60 题

### 5.2 7B 改写 vs 不改写（均为 normal 模式）

| 指标 | 不改写 | 7B 改写 | 变化 |
|------|:---:|:---:|:---:|
| Hit@3 | 93.33% | **100%** | +6.67% |
| Hit@5 | 93.33% | **100%** | +6.67% |
| MRR | 0.7972 | **0.8861** | **+11.1%** |
| 要点命中率 | 46.11% | **56.53%** | **+10.4%** |
| 平均时延 | 6.02s | 8.49s | +2.47s |
| P95 时延 | 11.53s | 15.43s | +3.90s |

### 5.3 改写覆盖

| 意图 | 题数 | 实际改写 |
|------|:---:|:---:|
| decompose（拆分） | 27 | 27 |
| step_back（泛化） | 2 | 1 |
| direct（不改写） | 31 | 0 |

改写覆盖率：**28/60 = 47%**。

### 5.4 时延分析

加改写前 6.02s → 加改写后 8.49s，额外 2.47s 为：路由规则（<1ms）+ Ollama 7B 推理（~1-3s）+ Drift Guard 嵌入（~60ms）。

### 5.5 1.5B vs 7B 对比

| 指标 | 1.5B (eval) | 7B (normal) |
|------|:---:|:---:|
| 改写覆盖率 | 12%（7/60） | 47%（28/60） |
| decompose 识别 | 10 题 | 27 题 |
| 子查询截断/变形 | 常见 | 极少 |

**结论：7B 模型改写质量明显优于 1.5B，覆盖率 4x 提升，时延在可接受范围。**

---

## 六、已尝试但放弃的方案

### decompose 预检索

**方案**：pipeline 中并行检索所有子查询，结果注入 System Prompt，Agent 直接综合。

**结果**：时延反而增加。原因：
- 并行检索 + 嵌入 API 调用 ~1-3s
- 预检索上下文 3000+ 字符撑大 System Prompt，LLM 生成变慢
- Agent 工具调用往返本身很快（每次 ~200ms），不是瓶颈

**结论**：已回滚。保留 Agent 强制分步检索指令。

---

## 七、配置项

```bash
# .env 新增
REWRITE_LOCAL_MODEL_NAME=qwen2.5:7b   # 改写模型（覆盖 config.py 默认的 1.5b）

# config.py 新增（可通过 .env 覆盖）
rewrite_enabled: bool = True
rewrite_local_model_url: str = "http://localhost:11434/v1"
rewrite_local_model_name: str = "qwen2.5:1.5b"
rewrite_local_model_temperature: float = 0.1
rewrite_local_model_timeout: int = 10
rewrite_router_enabled: bool = True
rewrite_drift_threshold: float = 0.65
rewrite_drift_moderate_threshold: float = 0.40
rewrite_parallel_retrieval_enabled: bool = True
rewrite_parallel_max_workers: int = 4
```

---

## 八、使用方式

### 评测

```bash
# 单次评测
python scripts/run_rag_eval.py --mode normal --rewrite

# A/B 对比（不改写 vs 改写）
python scripts/run_rag_eval.py --mode normal --compare-rewrite
```

### 单元测试

```bash
pytest tests/services/test_query_router.py \
       tests/services/test_query_rewriter.py \
       tests/services/test_drift_guard.py \
       tests/services/test_query_rewrite_pipeline.py -v
```

### HTTP Header 控制

```bash
curl -X POST http://127.0.0.1:9900/api/chat \
  -H "Content-Type: application/json" \
  -H "X-RAG-Rewrite-Enabled: true" \
  -d '{"Id":"s1","Question":"CPU和内存排查的区别是什么？"}'
```
