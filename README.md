# SuperBizAgent

> 企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能诊断


## ✨ 核心特性

- 🤖 **智能对话** - LangGraph Agent 多轮对话 + 流式输出 + SQLite 会话记忆
- 📚 **RAG 问答** - 混合检索（Dense + BM25 + RRF）+ Contextual Chunking 上下文增强 + Query Rewrite 查询改写，支持文档上传自动索引
- 🔧 **AIOps 诊断** - Plan-Execute-Replan 自动故障诊断和根因分析
- 🌐 **Web 界面** - 现代化 UI，支持快速问答、流式对话、AIOps 诊断
- 🔌 **MCP 集成** - 日志查询和监控数据工具接入
- 📊 **评估体系** - RAG 检索轨迹追踪 + 要点命中率评估 + 离线批量评测脚本

## 🛠️ 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope (通义千问 qwen3.6-flash, text-embedding-v4)
- **向量库**: Milvus (Dense COSINE + Sparse BM25 IP 双向量)
- **混合检索**: 二路召回 + RRF (Reciprocal Rank Fusion) 合并
- **工具协议**: MCP (Model Context Protocol) + streamable-http 传输
- **会话记忆**: SQLite 持久化 + Token 预算控制 + 结构化摘要压缩

## 🚀 快速开始

### 环境要求
- Python 3.11+
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))
- Docker Desktop（Milvus 向量数据库）

### 安装和启动

#### Linux/macOS 环境

```bash
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 安装依赖（推荐使用 uv）
# 方式 1: 使用 uv（推荐，更快）
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 方式 2: 使用 pip
pip install -e .

# 3. 编辑配置文件
# 首次使用需要编辑 .env 文件，填入你的 DASHSCOPE_API_KEY
vim .env  # 或使用其他编辑器

# 4. 一键初始化（启动 Docker + 服务 + 上传文档）
make init

# 5. 一键启动
make start
```

#### Windows 环境（PowerShell/CMD）

如果Windows 不支持 `make` 命令，可以手动执行以下步骤以启动服务：

```powershell
# 1. 克隆项目
git clone <repository_url>
cd super_biz_agent_py

# 2. 创建虚拟环境并安装依赖
# 方式 1: 使用 uv（推荐，更快）
pip install uv
# 创建虚拟环境
uv venv
# 激活虚拟环境
.venv\Scripts\activate
# 安装所有依赖
uv pip install -e .

# 方式 2: 使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 编辑配置文件
# 使用记事本或其他编辑器打开 .env 文件，填入你的 DASHSCOPE_API_KEY
notepad .env

# 4. 启动 Docker Desktop
# 确保 Docker Desktop 已安装并正在运行

# 5. 启动 Milvus 向量数据库（Docker Compose）
docker compose -f vector-database.yml up -d

# 6. 等待 Milvus 启动完成（约 5-10 秒）
timeout /t 10

# 7. 启动 MCP 服务
# 启动 CLS 日志查询服务（新开一个 PowerShell 窗口）
python mcp_servers/cls_server.py

# 启动 Monitor 监控服务（新开一个 PowerShell 窗口）
python mcp_servers/monitor_server.py

# 8. 启动 FastAPI 主服务（新开一个 PowerShell 窗口）
# 注意：日志会自动输出到 logs\app_YYYY-MM-DD.log
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 9. 上传文档到向量库（新开一个 PowerShell 窗口）
# 等待服务启动完成后执行
timeout /t 5
python -c "import requests, os, time; [requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')}) or time.sleep(1) for f in os.listdir('aiops-docs') if f.endswith('.md')]"
```

**Windows 一键启动脚本**（推荐）

使用启动脚本：

```powershell
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回，支持 RAG 评测模式 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| 会话历史 | GET | `/api/chat/session/{session_id}` | 查询会话历史 |
| 清除会话 | POST | `/api/chat/clear` | 清除指定会话记忆 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并索引文档（.md/.txt） |
| 目录索引 | POST | `/api/index_directory` | 批量索引目录下所有文档 |
| 健康检查 | GET | `/api/health` | 服务状态 + Milvus 连接检查 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer
```

## 📁 项目结构

```
super_biz_agent/
├── app/                                    # 应用核心
│   ├── __init__.py                         # 包初始化（自动加载日志配置）
│   ├── main.py                             # FastAPI 应用入口
│   ├── config.py                           # 配置管理（Pydantic Settings）
│   ├── api/                                # API 路由层
│   │   ├── chat.py                         # 对话接口（RAG + 会话管理）
│   │   ├── aiops.py                        # AIOps 接口（故障诊断）
│   │   ├── file.py                         # 文件管理（上传 + 目录索引）
│   │   └── health.py                       # 健康检查（含 Milvus 状态）
│   ├── services/                           # 业务服务层
│   │   ├── rag_agent_service.py            # RAG Agent（LangGraph + ChatQwen）
│   │   ├── aiops_service.py                # AIOps Plan-Execute-Replan
│   │   ├── aiops_prompt.py                 # AIOps 诊断提示词构建
│   │   ├── memory_manager.py               # SQLite 会话记忆 + 摘要压缩
│   │   ├── vector_store_manager.py         # Milvus 向量存储管理
│   │   ├── vector_embedding_service.py     # DashScope embedding 服务
│   │   ├── vector_index_service.py         # 文档索引 + Contextual Chunking
│   │   ├── vector_search_service.py        # Milvus 相似度搜索
│   │   ├── document_splitter_service.py    # 文档分割（Markdown 感知）
│   │   ├── hybrid_search_service.py        # 混合检索（Dense + BM25 + RRF）
│   │   ├── keyword_search_service.py       # BM25 稀疏向量编码
│   │   ├── prometheus_alert_service.py     # Prometheus 告警查询
│   │   ├── query_rewrite_pipeline.py       # 查询改写编排（route→rewrite→drift）
│   │   ├── query_router.py                 # 意图路由（Stage1 规则 + Stage2 模型）
│   │   ├── query_rewriter.py               # 三种改写器（decompose/step_back/contextualize）
│   │   ├── drift_guard.py                  # 查询漂移检测（余弦相似度）
│   │   ├── rewrite_model_service.py        # Ollama 改写模型封装
│   │   ├── async_retrieval_service.py      # 并行检索服务
│   │   ├── rag_eval_metrics.py             # RAG 评估指标（Hit@K, MRR）
│   │   └── rag_trace.py                    # RAG 检索路径追踪
│   ├── agent/                              # Agent 模块
│   │   ├── mcp_client.py                   # MCP 客户端（重试 + 全局管理）
│   │   └── aiops/                          # AIOps Agent 核心
│   │       ├── planner.py                  # 计划制定器
│   │       ├── executor.py                 # 步骤执行器（ToolNode）
│   │       ├── replanner.py                # 重规划器
│   │       ├── state.py                    # 状态定义
│   │       └── utils.py                    # 工具描述格式化
│   ├── models/                             # 数据模型（Pydantic）
│   │   ├── aiops.py                        # AIOps 模型
│   │   ├── document.py                     # 文档模型
│   │   ├── memory.py                       # 记忆摘要模型
│   │   ├── request.py                      # 请求模型
│   │   ├── response.py                     # 响应模型
│   │   └── rewrite.py                      # 查询改写模型
│   ├── tools/                              # Agent 工具
│   │   ├── knowledge_tool.py               # 知识库检索 + 格式化
│   │   └── time_tool.py                    # 时间查询
│   ├── core/                               # 核心组件
│   │   ├── llm_factory.py                  # LLM 工厂
│   │   └── milvus_client.py                # Milvus 连接管理
│   └── utils/
│       └── logger.py                       # Loguru 日志配置
├── static/                                 # Web 前端
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── mcp_servers/                            # MCP 服务器
│   ├── cls_server.py                       # CLS 日志查询（模拟数据）
│   ├── monitor_server.py                   # 监控数据（模拟数据）
│   └── README.md
├── scripts/                                # 评测 & 报告脚本
│   ├── run_rag_eval.py                     # RAG 批量评测（支持 rewrite A/B 对比）
│   ├── generate_eval_review.py             # 评测审查报告生成
│   ├── render_rag_eval_excel.py            # 评测结果 Excel 导出
│   ├── render_rag_eval_readable.py         # 评测结果可读报告
│   ├── update_rag_eval_excel_overview.py   # 评测 Excel 汇总更新
│   └── phase0_verify_json.py               # 评测数据 JSON 格式验证
├── docs/                                   # 文档 & 评测报告
│   ├── rag_optimization_2026-05-04.md      # RAG 优化总结
│   ├── query_rewrite_implementation_report.md  # Query Rewrite 实现报告
│   ├── rag_eval_dataset.md                 # 评测数据集说明
│   ├── rag_eval_dataset.json               # 评测数据集（60 题）
│   ├── rag_eval_results_readable.md        # 评测结果可读报告
│   ├── rag_eval_report/                    # 各次评测报告
│   ├── rag_eval_results_*.json/xlsx        # 历次评测结果
│   ├── plans/                              # 优化方案文档
│   └── superpowers/                        # 历史规划文档
├── tests/                                  # 测试
│   └── services/                           # 服务层测试
│       ├── test_hybrid_search_service.py
│       ├── test_keyword_search_service.py
│       ├── test_memory_manager.py
│       ├── test_rag_eval_metrics.py
│       ├── test_rag_eval_regression.py
│       ├── test_aiops_prompt.py            # AIOps 提示词测试
│       ├── test_prometheus_alert_service.py # Prometheus 告警服务测试
│       ├── test_rag_trace.py               # RAG 轨迹追踪测试
│       ├── test_query_router.py            # 意图路由测试（25 用例）
│       ├── test_query_rewriter.py          # 改写器测试（12 用例）
│       ├── test_drift_guard.py             # 漂移检测测试（15 用例）
│       └── test_query_rewrite_pipeline.py  # 管线集成测试（12 用例）
├── aiops-docs/                             # 运维知识库文档
│   ├── cpu_high_usage.md
│   ├── disk_high_usage.md
│   ├── memory_high_usage.md
│   ├── service_unavailable.md
│   └── slow_response.md
├── logs/                                   # 日志目录
├── uploads/                                # 上传文件目录
├── volumes/                                # 数据持久化
│   ├── bm25_model.pkl                      # BM25 模型文件
│   └── memory/                             # SQLite 会话记忆
├── .env                                    # 环境配置
├── Makefile                                # Linux/macOS 管理命令
├── start-windows.bat                       # Windows 启动脚本
├── stop-windows.bat                        # Windows 停止脚本
├── vector-database.yml                     # Milvus Docker Compose
├── pyproject.toml                          # 项目元数据 & 依赖
├── uv.lock                                 # 依赖锁文件
└── README.md
```

## ⚙️ 配置说明

通过 `.env` 文件配置：

```bash
# 应用配置
APP_NAME=SuperBizAgent
DEBUG=True
HOST=0.0.0.0
PORT=9900

# 阿里云 DashScope 配置（必填）
# API Key 管理：https://bailian.console.aliyun.com/
DASHSCOPE_API_KEY=your-api-key-here
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.6-flash
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TIMEOUT=10000

# RAG 配置
RAG_TOP_K=3
RAG_MODEL=qwen3.6-flash

# 文档分块配置
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100

# 混合检索配置（Dense + BM25 + RRF）
HYBRID_SEARCH_ENABLED=true
HYBRID_RRF_K=5
HYBRID_PER_RANKER_LIMIT=10
BM25_MODEL_PATH=volumes/bm25_model.pkl

# Contextual Chunking — 索引时为文档生成上下文摘要
CONTEXTUAL_CHUNKING_ENABLED=true

# 会话记忆配置
MEMORY_WINDOW_TURNS=5
MEMORY_PROMPT_BUDGET_TOKENS=24000
MEMORY_SUMMARY_SOFT_RATIO=0.6
MEMORY_SUMMARY_HARD_RATIO=0.8
MEMORY_RESERVED_OUTPUT_TOKENS=4096
MEMORY_SQLITE_PATH=volumes/memory/session_memory.sqlite3

# MCP 服务配置
MCP_CLS_TRANSPORT=streamable-http
MCP_CLS_URL=http://localhost:8003/mcp
MCP_MONITOR_TRANSPORT=streamable-http
MCP_MONITOR_URL=http://localhost:8004/mcp

# Prometheus 告警配置（可选）
PROMETHEUS_ENABLED=false
PROMETHEUS_BASE_URL=http://localhost:9090
PROMETHEUS_TIMEOUT_SECONDS=5.0

# Query Rewrite 配置（可选，需安装 Ollama）
REWRITE_ENABLED=true
REWRITE_LOCAL_MODEL_NAME=qwen2.5:7b  # 改写模型，默认 qwen2.5:1.5b
REWRITE_LOCAL_MODEL_URL=http://localhost:11434/v1
REWRITE_LOCAL_MODEL_TEMPERATURE=0.1
REWRITE_LOCAL_MODEL_TIMEOUT=10
REWRITE_ROUTER_ENABLED=true
REWRITE_DRIFT_THRESHOLD=0.65
REWRITE_DRIFT_MODERATE_THRESHOLD=0.40
REWRITE_PARALLEL_RETRIEVAL_ENABLED=true
REWRITE_PARALLEL_MAX_WORKERS=4
```

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 核心特性
- ✅ 自动制定诊断计划（Planner）
- ✅ 智能工具调用（Executor）
- ✅ 动态调整步骤（Replanner）
- ✅ 流式输出诊断过程
- ✅ 生成结构化报告

### 快速测试

```bash
# 服务已通过 make init 自动启动
# 如需重启服务：make restart

# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test"}' \
  --no-buffer
```

### 诊断流程
```
1. Planner 制定计划 → 生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续/调整/生成报告
4. 输出诊断报告 → 根因分析 + 运维建议
```

## 📝 开发指南

### 常用命令

```bash
# 项目管理
make init              # 一键初始化（Docker + 服务 + 文档）
make start             # 启动所有服务
make stop              # 停止所有服务
make restart           # 重启所有服务

# 依赖管理
make install-dev       # 安装开发依赖
make sync              # 同步依赖

# Docker 管理
make up                # 启动 Docker 容器
make down              # 停止 Docker 容器

# 代码质量
make format            # 格式化代码
make lint              # 代码检查
```


## 🐛 常见问题

### Windows 环境问题

#### 1. `make` 命令不可用
Windows 不支持 `make` 命令，请使用提供的批处理脚本：
```powershell
# 启动服务
.\start-windows.bat

# 停止服务
.\stop-windows.bat
```

#### 2. PowerShell 执行策略限制
如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 错误：
```powershell
# 临时允许脚本执行（管理员权限）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 或者使用 CMD 而不是 PowerShell
cmd
.\start-windows.bat
```

#### 3. 端口被占用（Windows）
```powershell
# 查看占用端口的进程
netstat -ano | findstr :9900

# 结束进程（替换 PID 为实际进程 ID）
taskkill /F /PID <PID>
```

### 通用问题

### API Key 错误
```bash
# 检查环境变量
cat .env | grep DASHSCOPE_API_KEY    # Linux/macOS
type .env | findstr DASHSCOPE_API_KEY  # Windows
```

### Milvus 连接失败
```bash
# 确保本机有 Docker 服务并且已经启动（可以使用 Docker Desktop）

# 检查 Milvus 状态
docker ps | grep milvus

# 重启 Milvus（使用 docker compose）
docker compose -f vector-database.yml restart

# 或者重启单个服务
docker compose -f vector-database.yml restart standalone
```

### 服务无法启动

**Linux/macOS:**
```bash
# 查看服务日志
tail -f logs/app_$(date +%Y-%m-%d).log  # FastAPI 主服务（Loguru 日志）
tail -f mcp_cls.log                      # CLS MCP 服务
tail -f mcp_monitor.log                  # Monitor MCP 服务

# 检查端口占用
lsof -i :9900  # FastAPI
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

**Windows:**
```powershell
# 查看服务日志（获取今天的日期）
$today = Get-Date -Format "yyyy-MM-dd"
type logs\app_$today.log  # FastAPI 主服务（Loguru 日志）
type mcp_cls.log          # CLS MCP 服务
type mcp_monitor.log      # Monitor MCP 服务

# 或者查看最新的日志文件
Get-ChildItem logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50

# 检查端口占用
netstat -ano | findstr :9900  # FastAPI
netstat -ano | findstr :8003  # CLS MCP
netstat -ano | findstr :8004  # Monitor MCP
```