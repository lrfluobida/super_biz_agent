# SuperBizAgent RAG 评测集

## 概览

- 评测集版本：`superbizagent-rag-eval-v1`
- 生成日期：`2026-03-30`
- 题目总数：`60`
- 来源文档：
  - `aiops-docs/cpu_high_usage.md`
  - `aiops-docs/memory_high_usage.md`
  - `aiops-docs/disk_high_usage.md`
  - `aiops-docs/service_unavailable.md`
  - `aiops-docs/slow_response.md`

这份评测集面向当前仓库里的 AIOps 知识问答 RAG 链路，重点评估：

- 检索是否召回正确知识文档
- 回答是否覆盖关键要点
- 回答是否忠于知识库内容，没有明显编造

结构化数据文件见 [rag_eval_dataset.json](/e:/develop/Python/super_biz_agent/docs/rag_eval_dataset.json)。

## 字段说明

- `id`：题目唯一标识
- `question`：评测问题
- `gold_docs`：标准证据文档，可用于计算检索命中率
- `category`：题型
  - `direct`：直接问答
  - `procedure`：排查流程
  - `scenario`：场景改写
  - `compare`：对比分析
- `difficulty`：难度，分为 `easy`、`medium`、`hard`
- `gold_answer_points`：标准答案要点，可用于回答质量评估

## 题型分布

- 每篇文档 `12` 题，共 `5` 篇文档，总计 `60` 题
- 每篇文档内部按以下结构组织：
  - `3` 题直接问答
  - `4` 题排查流程
  - `3` 题场景改写
  - `2` 题跨文档对比分析

## 使用建议

### 检索层

- 对每道题记录 Top3 或 Top5 检索结果
- 判断 `gold_docs` 是否命中
- 建议统计：
  - `Hit@3`
  - `Hit@5`
  - `MRR`

### 回答层

建议按下面 3 个维度做轻量人工评分：

- `Correctness`
  - `2` 分：回答正确且覆盖关键点
  - `1` 分：基本正确但有遗漏
  - `0` 分：错误或明显答偏
- `Groundedness`
  - `2` 分：回答明显基于检索内容
  - `1` 分：大体有依据但存在泛化
  - `0` 分：存在明显编造
- `Completeness`
  - `2` 分：覆盖大部分 `gold_answer_points`
  - `1` 分：覆盖部分关键点
  - `0` 分：遗漏严重

### 建议汇总指标

- `Hit@3`
- `MRR`
- `Answer Pass Rate`
  - 建议规则：`Correctness >= 1` 且 `Groundedness >= 1`
- `Average Score`
  - 建议使用 `Correctness + Groundedness + Completeness`
- `P95 Latency`

## 文档分组预览

### CPU 使用率过高

- `cpu_01` HighCPUUsage 触发条件与影响
- `cpu_02` CPU 高常见原因
- `cpu_03` CPU 高恢复验证标准
- `cpu_04` CPU 高首轮排查步骤
- `cpu_05` 区分流量突增与死循环
- `cpu_06` 固定时间周期性升高如何排查
- `cpu_07` 业务逻辑简单但 CPU 高如何定位
- `cpu_08` 单实例打满且大量重复错误堆栈
- `cpu_09` 多进程均匀升高且响应变慢
- `cpu_10` 定时任务导致 CPU 告警
- `cpu_11` CPU 高与慢响应的区别
- `cpu_12` CPU 高与内存高的区别

### 内存使用率过高

- `mem_01` HighMemoryUsage 触发条件与风险
- `mem_02` 内存高常见原因
- `mem_03` 内存高恢复验证标准
- `mem_04` 内存高首轮排查步骤
- `mem_05` 区分内存泄漏与短时流量波动
- `mem_06` 出现 OOM 或 GC overhead 后如何处理
- `mem_07` 缓存配置不当如何排查
- `mem_08` Full GC 后内存仍不下降
- `mem_09` 流量上升带来的短时内存高
- `mem_10` 大文件处理导致内存突增
- `mem_11` 内存高与 CPU 高的区别
- `mem_12` 内存问题与服务不可用的关系

### 磁盘使用率过高

- `disk_01` HighDiskUsage 触发阈值与风险
- `disk_02` 磁盘高常见原因
- `disk_03` 磁盘排查常用命令
- `disk_04` 磁盘高首轮排查步骤
- `disk_05` 如何快速定位占用空间的目录和文件
- `disk_06` 出现无法写日志时如何排查
- `disk_07` 临时文件堆积如何处理
- `disk_08` 日志文件过大导致磁盘告警
- `disk_09` 备份文件占满磁盘
- `disk_10` Docker 资源占满磁盘
- `disk_11` 磁盘高与服务不可用的区别
- `disk_12` 磁盘高与慢响应的区别

### 服务不可用

- `svc_01` ServiceUnavailable 触发条件与影响
- `svc_02` 服务不可用常见原因
- `svc_03` 应急时间线
- `svc_04` 服务不可用首轮排查步骤
- `svc_05` 区分应用崩溃与依赖故障
- `svc_06` 数据库连接错误如何排查
- `svc_07` 配置变更后服务不可用如何处理
- `svc_08` 新版本发布后全部实例健康检查失败
- `svc_09` 依赖服务故障导致不可用
- `svc_10` 网络异常导致服务不可用
- `svc_11` 服务不可用与慢响应的区别
- `svc_12` 资源耗尽与磁盘高之间的关系

### 服务响应时间过长

- `slow_01` SlowResponse 触发条件与影响
- `slow_02` 慢响应常见原因
- `slow_03` 慢响应关注的监控指标
- `slow_04` 慢响应首轮排查步骤
- `slow_05` 区分慢查询与外部 API 超时
- `slow_06` 缓存失效或缓存穿透如何判断
- `slow_07` 资源不足场景如何处理
- `slow_08` 慢查询导致的慢响应
- `slow_09` 第三方 API 导致的慢响应
- `slow_10` 缓存命中率下降导致的慢响应
- `slow_11` 慢响应与服务不可用的处理差异
- `slow_12` 慢响应与 CPU 高的关系

## 后续维护建议

- 先用这 60 题做第一版基线评测
- 如果后续新增知识文档，可以继续按每篇 `8` 到 `12` 题扩充
- 每次修改 `chunk_size`、`chunk_overlap`、`top_k` 或 Prompt 后，建议重跑整套评测集
