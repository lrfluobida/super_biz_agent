# SuperBizAgent RAG 评测结果报告

- 评测数据集：`superbizagent-rag-eval-v1`
- 评测模式：`eval (config)`
- 查询改写：`True`
- 评测题数：`60`
- Hit@3：`100.00%`
- Hit@5：`100.00%`
- MRR：`0.8722`
- 平均要点命中率：`43.47%`
- 平均响应时延：`12.08s`
- P95 响应时延：`18.30s`
- 召回路径分布：{'both': 164, 'keyword': 14, 'dense': 2}
- 改写意图分布：{'Intent.DIRECT': 48, 'Intent.DECOMPOSE': 10, 'Intent.STEP_BACK': 2}

完整逐题结果见对应的 JSON 文件。