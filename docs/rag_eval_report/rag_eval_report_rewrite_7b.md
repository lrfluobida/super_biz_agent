# SuperBizAgent RAG 评测结果报告

- 评测数据集：`superbizagent-rag-eval-v1`
- 评测模式：`normal`
- 查询改写：`True`
- 评测题数：`60`
- Hit@3：`73.33%`
- Hit@5：`73.33%`
- MRR：`0.6556`
- 平均要点命中率：`48.06%`
- 平均响应时延：`12.24s`
- P95 响应时延：`24.50s`
- 召回路径分布：{'both': 100, 'keyword': 3, 'dense': 3}
- 改写意图分布：{'Intent.DECOMPOSE': 29, 'Intent.DIRECT': 29, 'Intent.STEP_BACK': 2}

完整逐题结果见对应的 JSON 文件。