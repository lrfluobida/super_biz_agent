"""AIOps diagnosis task prompts."""

from textwrap import dedent


def build_aiops_task_prompt(prometheus_enabled: bool) -> str:
    """Build the AIOps diagnosis task prompt."""
    if prometheus_enabled:
        opening = (
            "诊断当前系统是否存在告警。如果存在告警，第一步必须使用 "
            "query_prometheus_alerts 工具从 Prometheus 拉取实时告警信息，"
            "并基于告警信息继续分析根因与处置方案。"
        )
        reminder = "在查询 CPU、内存、日志之前，优先使用 Prometheus 告警工具确定分析起点。"
    else:
        opening = (
            "诊断当前系统是否存在告警。Prometheus 当前未启用，请沿用 monitor 中原本的诊断方式，"
            "优先结合 CPU、内存、日志和其他现有 monitor 工具分析问题。"
        )
        reminder = "Prometheus 未启用时，直接使用现有 monitor 工具链。"

    return dedent(
        f"""{opening}诊断报告输出格式要求：
        ```
        # 告警分析报告

        ---

        ## 活跃告警清单

        | 告警名称 | 目标服务 | 触发时间 | 告警详情 |
        |---------|----------|----------|----------|
        | [告警1名称] | [服务名] | [时间] | [详情] |

        ---

        ## 告警根因分析 - [告警名称]

        ### 告警详情
        - **告警名称**: [名称]
        - **触发时间**: [时间]
        - **告警详情**: [详情]

        ### 症状描述
        [根据监控指标与日志描述症状]

        ### 日志证据
        [引用查询到的关键日志]

        ### 根因结论
        [基于证据得出的根本原因]

        ---

        ## 处理方案 - [告警名称]

        ### 已执行的排查步骤
        1. [步骤1]
        2. [步骤2]

        ### 处理建议
        [给出具体的处理建议]

        ### 预期效果
        [说明预期效果]

        ---

        ## 结论

        ### 整体评估
        [总结所有告警的整体情况]

        ### 关键发现
        - [发现1]
        - [发现2]

        ### 后续建议
        1. [建议1]
        2. [建议2]

        ### 风险评估
        [评估当前风险等级和影响范围]
        ```

        重要提醒：
        - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
        - 所有内容必须基于工具查询到的真实数据，严禁编造
        - 如果某个步骤失败，在结论中如实说明，不要跳过
        - {reminder}
        """
    ).strip()
