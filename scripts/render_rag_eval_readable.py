from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "docs" / "rag_eval_results.json"
OUTPUT_PATH = REPO_ROOT / "docs" / "rag_eval_results_readable.md"


CATEGORY_NAMES = {
    "cpu": "CPU 告警",
    "mem": "内存告警",
    "disk": "磁盘告警",
    "slow": "慢响应",
    "svc": "服务不可用",
}


def load_payload() -> dict:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def score_label(point_hit_rate: float, retrieval_rank: int | None) -> str:
    if retrieval_rank is None:
        return "高优先级：漏召"
    if point_hit_rate == 1.0:
        return "通过"
    if point_hit_rate >= 0.5:
        return "可接受，建议人工复核"
    return "高优先级：答案不足"


def review_focus(item: dict) -> str:
    point_hit_rate = float(item["point_hit_rate"])
    retrieval_rank = item["retrieval_rank"]
    latency = float(item["latency_seconds"])

    focus: list[str] = []
    if retrieval_rank is None:
        focus.append("先看检索是否漏召")
    elif retrieval_rank and retrieval_rank > 3:
        focus.append("先看相关文档排序是否偏后")

    if point_hit_rate == 0:
        focus.append("重点看答案是否答偏/幻觉扩写")
    elif point_hit_rate < 0.5:
        focus.append("重点看关键要点缺失")
    elif point_hit_rate < 1.0:
        focus.append("重点看是否遗漏部分要点")

    if latency >= 8:
        focus.append("顺带关注时延偏高")

    return "；".join(focus) if focus else "整体表现稳定，可抽样复核"


def manual_verdict_template() -> str:
    return "\n".join(
        [
            "- 人工结论：`待评审`",
            "- 是否可直接上线：`待定`",
            "- 主要问题：`待补充`",
            "- 修订建议：`待补充`",
        ]
    )


def render_summary(payload: dict) -> list[str]:
    summary = payload["summary"]
    results = payload["results"]

    retrieval_miss = sum(1 for item in results if item["retrieval_rank"] is None)
    low_point = sum(1 for item in results if float(item["point_hit_rate"]) < 0.5)
    mid_point = sum(1 for item in results if 0.5 <= float(item["point_hit_rate"]) < 1.0)
    perfect = sum(1 for item in results if float(item["point_hit_rate"]) == 1.0)

    lines = [
        "# RAG 评测结果可读版",
        "",
        "这份文档把 `docs/rag_eval_results.json` 转成了更适合人工评审的格式。",
        "",
        "## 总览",
        "",
        f"- 数据集：`{payload['dataset']}`",
        f"- 题目数：`{summary['total']}`",
        f"- 检索配置：`top_k={payload['retrieval_k']}`",
        f"- Hit@3：`{pct(summary['hit_at_3'])}`",
        f"- Hit@5：`{pct(summary['hit_at_5'])}`",
        f"- MRR：`{summary['mrr']:.4f}`",
        f"- 平均要点命中率：`{pct(summary['avg_point_hit_rate'])}`",
        f"- 平均时延：`{summary['avg_latency_seconds']:.2f}s`",
        f"- P95 时延：`{summary['p95_latency_seconds']:.2f}s`",
        "",
        "## 自动评测分布",
        "",
        f"- 漏召样本：`{retrieval_miss}`",
        f"- 要点命中率 < 50%：`{low_point}`",
        f"- 50% <= 要点命中率 < 100%：`{mid_point}`",
        f"- 满分样本：`{perfect}`",
        "",
        "## 人工评审建议",
        "",
        "推荐人工优先级：",
        "",
        "1. 先看 `retrieval_rank = null` 的漏召样本。",
        "2. 再看 `point_hit_rate = 0` 的答偏样本。",
        "3. 最后抽样复核 `point_hit_rate = 0.5~0.75` 的边界样本。",
        "",
        "建议人工评审时重点看 4 个问题：",
        "",
        "- 检索是否召回了正确知识。",
        "- 回答是否真的覆盖了 gold points。",
        "- 回答是否出现了无依据扩写。",
        "- 回答结构是否适合最终用户直接使用。",
        "",
        "## 高优先级样本",
        "",
        "| ID | 问题 | 检索 | 要点命中率 | 时延 | 评审重点 |",
        "|---|---|---:|---:|---:|---|",
    ]

    priority_items = sorted(
        results,
        key=lambda item: (float(item["point_hit_rate"]), -(float(item["latency_seconds"]))),
    )[:12]
    for item in priority_items:
        retrieval_rank = item["retrieval_rank"]
        retrieval_text = "漏召" if retrieval_rank is None else str(retrieval_rank)
        lines.append(
            "| {id} | {question} | {retrieval} | {point} | {latency:.2f}s | {focus} |".format(
                id=item["id"],
                question=item["question"].replace("|", "\\|"),
                retrieval=retrieval_text,
                point=pct(float(item["point_hit_rate"])),
                latency=float(item["latency_seconds"]),
                focus=review_focus(item).replace("|", "\\|"),
            )
        )

    return lines


def render_category_overview(results: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[item["id"].split("_")[0]].append(item)

    lines = [
        "",
        "## 分类视图",
        "",
        "| 类别 | 题数 | 平均要点命中率 | 漏召数 |",
        "|---|---:|---:|---:|",
    ]

    for key in sorted(grouped):
        items = grouped[key]
        avg = sum(float(item["point_hit_rate"]) for item in items) / len(items)
        miss = sum(1 for item in items if item["retrieval_rank"] is None)
        lines.append(
            f"| {CATEGORY_NAMES.get(key, key)} | {len(items)} | {pct(avg)} | {miss} |"
        )

    return lines


def render_item(item: dict) -> list[str]:
    retrieved_docs = item.get("retrieved_docs", [])
    gold_points = item.get("gold_answer_points", [])
    matched_points = set(item.get("matched_points", []))

    top_docs = []
    for doc in retrieved_docs[:3]:
        top_docs.append(
            f"`#{doc.get('rank', '-')}` {doc.get('file_name', '-')}"
        )
    top_docs_text = "、".join(top_docs) if top_docs else "无"

    lines = [
        f"### {item['id']} - {item['question']}",
        "",
        f"- 自动结论：`{score_label(float(item['point_hit_rate']), item['retrieval_rank'])}`",
        f"- 检索排名：`{item['retrieval_rank'] if item['retrieval_rank'] is not None else '漏召'}`",
        f"- Hit@3 / Hit@5：`{item['hit_at_3']}` / `{item['hit_at_5']}`",
        f"- Reciprocal Rank：`{float(item['reciprocal_rank']):.2f}`",
        f"- 要点命中率：`{pct(float(item['point_hit_rate']))}`",
        f"- 响应时延：`{float(item['latency_seconds']):.2f}s`",
        f"- Gold 文档：`{', '.join(item.get('gold_docs', []))}`",
        f"- Top-3 召回：{top_docs_text}",
        f"- 建议人工关注：{review_focus(item)}",
        "",
        manual_verdict_template(),
        "",
        "<details>",
        "<summary>查看答案、gold points 与召回详情</summary>",
        "",
        "**Gold Points**",
        "",
    ]

    for point in gold_points:
        status = "x" if point in matched_points else " "
        lines.append(f"- [{status}] {point}")

    lines.extend(
        [
            "",
            "**模型回答**",
            "",
            "```markdown",
            item["answer"].rstrip(),
            "```",
            "",
            "**召回详情（前 5 条）**",
            "",
        ]
    )

    if retrieved_docs:
        for doc in retrieved_docs[:5]:
            headers = " / ".join(doc.get("headers", [])) or "-"
            queries = "；".join(doc.get("queries", [])) or "-"
            preview = str(doc.get("preview", "")).replace("\n", " ").strip()
            lines.extend(
                [
                    f"- Rank `{doc.get('rank', '-')}`: `{doc.get('file_name', '-')}`",
                    f"  headers: {headers}",
                    f"  queries: {queries}",
                    f"  preview: {preview}",
                ]
            )
    else:
        lines.append("- 无召回文档")

    lines.extend(
        [
            "",
            "</details>",
            "",
        ]
    )

    return lines


def render_details(results: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[item["id"].split("_")[0]].append(item)

    lines = ["## 逐题评审卡", ""]
    for key in sorted(grouped):
        lines.append(f"## {CATEGORY_NAMES.get(key, key)}")
        lines.append("")
        items = sorted(
            grouped[key],
            key=lambda item: (float(item["point_hit_rate"]), float(item["latency_seconds"]) * -1),
        )
        for item in items:
            lines.extend(render_item(item))

    return lines


def main() -> None:
    payload = load_payload()
    results = payload["results"]

    lines: list[str] = []
    lines.extend(render_summary(payload))
    lines.extend(render_category_overview(results))
    lines.append("")
    lines.extend(render_details(results))

    OUTPUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
