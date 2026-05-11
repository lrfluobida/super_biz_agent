"""复用型 RAG 评测脚本。

用法示例：
  python scripts/run_rag_eval.py --base-url http://127.0.0.1:9900 --retrieval-k 5
  python scripts/run_rag_eval.py --mode normal
  python scripts/run_rag_eval.py --rewrite            # 开启查询改写
  python scripts/run_rag_eval.py --compare-rewrite    # A/B 对比：改写 vs 不改写
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.rag_eval_metrics import find_first_relevant_rank, hit_at_k, reciprocal_rank


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or "一" <= ch <= "鿿")


def char_bigrams(text: str) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index:index + 2] for index in range(len(normalized) - 1)}


def point_matched(point: str, answer: str) -> bool:
    normalized_point = normalize_text(point)
    normalized_answer = normalize_text(answer)
    if not normalized_point or not normalized_answer:
        return False
    if normalized_point in normalized_answer:
        return True
    point_bigrams = char_bigrams(point)
    overlap = len(point_bigrams & char_bigrams(answer)) / max(len(point_bigrams), 1)
    return overlap >= 0.62


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    lower_weight = upper - index
    upper_weight = index - lower
    return sorted_values[lower] * lower_weight + sorted_values[upper] * upper_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SuperBizAgent RAG evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:9900")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "docs" / "rag_eval_dataset.json"))
    parser.add_argument("--output-json", default=str(REPO_ROOT / "docs" / "rag_eval_results.json"))
    parser.add_argument("--output-report", default=str(REPO_ROOT / "docs" / "rag_eval_report.md"))
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=None,
        dest="hybrid",
        help="启用混合检索（二路召回+RRF）",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_false",
        default=None,
        dest="hybrid",
        help="禁用混合检索，仅用纯向量召回",
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "normal"],
        default="eval",
        help="评测模式：eval=走确定性的先检索后生成链路（默认），normal=走正常 Agent 流程",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        default=None,
        dest="rewrite",
        help="启用查询改写",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_false",
        default=None,
        dest="rewrite",
        help="禁用查询改写",
    )
    parser.add_argument(
        "--compare-rewrite",
        action="store_true",
        default=False,
        help="A/B 对比模式：分别跑改写和不改写，输出对比报告",
    )
    return parser.parse_args()


def request_answer(
    base_url: str,
    question: str,
    retrieval_k: int,
    use_hybrid: bool | None = None,
    eval_mode: bool = True,
    rewrite_enabled: bool | None = None,
) -> tuple[dict, float]:
    payload = json.dumps(
        {
            "Id": f"eval-{uuid.uuid4().hex[:12]}",
            "Question": question,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if eval_mode:
        headers["X-RAG-Eval-Mode"] = "true"
        headers["X-RAG-Eval-Top-K"] = str(retrieval_k)
        if use_hybrid is not None:
            headers["X-RAG-Hybrid-Search"] = "true" if use_hybrid else "false"
    if rewrite_enabled is not None:
        headers["X-RAG-Rewrite-Enabled"] = "true" if rewrite_enabled else "false"

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        method="POST",
        headers=headers,
    )
    started_at = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, round(time.perf_counter() - started_at, 3)


def run_eval_pass(
    dataset: dict,
    args: argparse.Namespace,
    rewrite_enabled: bool | None,
    pass_label: str,
) -> tuple[list[dict], dict]:
    """执行一轮评测，返回 (详细结果列表, 汇总 dict)"""
    eval_mode = args.mode == "eval"
    results = []
    retrieval_ranks: list[int | None] = []
    latencies: list[float] = []
    rewrite_stats: dict[str, int] = {}

    for item in dataset["items"]:
        response, latency = request_answer(
            args.base_url, item["question"], args.retrieval_k,
            use_hybrid=args.hybrid, eval_mode=eval_mode,
            rewrite_enabled=rewrite_enabled,
        )
        data = response.get("data", {}) or {}
        answer = data.get("answer") or ""
        evaluation = data.get("evaluation", {}) if data else {}
        if not answer:
            error_msg = data.get("errorMessage", "unknown") or "unknown"
            answer = f"[ERROR: {error_msg}]"

        matched_points = [point for point in item["gold_answer_points"] if point_matched(point, answer)]
        point_hit_rate = len(matched_points) / max(len(item["gold_answer_points"]), 1)
        latencies.append(latency)

        retrieved_docs = evaluation.get("retrieved_docs", [])
        recall_meta = evaluation.get("recall_meta", {})
        recall_path_stats = evaluation.get("recall_path_stats", {})
        rank = find_first_relevant_rank(item["gold_docs"], retrieved_docs)
        retrieval_ranks.append(rank)

        # 提取改写信息
        rewrite_info = evaluation.get("rewrite") or {}
        if rewrite_info:
            intent = rewrite_info.get("intent", "unknown")
            rewrite_stats[intent] = rewrite_stats.get(intent, 0) + 1

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_docs": item["gold_docs"],
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "retrieval_rank": rank,
                "hit_at_3": hit_at_k(rank, 3),
                "hit_at_5": hit_at_k(rank, 5),
                "reciprocal_rank": reciprocal_rank(rank),
                "gold_answer_points": item["gold_answer_points"],
                "matched_points": matched_points,
                "point_hit_rate": round(point_hit_rate, 4),
                "latency_seconds": latency,
                "recall_mode": recall_meta.get("mode", "unknown"),
                "recall_path_stats": recall_path_stats,
                "mode": "normal" if not eval_mode else "eval",
                "rewrite": rewrite_info,
            }
        )

    # 汇总全局 recall_path 分布
    global_path_stats: dict[str, int] = {}
    for item in results:
        for path, count in item.get("recall_path_stats", {}).items():
            global_path_stats[path] = global_path_stats.get(path, 0) + count

    summary = {
        "pass": pass_label,
        "rewrite_enabled": rewrite_enabled,
        "total": len(results),
        "hit_at_3": sum(hit_at_k(rank, 3) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "hit_at_5": sum(hit_at_k(rank, 5) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "mrr": sum(reciprocal_rank(rank) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "avg_point_hit_rate": sum(item["point_hit_rate"] for item in results) / max(len(results), 1),
        "avg_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "hybrid_search": args.hybrid if args.hybrid is not None else "config",
        "recall_path_distribution": global_path_stats,
        "rewrite_intent_distribution": rewrite_stats if rewrite_stats else {},
    }

    return results, summary


def build_compare_report(summary_a: dict, summary_b: dict) -> str:
    """生成 A/B 对比报告"""
    def _delta(key: str) -> str:
        a = summary_a.get(key, 0)
        b = summary_b.get(key, 0)
        if isinstance(a, float) and isinstance(b, float):
            diff = a - b
            sign = "+" if diff > 0 else ""
            pct = (diff / max(b, 0.001)) * 100 if b else 0
            return f"{sign}{diff:.4f} ({sign}{pct:.1f}%)"
        return f"{a} vs {b}"

    lines = [
        "# SuperBizAgent RAG 查询改写 A/B 评测报告",
        "",
        f"## 对比总览",
        "",
        f"| 指标 | 不改写 (baseline) | 改写 (rewrite) | 变化 |",
        f"|------|-------------------|----------------|------|",
        f"| Hit@3 | {summary_a['hit_at_3']:.2%} | {summary_b['hit_at_3']:.2%} | {_delta('hit_at_3')} |",
        f"| Hit@5 | {summary_a['hit_at_5']:.2%} | {summary_b['hit_at_5']:.2%} | {_delta('hit_at_5')} |",
        f"| MRR | {summary_a['mrr']:.4f} | {summary_b['mrr']:.4f} | {_delta('mrr')} |",
        f"| 平均要点命中率 | {summary_a['avg_point_hit_rate']:.2%} | {summary_b['avg_point_hit_rate']:.2%} | {_delta('avg_point_hit_rate')} |",
        f"| 平均时延 | {summary_a['avg_latency_seconds']:.2f}s | {summary_b['avg_latency_seconds']:.2f}s | {_delta('avg_latency_seconds')} |",
        f"| P95 时延 | {summary_a['p95_latency_seconds']:.2f}s | {summary_b['p95_latency_seconds']:.2f}s | {_delta('p95_latency_seconds')} |",
        "",
        f"## 改写统计",
        f"",
        f"- 改写意图分布：{summary_b.get('rewrite_intent_distribution', {})}",
        "",
        "## 说明",
        "",
        "- `+` 表示改写后指标提升，`-` 表示改写后指标下降",
        "- 时延增加属于预期行为（改写模型推理需要 ~1-3s）",
    ]
    return "\n".join(lines)


def build_report(results_payload: dict) -> str:
    summary = results_payload["summary"]
    lines = [
        "# SuperBizAgent RAG 评测结果报告",
        "",
        f"- 评测数据集：`{results_payload['dataset']}`",
        f"- 评测模式：`{summary.get('mode', 'unknown')}`",
        f"- 查询改写：`{summary.get('rewrite_enabled', 'config')}`",
        f"- 评测题数：`{summary['total']}`",
        f"- Hit@3：`{summary['hit_at_3']:.2%}`",
        f"- Hit@5：`{summary['hit_at_5']:.2%}`",
        f"- MRR：`{summary['mrr']:.4f}`",
        f"- 平均要点命中率：`{summary['avg_point_hit_rate']:.2%}`",
        f"- 平均响应时延：`{summary['avg_latency_seconds']:.2f}s`",
        f"- P95 响应时延：`{summary['p95_latency_seconds']:.2f}s`",
    ]

    path_dist = summary.get("recall_path_distribution", {})
    if path_dist:
        lines.append(f"- 召回路径分布：{path_dist}")

    rewrite_dist = summary.get("rewrite_intent_distribution", {})
    if rewrite_dist:
        lines.append(f"- 改写意图分布：{rewrite_dist}")

    lines += [
        "",
        "完整逐题结果见对应的 JSON 文件。",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    with open(args.dataset, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.compare_rewrite:
        # A/B 对比模式
        print("=" * 60)
        print("A/B 对比评测：不改写 (baseline) vs 改写 (rewrite)")
        print("=" * 60)

        print("\n>>> Pass A: 不改写 (baseline)")
        results_a, summary_a = run_eval_pass(
            dataset, args, rewrite_enabled=False, pass_label="baseline",
        )
        print(json.dumps(summary_a, ensure_ascii=False, indent=2))

        print("\n>>> Pass B: 改写 (rewrite)")
        results_b, summary_b = run_eval_pass(
            dataset, args, rewrite_enabled=True, pass_label="rewrite",
        )
        print(json.dumps(summary_b, ensure_ascii=False, indent=2))

        # 逐题对比
        compare_results = []
        for ra, rb in zip(results_a, results_b):
            compare_results.append({
                "id": ra["id"],
                "question": ra["question"],
                "baseline": {
                    "hit_at_3": ra["hit_at_3"],
                    "hit_at_5": ra["hit_at_5"],
                    "point_hit_rate": ra["point_hit_rate"],
                    "latency_seconds": ra["latency_seconds"],
                },
                "rewrite": {
                    "hit_at_3": rb["hit_at_3"],
                    "hit_at_5": rb["hit_at_5"],
                    "point_hit_rate": rb["point_hit_rate"],
                    "latency_seconds": rb["latency_seconds"],
                    "rewrite_info": rb.get("rewrite"),
                },
                "category": dataset["items"][len(compare_results)].get("category", ""),
                "rewrite_expected": dataset["items"][len(compare_results)].get("rewrite_expected"),
            })

        compare_payload = {
            "dataset": dataset["name"],
            "base_url": args.base_url,
            "retrieval_k": args.retrieval_k,
            "mode": args.mode,
            "summary_baseline": summary_a,
            "summary_rewrite": summary_b,
            "compare_results": compare_results,
        }

        compare_json = str(REPO_ROOT / "docs" / "rag_eval_results_compare.json")
        compare_report = str(REPO_ROOT / "docs" / "rag_eval_report_compare.md")
        Path(compare_json).write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(compare_report).write_text(build_compare_report(summary_a, summary_b), encoding="utf-8")

        print(f"\n对比报告已保存: {compare_report}")
        print(f"对比数据已保存: {compare_json}")

    else:
        # 单次评测模式
        rewrite_enabled = args.rewrite  # None = 使用服务端配置, True/False = 显式覆盖
        results, summary = run_eval_pass(
            dataset, args, rewrite_enabled=rewrite_enabled, pass_label="single",
        )

        summary["mode"] = "normal" if args.mode == "normal" else (
            f"hybrid={args.hybrid}" if args.hybrid is not None else "eval (config)"
        )

        payload = {
            "dataset": dataset["name"],
            "base_url": args.base_url,
            "retrieval_k": args.retrieval_k,
            "mode": args.mode,
            "rewrite_enabled": rewrite_enabled,
            "hybrid_search": args.hybrid if args.hybrid is not None else "config",
            "summary": summary,
            "results": results,
        }

        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.output_report).write_text(build_report(payload), encoding="utf-8")

        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
