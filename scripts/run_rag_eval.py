"""复用型 RAG 评测脚本。

用法示例：
python scripts/run_rag_eval.py --base-url http://127.0.0.1:9900 --retrieval-k 5
python scripts/run_rag_eval.py --mode normal  # 走正常 Agent 流程，不触发评测链路
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
    return "".join(ch.lower() for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


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
    return parser.parse_args()


def request_answer(
    base_url: str,
    question: str,
    retrieval_k: int,
    use_hybrid: bool | None = None,
    eval_mode: bool = True,
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


def build_report(results_payload: dict) -> str:
    summary = results_payload["summary"]
    lines = [
        "# SuperBizAgent RAG 评测结果报告",
        "",
        f"- 评测数据集：`{results_payload['dataset']}`",
        f"- 评测模式：`{summary.get('mode', 'unknown')}`",
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

    lines += [
        "",
        "完整逐题结果见对应的 JSON 文件。",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    with open(args.dataset, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    eval_mode = args.mode == "eval"
    results = []
    retrieval_ranks: list[int | None] = []
    latencies: list[float] = []

    for item in dataset["items"]:
        response, latency = request_answer(
            args.base_url, item["question"], args.retrieval_k,
            use_hybrid=args.hybrid, eval_mode=eval_mode,
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
            }
        )

    mode_label = f"normal (agent)" if not eval_mode else (
        f"hybrid={args.hybrid}" if args.hybrid is not None else "eval (config)"
    )

    # 汇总全局 recall_path 分布
    global_path_stats: dict[str, int] = {}
    for item in results:
        for path, count in item.get("recall_path_stats", {}).items():
            global_path_stats[path] = global_path_stats.get(path, 0) + count

    summary = {
        "mode": mode_label,
        "total": len(results),
        "hit_at_3": sum(hit_at_k(rank, 3) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "hit_at_5": sum(hit_at_k(rank, 5) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "mrr": sum(reciprocal_rank(rank) for rank in retrieval_ranks) / max(len(retrieval_ranks), 1),
        "avg_point_hit_rate": sum(item["point_hit_rate"] for item in results) / max(len(results), 1),
        "avg_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "hybrid_search": args.hybrid if args.hybrid is not None else "config",
        "recall_path_distribution": global_path_stats,
    }

    payload = {
        "dataset": dataset["name"],
        "base_url": args.base_url,
        "retrieval_k": args.retrieval_k,
        "mode": args.mode,
        "hybrid_search": args.hybrid if args.hybrid is not None else "config",
        "summary": summary,
        "results": results,
    }

    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_report).write_text(build_report(payload), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
