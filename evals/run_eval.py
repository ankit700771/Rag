import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
)
from app.db import SessionLocal
from app.retrieval.search import search

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.json")
CONFIGS = [
    ("vector", "vector", False),
    ("keyword", "keyword", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]
KS = (1, 3, 5)


def dedupe_sources(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))


def evaluate_config(session, dataset: list[dict], mode: str, use_rerank: bool) -> dict:
    ranks: list[int | None] = []
    latencies: list[float] = []
    for item in dataset:
        started = time.perf_counter()
        hits = search(session, item["question"], top_k=max(KS) * 2, mode=mode, use_rerank=use_rerank)
        latencies.append((time.perf_counter() - started) * 1000)
        sources = dedupe_sources([h.source for h in hits])
        ranks.append(first_relevant_rank(sources, item["expected"]))
    result = {f"recall@{k}": round(recall_at_k(ranks, k), 3) for k in KS}
    result["mrr"] = round(mean_reciprocal_rank(ranks), 3)
    result["latency_p50_ms"] = round(percentile(latencies, 50), 1)
    result["latency_p95_ms"] = round(percentile(latencies, 95), 1)
    result["misses"] = [
        item["question"] for item, rank in zip(dataset, ranks) if rank is None or rank > max(KS)
    ]
    return result


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    report: dict[str, dict] = {}
    with SessionLocal() as session:
        search(session, dataset[0]["question"], top_k=1, mode="hybrid", use_rerank=True)
        for name, mode, use_rerank in CONFIGS:
            report[name] = evaluate_config(session, dataset, mode, use_rerank)

    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = f"{'config':<16}" + "".join(f"{'recall@' + str(k):>10}" for k in KS)
    header += f"{'mrr':>8}{'p50 ms':>10}{'p95 ms':>10}"
    print(f"questions: {len(dataset)}")
    print(header)
    for name, metrics in report.items():
        row = f"{name:<16}" + "".join(f"{metrics[f'recall@{k}']:>10.3f}" for k in KS)
        row += f"{metrics['mrr']:>8.3f}{metrics['latency_p50_ms']:>10.1f}{metrics['latency_p95_ms']:>10.1f}"
        print(row)


if __name__ == "__main__":
    main()
