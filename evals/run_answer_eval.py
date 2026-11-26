import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import SessionLocal
from app.generation.gemini import GenerationUnavailable, get_generator
from app.generation.rag import answer_question
from evals.metrics import cited_expected_source, is_refusal, percentile

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("answer_results.json")
CHECKPOINT_PATH = Path(__file__).with_name("answer_checkpoint.json")
IN_SCOPE_STEP = 3
MAX_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 60
PACING_SECONDS = 13
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 football world cup?",
    "How do I configure a Kubernetes ingress controller in Rust?",
]


def pick_in_scope(dataset: list[dict]) -> list[dict]:
    return dataset[::IN_SCOPE_STEP]


def load_checkpoint(fresh: bool) -> dict[str, dict]:
    if fresh or not CHECKPOINT_PATH.exists():
        return {}
    rows = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {question: row for question, row in rows.items() if "error" not in row}


def save_checkpoint(rows: dict[str, dict]) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_question(session, generator, question: str) -> dict:
    started = time.perf_counter()
    result = None
    error = ""
    for _ in range(MAX_ATTEMPTS):
        try:
            result = answer_question(session, generator, question)
            break
        except GenerationUnavailable as exc:
            error = str(exc)
            time.sleep(RETRY_DELAY_SECONDS)
    if result is None:
        return {"question": question, "error": error}
    row = {
        "question": question,
        "answer": result.answer,
        "sources": [hit.source for hit in result.sources],
        "cited": result.cited,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    time.sleep(PACING_SECONDS)
    return row


def rate(part: list, whole: list) -> float:
    return round(len(part) / len(whole), 3) if whole else 0.0


def summarize(in_scope: list[dict], off_topic: list[dict]) -> dict:
    scored = [r for r in in_scope if "error" not in r]
    answered = [r for r in scored if not is_refusal(r["answer"])]
    cited = [r for r in answered if r["cited"]]
    grounded = [
        r for r in cited if cited_expected_source(r["cited"], r["sources"], r["expected"])
    ]
    off_scored = [r for r in off_topic if "error" not in r]
    refused = [r for r in off_scored if is_refusal(r["answer"])]
    latencies = [r["latency_ms"] for r in scored + off_scored]
    return {
        "in_scope_questions": len(in_scope),
        "in_scope_errors": len(in_scope) - len(scored),
        "answer_rate": rate(answered, scored),
        "citation_rate": rate(cited, answered),
        "cited_expected_source_rate": rate(grounded, answered),
        "off_topic_questions": len(off_topic),
        "off_topic_errors": len(off_topic) - len(off_scored),
        "off_topic_refusal_rate": rate(refused, off_scored),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    dataset = pick_in_scope(json.loads(DATASET_PATH.read_text(encoding="utf-8")))
    generator = get_generator()
    done = load_checkpoint(args.fresh)
    if done:
        print(f"resuming with {len(done)} answered questions from checkpoint")

    in_scope: list[dict] = []
    off_topic: list[dict] = []
    with SessionLocal() as session:
        for item in dataset:
            question = item["question"]
            row = done.get(question) or run_question(session, generator, question)
            row["expected"] = item["expected"]
            done[question] = row
            save_checkpoint(done)
            in_scope.append(row)
            print(f"in-scope  {'ERR' if 'error' in row else 'ok '} {question}", flush=True)
        for question in OFF_TOPIC:
            row = done.get(question) or run_question(session, generator, question)
            done[question] = row
            save_checkpoint(done)
            off_topic.append(row)
            print(f"off-topic {'ERR' if 'error' in row else 'ok '} {question}", flush=True)

    summary = summarize(in_scope, off_topic)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "in_scope": in_scope, "off_topic": off_topic}, indent=2),
        encoding="utf-8",
    )
    print()
    for key, value in summary.items():
        print(f"{key:<30}{value}")


if __name__ == "__main__":
    main()
