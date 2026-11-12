"""Measure retrieval latency distribution over the eval questions."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"


def load_questions() -> list[dict]:
    import json

    with EVAL_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="hybrid_rerank")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    from run_eval import build_retriever

    retriever = build_retriever(args.retriever)
    questions = load_questions()
    for q in questions[: args.warmup]:
        retriever.search(q["question"], top_k=5)

    timings: list[float] = []
    for _ in range(args.repeat):
        for q in questions:
            started = time.perf_counter()
            retriever.search(q["question"], top_k=5)
            timings.append((time.perf_counter() - started) * 1000)

    timings.sort()
    n = len(timings)

    def pct(p: float) -> float:
        return timings[min(n - 1, round(p / 100 * (n - 1)))]

    print(f"retriever={args.retriever} n={n}")
    print(f"mean={statistics.mean(timings):.2f} ms")
    print(f"p50={pct(50):.2f} ms")
    print(f"p95={pct(95):.2f} ms")
    print(f"max={timings[-1]:.2f} ms")


if __name__ == "__main__":
    main()
