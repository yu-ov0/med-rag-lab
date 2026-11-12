"""Calibrate the reranker refusal threshold.

We compare the top-1 rerank score on answerable questions against questions
that are outside the corpus. A good threshold keeps most answerable questions
above it while rejecting most unanswerable ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import build_retriever

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
UNANSWERABLE = ROOT / "eval" / "unanswerable.jsonl"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(name: str, values: list[float]) -> None:
    values = sorted(values)
    n = len(values)

    def pct(p: float) -> float:
        return values[min(n - 1, round(p / 100 * (n - 1)))]

    print(
        f"{name:<14} n={n:<3} min={values[0]:7.3f} p10={pct(10):7.3f} "
        f"p50={pct(50):7.3f} p90={pct(90):7.3f} max={values[-1]:7.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = build_retriever(args.retriever)
    answerable = load(EVAL_SET)
    unanswerable = load(UNANSWERABLE)

    pos = [retriever.search(q["question"], top_k=args.top_k)[0]["score"] for q in answerable]
    neg = [retriever.search(q["question"], top_k=args.top_k)[0]["score"] for q in unanswerable]

    print("top-1 score distribution")
    describe("answerable", pos)
    describe("unanswerable", neg)

    print("\nthreshold  false_refusal  false_answer")
    candidates = sorted({round(x, 2) for x in neg + pos})
    for t in candidates:
        fr = sum(1 for s in pos if s < t) / len(pos)
        fa = sum(1 for s in neg if s >= t) / len(neg)
        print(f"{t:9.2f}  {fr:13.2%}  {fa:12.2%}")


if __name__ == "__main__":
    main()
