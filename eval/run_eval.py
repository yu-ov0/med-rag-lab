"""Run a retrieval ablation over the hand-built evaluation set.

Usage:
    python eval/run_eval.py --retriever bm25
    python eval/run_eval.py --retriever dense
    python eval/run_eval.py --retriever hybrid
    python eval/run_eval.py --retriever hybrid_rerank

Results are printed and appended to eval/results/<retriever>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import (
    hit_rank,
    ndcg_at_k,
    phrase_rank,
    recall_at_k,
    reciprocal_rank,
    summarize,
)

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_retriever(name: str):
    if name == "bm25":
        from retrieval.bm25 import BM25Retriever

        return BM25Retriever()
    if name == "dense":
        from retrieval.dense import DenseRetriever

        return DenseRetriever()
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever()
    if name == "hybrid_rerank":
        from retrieval.hybrid import HybridRetriever
        from retrieval.rerank import Reranker

        return Reranker(HybridRetriever())
    raise SystemExit(f"unknown retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    questions = load_jsonl(EVAL_SET)
    retriever = build_retriever(args.retriever)

    rows: list[dict] = []
    started = time.perf_counter()
    for q in questions:
        hits = retriever.search(q["question"], top_k=args.top_k)
        ranked = [h["chunk_id"] for h in hits]
        prow = phrase_rank(hits, q["gold_phrase"])
        row = {"qid": q["qid"], "question": q["question"], "gold": q["gold_chunk_id"]}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(ranked, q["gold_chunk_id"], k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, q["gold_chunk_id"], k)
            row[f"phrase_recall@{k}"] = 1.0 if prow and prow <= k else 0.0
        row["mrr"] = reciprocal_rank(ranked, q["gold_chunk_id"])
        row["rank"] = hit_rank(ranked, q["gold_chunk_id"])
        row["phrase_rank"] = prow
        row["phrase_mrr"] = 1.0 / prow if prow else 0.0
        rows.append(row)
        if args.verbose:
            mark = "OK " if row["rank"] else "MISS"
            print(f"{mark} {q['qid']} rank={row['rank']}  {q['question']}")
    elapsed = time.perf_counter() - started

    summary = summarize(rows)
    summary["retriever"] = args.retriever
    summary["top_k"] = args.top_k
    summary["elapsed_s"] = round(elapsed, 3)
    summary["ms_per_query"] = round(elapsed * 1000 / len(rows), 2)

    print(f"\n== {args.retriever} ==")
    for key in (
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@10",
        "phrase_recall@5",
        "phrase_recall@10",
        "phrase_mrr",
    ):
        print(f"{key:>16}: {summary[key]:.4f}")
    print(f"{'ms/query':>16}: {summary['ms_per_query']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.retriever}.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    misses = [r["qid"] for r in rows if r["rank"] is None]
    if misses:
        print(f"misses@{args.top_k}: {', '.join(misses)}")


if __name__ == "__main__":
    main()
