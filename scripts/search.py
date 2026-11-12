"""Unified retrieval CLI.

Usage:
    python scripts/search.py "高血压诊断标准 140 90"
    python scripts/search.py --retriever bm25 "高血压诊断标准 140 90"
    python scripts/search.py --retriever hybrid_rerank -k 5 "..."

This wrapper exists because running `src/retrieval/*.py` directly puts the
module's own directory on sys.path, not `src/`, so `import retrieval.*` fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument(
        "--retriever",
        default="hybrid_rerank",
        choices=["bm25", "dense", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    query = " ".join(args.query)
    retriever = build_retriever(args.retriever)
    for rank, hit in enumerate(retriever.search(query, top_k=args.top_k), 1):
        extra = ""
        if "first_stage_rank" in hit:
            extra = f" first_stage={hit['first_stage_rank']}"
        print(f"\n[{rank}] {hit['chunk_id']} score={hit['score']:.4f}{extra}")
        print(hit["text"][:500])


if __name__ == "__main__":
    main()
