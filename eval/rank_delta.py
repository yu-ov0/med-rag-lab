"""Per-question rank movement between two retrievers.

Shows which questions reranking fixed and which it broke, instead of only
looking at aggregate metrics. A rank of None means "not in top-k".
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"


def load(name: str) -> dict[str, dict]:
    data = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    return {row["qid"]: row for row in data["rows"]}


def fmt(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="bm25")
    parser.add_argument("--after", default="hybrid_rerank")
    args = parser.parse_args()

    before = load(args.before)
    after = load(args.after)
    print(f"before={args.before}  after={args.after}")

    gains: list[tuple[int, str, str]] = []
    losses: list[tuple[int, str, str]] = []
    for qid in before:
        b, a = before[qid]["rank"], after[qid]["rank"]
        bv = b if b is not None else 999
        av = a if a is not None else 999
        if av < bv:
            gains.append((bv - av, qid, f"{fmt(b)} -> {fmt(a)}"))
        elif av > bv:
            losses.append((av - bv, qid, f"{fmt(b)} -> {fmt(a)}"))

    gains.sort(reverse=True)
    losses.sort(reverse=True)

    print(f"改善 {len(gains)} 题 / 变差 {len(losses)} 题 / 共 {len(before)} 题\n")
    print("改善最大：")
    for delta, qid, move in gains[:8]:
        print(f"  {qid}  {move:>12}  (+{delta})")
    print("\n变差：")
    for delta, qid, move in losses[:8]:
        print(f"  {qid}  {move:>12}  (-{delta})")


if __name__ == "__main__":
    main()
