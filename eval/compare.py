"""Compare all retriever result files and print a markdown table + badcases."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
ORDER = ["bm25", "dense", "hybrid", "hybrid_rerank"]
LABELS = {
    "bm25": "BM25",
    "dense": "向量（bge-small-zh）",
    "hybrid": "混合 RRF",
    "hybrid_rerank": "混合 + 重排",
}
COLS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]


def main() -> None:
    loaded = {}
    for name in ORDER:
        path = RESULTS / f"{name}.json"
        if path.exists():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))

    print("| 方案 | " + " | ".join(COLS) + " |")
    print("| --- | " + " | ".join("---" for _ in COLS) + " |")
    for name, data in loaded.items():
        s = data["summary"]
        cells = " | ".join(f"{s[c]:.4f}" for c in COLS)
        print(f"| {LABELS[name]} | {cells} |")

    if "hybrid_rerank" in loaded:
        rows = loaded["hybrid_rerank"]["rows"]
        misses = [r for r in rows if r["rank"] is None]
        print(f"\n重排后仍未命中：{len(misses)}")
        for r in misses:
            print(f"- {r['qid']}: {r['question']} (gold={r['gold']})")

        gains = [
            r for r in rows
            if r["rank"] is not None and r["rank"] <= 3
        ]
        print(f"\n重排后进入 Top3：{len(gains)}/{len(rows)}")


if __name__ == "__main__":
    main()
