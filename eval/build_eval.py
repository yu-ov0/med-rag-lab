"""Resolve each seed question's gold_phrase to a concrete chunk id.

The seed file stores a short verbatim phrase that must appear in the source
guideline. We map it to the *shortest* chunk containing that phrase, which is
usually the chunk where the fact is stated most directly. Every question must
resolve to exactly one chunk; otherwise the build fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
SEED = ROOT / "eval" / "seed_qa.jsonl"
OUT = ROOT / "eval" / "eval_set.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve(question: dict, chunks: list[dict]) -> dict:
    phrase = question["gold_phrase"]
    candidates = [
        c for c in chunks
        if c["doc_id"] == question["doc_id"] and phrase in c["text"]
    ]
    if not candidates:
        raise ValueError(f"{question['qid']}: phrase not found: {phrase!r}")
    best = min(candidates, key=lambda c: len(c["text"]))
    return {
        **question,
        "gold_chunk_id": best["chunk_id"],
        "gold_page": best["page"],
        "n_candidates": len(candidates),
    }


def main() -> None:
    chunks = load_jsonl(CHUNKS)
    seed = load_jsonl(SEED)
    resolved: list[dict] = []
    problems: list[str] = []

    for question in seed:
        try:
            item = resolve(question, chunks)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        resolved.append(item)
        if item["n_candidates"] > 1:
            print(
                f"note {item['qid']}: {item['n_candidates']} chunks contain the phrase, "
                f"picked shortest ({item['gold_chunk_id']})"
            )

    if problems:
        print("\nUNRESOLVED:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    with OUT.open("w", encoding="utf-8") as f:
        for item in resolved:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_doc: dict[str, int] = {}
    for item in resolved:
        by_doc[item["doc_id"]] = by_doc.get(item["doc_id"], 0) + 1
    print(f"\nwrote {len(resolved)} questions -> {OUT.relative_to(ROOT)}")
    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id}: {count}")


if __name__ == "__main__":
    main()
