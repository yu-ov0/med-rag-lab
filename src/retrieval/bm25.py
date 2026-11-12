"""BM25 retrieval over the medical guideline chunks."""

from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    # jieba handles Chinese word segmentation; lower-case Latin terms.
    return [t.lower() for t in jieba.lcut(text) if t.strip()]


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class BM25Retriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or load_chunks()
        self.corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:top_k]
        results = []
        for idx in order:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results


if __name__ == "__main__":
    import sys

    retriever = BM25Retriever()
    query = " ".join(sys.argv[1:]) or "高血压诊断标准"
    for rank, hit in enumerate(retriever.search(query, top_k=5), 1):
        print(f"\n[{rank}] {hit['chunk_id']} score={hit['score']:.3f}")
        print(hit["text"][:500])
