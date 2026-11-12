"""Ranking metrics for the retrieval evaluation."""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def hit_rank(ranked_ids: list[str], gold_id: str) -> int | None:
    for i, cid in enumerate(ranked_ids, 1):
        if cid == gold_id:
            return i
    return None


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    rank = hit_rank(ranked_ids, gold_id)
    return 1.0 / rank if rank else 0.0


def phrase_rank(hits: list[dict], phrase: str) -> int | None:
    """First rank whose text contains the answer phrase."""
    for i, hit in enumerate(hits, 1):
        if phrase in hit["text"]:
            return i
    return None


def ndcg_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    rank = hit_rank(ranked_ids, gold_id)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def summarize(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("no evaluation rows")
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
        if f"phrase_recall@{k}" in rows[0]:
            out[f"phrase_recall@{k}"] = (
                sum(r[f"phrase_recall@{k}"] for r in rows) / n
            )
    out["mrr"] = sum(r["mrr"] for r in rows) / n
    if "phrase_mrr" in rows[0]:
        out["phrase_mrr"] = sum(r["phrase_mrr"] for r in rows) / n
    for k in ks:
        out[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in rows) / n
    out["n"] = n
    return out
