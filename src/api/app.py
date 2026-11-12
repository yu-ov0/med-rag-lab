"""FastAPI service: hybrid retrieval + rerank + citation-constrained answer."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from generation.answer import answer_question
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import Reranker

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base = HybridRetriever()
    STATE["retriever"] = Reranker(base)
    STATE["threshold"] = float(os.getenv("RERANK_THRESHOLD", "2.0"))
    yield
    STATE.clear()


app = FastAPI(title="med-rag-lab", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    threshold: float | None = None


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    refused: bool
    mode: str
    top_score: float | None
    reason: str | None
    latency_ms: float
    hits: list[Hit]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retriever": "hybrid+rerank"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    started = time.perf_counter()
    retriever: Reranker = STATE["retriever"]
    threshold = STATE["threshold"] if req.threshold is None else req.threshold
    hits = retriever.search(req.question, top_k=req.top_k)
    result = answer_question(
        req.question, hits, threshold=threshold, max_sources=req.top_k
    )
    latency = (time.perf_counter() - started) * 1000
    return AskResponse(
        **result.to_dict(),
        latency_ms=round(latency, 2),
        hits=[
            Hit(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                page=h["page"],
                score=float(h.get("score", 0.0)),
            )
            for h in hits
        ],
    )
