"""Answer construction: citations, low-score refusal, optional LLM generation.

Two modes:
- extractive (default, no API key): return the best chunk with a citation.
- llm (when LLM_API_KEY is set): ask an OpenAI-compatible chat model to answer
  using only the retrieved context, then verify that every cited index exists.

The refusal rule is intentionally simple and auditable: if the reranker's top
score is below a threshold, we do not answer at all.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

DOC_TITLES = {
    "hypertension_2024": "《中国高血压防治指南（2024年修订版）》",
    "t2dm_2020": "《中国2型糖尿病防治指南（2020年版）》",
}

SYSTEM_PROMPT = """你是严谨的医学指南问答助手。只能依据给定的【资料】回答。
规则：
1. 每个结论后必须标注来源编号，如 [1]。
2. 资料中没有的信息，直接回答“根据现有资料无法确定”，不要补充常识。
3. 不要给出个体化诊断或处方建议，提示以临床医生判断为准。
4. 用中文回答，简洁、分点。"""


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False
    mode: str = "extractive"
    top_score: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "mode": self.mode,
            "top_score": self.top_score,
            "reason": self.reason,
        }


def citation_for(hit: dict, index: int) -> dict:
    return {
        "index": index,
        "chunk_id": hit["chunk_id"],
        "doc_id": hit["doc_id"],
        "doc_title": DOC_TITLES.get(hit["doc_id"], hit["doc_id"]),
        "page": hit["page"],
        "text": hit["text"],
    }


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        title = DOC_TITLES.get(hit["doc_id"], hit["doc_id"])
        blocks.append(f"[{i}] {title} 第{hit['page']}页\n{hit['text']}")
    return "\n\n".join(blocks)


def _llm_answer(question: str, hits: list[dict]) -> str | None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"【资料】\n{build_context(hits)}\n\n【问题】{question}",
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def validate_citations(text: str, n_sources: int) -> list[int]:
    """Return citation indices mentioned in the answer that are out of range."""
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in cited if i < 1 or i > n_sources)


def clean_excerpt(text: str, head_limit: int = 60) -> str:
    """Drop the overlap fragment at the start of a chunk.

    Chunks carry the tail of the previous chunk, so the first "sentence" is
    often a dangling half-sentence. If a sentence end appears very early, cut
    before it.
    """
    text = text.strip()
    idx = text.find("。")
    if 0 <= idx < head_limit:
        text = text[idx + 1 :].strip()
    return text


def answer_question(
    question: str,
    hits: list[dict],
    threshold: float = 0.0,
    max_sources: int = 5,
) -> Answer:
    hits = hits[:max_sources]
    if not hits:
        return Answer(
            question=question,
            answer="根据现有资料无法确定。",
            refused=True,
            reason="no_candidates",
        )

    top_score = float(hits[0].get("score", 0.0))
    if top_score < threshold:
        return Answer(
            question=question,
            answer="根据现有资料无法确定。",
            refused=True,
            top_score=top_score,
            reason="below_threshold",
        )

    citations = [citation_for(hit, i) for i, hit in enumerate(hits, 1)]
    generated = _llm_answer(question, hits)
    if generated:
        bad = validate_citations(generated, len(hits))
        if bad:
            # Do not return an answer whose citations point nowhere.
            return Answer(
                question=question,
                answer="根据现有资料无法确定。",
                refused=True,
                top_score=top_score,
                mode="llm",
                reason=f"invalid_citations:{bad}",
            )
        return Answer(
            question=question,
            answer=generated,
            citations=citations,
            top_score=top_score,
            mode="llm",
        )

    # Extractive fallback keeps the demo usable without an API key.
    best = hits[0]
    title = DOC_TITLES.get(best["doc_id"], best["doc_id"])
    answer = (
        f"{clean_excerpt(best['text'])}\n\n"
        f"（来源：[1] {title} 第{best['page']}页。以上为指南原文摘录，"
        f"具体诊疗请遵医嘱。）"
    )
    return Answer(
        question=question,
        answer=answer,
        citations=citations,
        top_score=top_score,
        mode="extractive",
    )
