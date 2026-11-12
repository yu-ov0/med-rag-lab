"""Extract and chunk the medical guideline PDFs.

The goal is a reproducible corpus for retrieval evaluation, not a perfect
PDF-to-Markdown converter. We keep page numbers so every chunk can be cited.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DOCS = {
    "hypertension_2024": RAW / "hypertension_2024.pdf",
    "t2dm_2020": RAW / "t2dm_2020.pdf",
}

# Lines that are page headers/footers in the source PDFs.
HEADER_PATTERNS = [
    r"^中华.*杂志.*第.*卷.*第.*期$",
    r"^Chin J .*Vol\..*No\..*$",
    r"^\d+\s*$",
    r"^·?\s*指南\s*·?$",
]


def normalize_line(line: str) -> str:
    line = line.replace("\u3000", " ").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def is_header_footer(line: str) -> bool:
    return any(re.search(p, line) for p in HEADER_PATTERNS)


def detect_repeated_lines(pages: list[dict]) -> set[str]:
    counter: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for raw in page["text"].splitlines():
            line = normalize_line(raw)
            if 3 <= len(line) <= 80 and line not in seen:
                counter[line] += 1
                seen.add(line)
    threshold = max(3, int(len(pages) * 0.2))
    return {line for line, count in counter.items() if count >= threshold}


def clean_page(text: str, repeated: set[str]) -> str:
    kept: list[str] = []
    for raw in text.splitlines():
        line = normalize_line(raw)
        if not line:
            continue
        if is_header_footer(line) or line in repeated:
            continue
        kept.append(line)
    # Join wrapped lines. A new line starts a new paragraph only when the
    # previous line already ended with sentence-final punctuation.
    paragraphs: list[str] = []
    buf = ""
    for line in kept:
        if not buf:
            buf = line
            continue
        if re.search(r"[。！？；：）】]$", buf) or re.match(
            r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|\d+(\.\d+)+|要点\s*\d)",
            line,
        ):
            paragraphs.append(buf)
            buf = line
        else:
            # Chinese PDFs often break a sentence across lines without a space.
            if re.search(r"[A-Za-z0-9]$", buf) and re.match(r"^[A-Za-z0-9]", line):
                buf += " " + line
            else:
                buf += line
    if buf:
        paragraphs.append(buf)
    return "\n".join(paragraphs)


def split_sentences(text: str) -> list[str]:
    # Keep the delimiter with the sentence.
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 450, overlap: int = 80) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
            continue
        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            chunks.append(buf)
            # Carry the tail of the previous chunk as overlap.
            tail = buf[-overlap:] if overlap else ""
            buf = tail + sent
    if buf:
        chunks.append(buf)
    # Drop chunks that are only page numbers or too short to be useful.
    return [c.strip() for c in chunks if len(c.strip()) >= 80]


def main() -> None:
    all_chunks: list[dict] = []
    for doc_id, path in DOCS.items():
        doc = pymupdf.open(path)
        pages = [
            {"doc_id": doc_id, "page": i + 1, "text": page.get_text("text")}
            for i, page in enumerate(doc)
        ]
        doc.close()
        repeated = detect_repeated_lines(pages)
        for page in pages:
            page["text"] = clean_page(page["text"], repeated)
        with (PROCESSED / f"{doc_id}_pages.jsonl").open("w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        for page in pages:
            for j, chunk in enumerate(chunk_text(page["text"])):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_id}:p{page['page']}:c{j}",
                        "doc_id": doc_id,
                        "page": page["page"],
                        "text": chunk,
                    }
                )
    with (PROCESSED / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"docs={len(DOCS)} chunks={len(all_chunks)}")
    for doc_id in DOCS:
        n = sum(1 for c in all_chunks if c["doc_id"] == doc_id)
        chars = sum(len(c["text"]) for c in all_chunks if c["doc_id"] == doc_id)
        print(f"  {doc_id}: chunks={n} chars={chars}")


if __name__ == "__main__":
    main()
