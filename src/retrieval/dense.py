"""Dense retrieval with a fastembed ONNX bi-encoder.

We cache the corpus embeddings on disk so repeated evaluation runs do not
re-encode the 833 chunks. The vector index is a plain normalized matrix; with
a corpus this small, a numpy dot product is faster than shipping FAISS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
# BGE zh models expect this instruction on the query side only.
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunks: list[dict] | None = None,
        use_instruction: bool = False,
        batch_size: int = 32,
        model_path: Path | None = None,
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.chunks = chunks or load_chunks()
        self.use_instruction = use_instruction
        local = model_path or LOCAL_MODEL_DIR
        kwargs = {"specific_model_path": str(local)} if local.is_dir() else {}
        self.model = TextEmbedding(model_name=model_name, batch_size=batch_size, **kwargs)
        self.matrix = self._load_or_build_matrix()

    def _cache_path(self) -> Path:
        slug = self.model_name.replace("/", "__")
        return INDEX_DIR / f"dense_{slug}.npy"

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts, batch_size=32))
        mat = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-12, None)

    def _load_or_build_matrix(self) -> np.ndarray:
        cache = self._cache_path()
        if cache.exists():
            mat = np.load(cache)
            if mat.shape[0] == len(self.chunks):
                return mat
        mat = self._embed([c["text"] for c in self.chunks])
        np.save(cache, mat)
        return mat

    def embed_query(self, query: str) -> np.ndarray:
        text = QUERY_INSTRUCTION + query if self.use_instruction else query
        vec = next(iter(self.model.embed([text])))
        vec = np.asarray(vec, dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.matrix @ self.embed_query(query)
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results


if __name__ == "__main__":
    import sys

    retriever = DenseRetriever()
    query = " ".join(sys.argv[1:]) or "高血压诊断标准"
    for rank, hit in enumerate(retriever.search(query, top_k=5), 1):
        print(f"\n[{rank}] {hit['chunk_id']} score={hit['score']:.3f}")
        print(hit["text"][:400])
