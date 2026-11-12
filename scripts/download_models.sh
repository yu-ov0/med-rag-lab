#!/usr/bin/env bash
# Download the ONNX models used by the retrieval stack.
#
# fastembed's built-in downloader writes empty files in some sandboxed
# environments, so we fetch the exact files ourselves and point fastembed at
# the local directory via `specific_model_path`.
set -euo pipefail

cd "$(dirname "$0")/.."

fetch() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  echo "-> $out"
  curl -sSL --fail --retry 3 --max-time 900 -o "$out" "$url"
}

EMBED_DIR=models/bge-small-zh-v1.5
for f in config.json tokenizer_config.json tokenizer.json special_tokens_map.json model_optimized.onnx; do
  fetch "https://huggingface.co/Qdrant/bge-small-zh-v1.5/resolve/main/$f" "$EMBED_DIR/$f"
done

RERANK_DIR=models/bge-reranker-v2-m3-int8
for f in config.json tokenizer_config.json tokenizer.json special_tokens_map.json; do
  fetch "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/$f" "$RERANK_DIR/$f"
done
# The ONNX weight is large and often throttled; ranged parallel download is
# much faster than a single curl connection.
.venv/bin/python scripts/fetch_parallel.py \
  "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/onnx/model_int8.onnx" \
  "$RERANK_DIR/onnx/model_int8.onnx" --workers 8 --chunk-mb 8

du -sh "$EMBED_DIR" "$RERANK_DIR"
