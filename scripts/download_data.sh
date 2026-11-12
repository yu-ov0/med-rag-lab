#!/usr/bin/env bash
# Download the two public guideline PDFs and verify their checksums.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw

fetch() {
  local url="$1" out="$2" sha="$3"
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --quiet -; then
    echo "ok (cached): $out"
    return
  fi
  echo "-> $out"
  curl -sSL --fail --retry 3 --max-time 300 -o "$out" "$url"
  echo "$sha  $out" | sha256sum -c -
}

fetch \
  "https://www.scdc.sh.cn/shjk/shjk/upload/202410/1011_162256_499.pdf" \
  "data/raw/hypertension_2024.pdf" \
  "32b2e11aec9b51bbfafbd43b4d0b60775fd845dac4c03499cb9e38821e83e0f9"

fetch \
  "https://www.scdc.sh.cn/shjk/shjk/upload/202303/0306_165758_510.pdf" \
  "data/raw/t2dm_2020.pdf" \
  "df00aed6e7f6e7ed20989f1f56cdb0dde4ab2187582cbec4f9a2f0f72ffadc57"

echo "done"
