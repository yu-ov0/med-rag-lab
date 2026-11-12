"""Parallel ranged download for large files on slow/throttled links.

Single-connection downloads from HuggingFace can be throttled to ~150 KB/s.
This script splits the file into ranges and downloads them concurrently, then
concatenates in order. Safe to re-run: completed parts are kept.

Usage:
    python scripts/fetch_parallel.py URL OUT_PATH [--workers 8] [--chunk-mb 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent import futures
from pathlib import Path

import requests

RETRIES = 5
TIMEOUT = 120


def ca_bundle() -> str | bool:
    """Prefer an explicit bundle; fall back to the system store.

    In this environment Python's default CA lookup fails, while curl and the
    system bundle work, so pass the bundle explicitly to requests.
    """
    for candidate in (
        os.getenv("REQUESTS_CA_BUNDLE"),
        os.getenv("SSL_CERT_FILE"),
        "/etc/ssl/certs/ca-certificates.crt",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return True


VERIFY = ca_bundle()


def remote_size(url: str) -> int:
    # Some HF endpoints return content-length: 0 on HEAD but expose the real
    # size in Content-Range for a one-byte ranged GET.
    resp = requests.get(
        url,
        headers={"Range": "bytes=0-0"},
        allow_redirects=True,
        timeout=60,
        verify=VERIFY,
    )
    resp.raise_for_status()
    content_range = resp.headers.get("content-range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    size = resp.headers.get("content-length")
    if size and int(size) > 1:
        return int(size)
    raise SystemExit("server did not return a usable content length")


def download_part(url: str, start: int, end: int, path: Path) -> None:
    expected = end - start + 1
    for attempt in range(1, RETRIES + 1):
        try:
            if path.exists() and path.stat().st_size == expected:
                return
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(
                url, headers=headers, stream=True, timeout=TIMEOUT, verify=VERIFY
            ) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
                if tmp.stat().st_size != expected:
                    raise OSError(
                        f"short read: {tmp.stat().st_size} != {expected}"
                    )
                tmp.replace(path)
                return
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt}/{RETRIES} {path.name}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise SystemExit(f"failed to download range {start}-{end}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(args.url)
    chunk = args.chunk_mb * 1024 * 1024
    parts_dir = out.parent / f".{out.name}.parts"
    parts_dir.mkdir(exist_ok=True)

    ranges = []
    for i, start in enumerate(range(0, size, chunk)):
        end = min(start + chunk - 1, size - 1)
        ranges.append((i, start, end))

    print(f"size={size / 1e6:.1f} MB, chunks={len(ranges)}, workers={args.workers}")
    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(download_part, args.url, s, e, parts_dir / f"{i:05d}.part")
            for i, s, e in ranges
        ]
        for job in futures.as_completed(jobs):
            job.result()

    with out.open("wb") as dst:
        for i, _, _ in ranges:
            with (parts_dir / f"{i:05d}.part").open("rb") as src:
                while block := src.read(1 << 20):
                    dst.write(block)

    elapsed = time.perf_counter() - started
    actual = out.stat().st_size
    if actual != size:
        raise SystemExit(f"size mismatch: {actual} != {size}")
    print(f"done: {out} {actual / 1e6:.1f} MB in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
