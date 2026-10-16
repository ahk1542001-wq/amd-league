#!/usr/bin/env python3
"""MC3 submission: RAG over a mixed-filetype corpus on AMD ROCm.

Phase 1 (--index): walk corpus defensively, extract text per format,
transcribe images via the VLM daemon, embed everything, persist index.
A GPU daemon is left running on a unix socket so the ten per-question
processes never reload weights.

Phase 2 (--query): thin client -> daemon -> {answer, citations, confidence}.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from rag import (  # noqa: E402
    SOCK_PATH, _rpc, chunk_text, extract_text_file, is_image, iter_corpus_files,
)

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))


def _wait_for_daemon(timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = _rpc("ping", timeout=5.0)
            if r.get("ok"):
                return
        except Exception as e:
            last = e
        time.sleep(2.0)
    raise RuntimeError(f"daemon never came up: {last}")


def index(corpus: Path) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Start the GPU daemon first (loads embedder + VLM once).
    subprocess.Popen(
        [sys.executable, str(APP_DIR / "daemon.py")],
        start_new_session=True,
        stdout=open(INDEX_DIR / "daemon.log", "ab"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _wait_for_daemon()

    # 2. Defensive walk + per-format extraction (no GPU needed for text).
    files = iter_corpus_files(corpus)
    chunks: list[dict] = []
    images: list[tuple[str, Path]] = []
    for rel, p in files:
        if is_image(rel):
            images.append((rel, p))
            continue
        try:
            text = extract_text_file(rel, p)
        except Exception:
            continue
        # Scanned-PDF fallback: page rendered -> VLM transcription.
        if rel.lower().endswith(".pdf") and len(text.strip()) < 50:
            text = _ocr_pdf_pages(p)
        if text.strip():
            chunks.extend(chunk_text(text, rel))

    # 3. Transcribe images through the daemon, add as chunks.
    for rel, p in images:
        text = _transcribe_image_file(p)
        if text:
            chunks.extend(chunk_text(text, rel))

    # 4. Persist chunks; daemon embeds them.
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    r = _rpc("index_embeddings", timeout=1800.0)
    if not r.get("ok"):
        raise RuntimeError(f"embedding failed: {r}")

    _rpc("ping", timeout=10.0)
    print(f"[index] done: {len(chunks)} chunks from "
          f"{len(files)} files ({len(images)} images)", flush=True)


def _transcribe_image_file(p: Path) -> str:
    try:
        r = _rpc("transcribe", timeout=300.0, image=str(p))
        return (r.get("text") or "").strip()
    except Exception:
        return ""


def _ocr_pdf_pages(p: Path) -> str:
    """Render PDF pages and OCR each via the VLM. Returns joined text."""
    from rag import render_pdf_pages
    import tempfile
    parts = []
    try:
        pages = render_pdf_pages(p)
    except Exception:
        return ""
    for i, img in enumerate(pages[:20]):  # cap: first 20 pages
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
                tmp = f.name
            parts.append(_transcribe_image_file(Path(tmp)))
        except Exception:
            continue
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return "\n".join(t for t in parts if t)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    try:
        r = _rpc("answer", timeout=120.0, query=query)
    except Exception:
        return "", [], 0.0
    if "error" in r:
        return "", [], 0.0
    return (r.get("answer") or "", list(r.get("citations") or []),
            float(r.get("confidence") or 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    out.write_text(json.dumps(
        {"answer": text, "citations": list(citations),
         "confidence": confidence}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
