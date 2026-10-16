#!/usr/bin/env python3
"""GPU model server for MC3. Started once at the end of index(), kept alive
behind a unix socket so the ten per-question processes never reload weights.

Protocol: length-prefixed JSON request -> length-prefixed JSON response.
Ops: ping | transcribe {image} | index_embeddings {} | answer {query}

Models (all local, no network at eval):
  - embedder: BAAI/bge-small-en-v1.5  (~130 MB)
  - vlm:      Qwen/Qwen2.5-VL-7B-Instruct (~15 GB bf16) — text QA + image OCR
"""

from __future__ import annotations

import json
import os
import struct
import socket
import sys
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
SOCK_PATH = Path(os.environ.get("MC3_SOCK", "/app/index/rag.sock"))
MODELS_DIR = Path(os.environ.get("MC3_MODELS_DIR", "/models"))

sys.path.insert(0, str(APP_DIR))
from rag import (  # noqa: E402
    ANSWER_PROMPT, build_context, chunk_text, cosine_topk,
    is_image, parse_model_output, stale_penalty, verify_citations,
)

import numpy as np
import torch

DEVICE = "cuda:0"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print("[daemon] loading embedder ...", flush=True)
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer(
    str(MODELS_DIR / "bge-small-en-v1.5"), device=DEVICE, trust_remote_code=False
)

print("[daemon] loading VLM ...", flush=True)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_vlm_id = str(MODELS_DIR / "Qwen2.5-VL-7B-Instruct")
_processor = AutoProcessor.from_pretrained(_vlm_id, trust_remote_code=True)
_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    _vlm_id, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
).eval()

print("[daemon] models ready", flush=True)

# ---------------------------------------------------------------------------
# Index state (built once per container lifetime)
# ---------------------------------------------------------------------------

_chunks: list[dict] = []          # {"text","file","chunk"}
_emb: np.ndarray | None = None    # (n, d) float32


def _recvall(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return buf


def transcribe_image(image_path: str) -> str:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text":
             "Transcribe ALL text visible in this image EXACTLY, character for "
             "character, including labels, numbers, part numbers, revisions, "
             "table contents and small print. Preserve the layout with line "
             "breaks. After the transcription, add one line starting with "
             "'LAYOUT:' describing briefly what the image shows."},
        ],
    }]
    text = _processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = _processor(text=[text], images=[img], padding=True,
                        return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = _vlm.generate(**inputs, max_new_tokens=1024, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return _processor.decode(gen, skip_special_tokens=True).strip()


def op_index_embeddings(_req) -> dict:
    """Embed chunks.json -> embeddings.npy. Called once from index()."""
    global _chunks, _emb
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vecs = []
    bs = 128
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            v = _embedder.encode(texts[i:i + bs], convert_to_numpy=True,
                                 normalize_embeddings=False,
                                 show_progress_bar=False)
            vecs.append(v)
    _emb = np.vstack(vecs).astype(np.float32)
    np.save(INDEX_DIR / "embeddings.npy", _emb)
    _chunks = chunks
    (INDEX_DIR / "meta.json").write_text(json.dumps(
        {"n_chunks": len(chunks), "dim": int(_emb.shape[1])}))
    return {"ok": True, "n_chunks": len(chunks)}


def _ensure_loaded():
    global _chunks, _emb
    if _chunks and _emb is not None:
        return
    _chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    _emb = np.load(INDEX_DIR / "embeddings.npy")


@torch.inference_mode()
def op_answer(req) -> dict:
    _ensure_loaded()
    query = req["query"]
    qv = _embedder.encode([query], convert_to_numpy=True)[0]
    ranked_idx = cosine_topk(qv, _emb, 12)
    ranked = []
    for i, s in ranked_idx:
        ch = _chunks[i]
        ranked.append((ch, s * stale_penalty(ch["file"])))
    ranked.sort(key=lambda t: -t[1])
    top = ranked[:10]

    # Refusal gate: nothing even vaguely relevant retrieved.
    if not top or top[0][1] < 0.18:
        return {"answer": "", "citations": [], "confidence": 0.0}

    context = build_context(top)
    prompt = ANSWER_PROMPT.format(context=context, query=query)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = _processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = _processor(text=[text], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = _vlm.generate(**inputs, max_new_tokens=256, do_sample=False,
                            temperature=None, top_p=None)
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = _processor.decode(gen, skip_special_tokens=True)
    answer, citations = parse_model_output(raw)
    if not answer:
        return {"answer": "", "citations": [], "confidence": 0.0}

    by_file: dict[str, list[str]] = {}
    for ch, _s in top:
        by_file.setdefault(ch["file"], []).append(ch["text"])
    good = verify_citations(answer, citations, by_file)
    if not good:
        return {"answer": "", "citations": [], "confidence": 0.0}
    conf = float(min(0.99, max(0.05, top[0][1])))
    return {"answer": answer, "citations": good, "confidence": conf}


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True}
    if op == "transcribe":
        return {"text": transcribe_image(req["image"])}
    if op == "index_embeddings":
        return op_index_embeddings(req)
    if op == "answer":
        return op_answer(req)
    return {"error": f"unknown op {op}"}


def serve():
    if SOCK_PATH.exists():
        SOCK_PATH.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK_PATH))
    srv.listen(8)
    print(f"[daemon] listening on {SOCK_PATH}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            hdr = _recvall(conn, 4)
            (n,) = struct.unpack(">I", hdr)
            req = json.loads(_recvall(conn, n).decode("utf-8"))
            resp = handle(req)
        except Exception as e:  # never kill the server on a bad request
            resp = {"error": f"{type(e).__name__}: {e}"}
        try:
            payload = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            conn.sendall(struct.pack(">I", len(payload)) + payload)
        finally:
            conn.close()


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=False).start()
    threading.Event().wait()  # keep the main thread (and CUDA ctx) alive
