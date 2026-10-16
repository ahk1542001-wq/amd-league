#!/usr/bin/env python3
"""Download model weights at docker BUILD time (needs network then).

Models land in /models and are COPY-safe into the image:
  - BAAI/bge-small-en-v1.5            (~130 MB, embeddings)
  - Qwen/Qwen2.5-VL-7B-Instruct       (~15 GB bf16, VLM for QA + image OCR)

Total ~15.5 GB + 29.2 GB base = well under the 60 GiB image limit.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

out = Path("/models")
out.mkdir(parents=True, exist_ok=True)
for m in MODELS:
    print(f"[download] {m}", flush=True)
    snapshot_download(
        repo_id=m,
        local_dir=str(out / m.split("/")[-1]),
        local_dir_use_symlinks=False,
        # skip files we never load at eval
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )
print("[download] done", flush=True)
