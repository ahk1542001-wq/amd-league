#!/usr/bin/env python3
"""RAG core for AMD MC3: defensive corpus parsing, chunking, retrieval, prompts.

No model weights are loaded here. Heavy model calls go through the daemon
(daemon.py) over a unix socket, so this module also runs on machines without
a GPU (used for local parser testing).
"""

from __future__ import annotations

import csv
import json
import os
import re
import socket
import struct
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Socket client (talks to daemon.py)
# ---------------------------------------------------------------------------

SOCK_PATH = Path(os.environ.get("MC3_SOCK", "/app/index/rag.sock"))


def _rpc(op: str, timeout: float = 600.0, **kwargs):
    """Send one JSON request, read one JSON response (length-prefixed)."""
    payload = json.dumps({"op": op, **kwargs}).encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(SOCK_PATH))
        s.sendall(struct.pack(">I", len(payload)) + payload)
        hdr = _recvall(s, 4)
        (n,) = struct.unpack(">I", hdr)
        data = _recvall(s, n)
    return json.loads(data.decode("utf-8"))


def _recvall(s, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# Defensive corpus walk
# ---------------------------------------------------------------------------

# Extensions we know how to parse. Anything else is skipped silently
# (unknown-type trap, e.g. telemetry_capture.dat).
KNOWN_EXTS = {
    ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".log", ".py",
    ".png", ".jpg", ".jpeg",
}

# Filenames carrying these markers get down-ranked at retrieval time
# (superseded-document trap, e.g. tq40_datasheet_r1_WITHDRAWN.pdf).
STALE_MARKERS = ("withdrawn", "superseded", "obsolete", "draft", "deprecated")

MAX_TEXT_BYTES = 2_000_000  # per-file cap so one giant log can't OOM us


def iter_corpus_files(corpus: Path):
    """Yield (relative_posix_path, absolute_path) for parseable files.

    Defensive: per-file try/except, sorted order, skips unreadable files,
    unknown extensions, and anything that is not a regular file.
    """
    corpus = Path(corpus)
    found = []
    for root, dirs, files in os.walk(corpus):
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            if p.suffix.lower() not in KNOWN_EXTS:
                continue
            try:
                # Touch the file: raises PermissionError on chmod-000 files
                # when the harness drops DAC_OVERRIDE.
                with open(p, "rb") as f:
                    f.read(1)
            except OSError:
                continue
            try:
                rel = p.relative_to(corpus).as_posix()
            except ValueError:
                continue
            found.append((rel, p))
    return found


# ---------------------------------------------------------------------------
# Per-format text extractors — every one returns "" on any failure.
# ---------------------------------------------------------------------------

def _read_bytes_capped(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read(MAX_TEXT_BYTES + 1)[:MAX_TEXT_BYTES]


def extract_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        if reader.is_encrypted:
            return ""  # encrypted trap -> skip, don't fail
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        return ""


def extract_docx(p: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(p))
        parts = [para.text for para in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(t for t in parts if t.strip())
    except Exception:
        return ""


def extract_xlsx(p: Path) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(p), read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"[sheet: {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception:
        return ""


def extract_csv(p: Path) -> str:
    try:
        raw = _read_bytes_capped(p).decode("utf-8", errors="replace")
        lines = []
        for row in csv.reader(raw.splitlines()):
            cells = [c.strip() for c in row if c.strip()]
            if cells:
                lines.append(" | ".join(cells))
        return "\n".join(lines)
    except Exception:
        return ""


def extract_plaintext(p: Path) -> str:
    try:
        return _read_bytes_capped(p).decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_text_file(rel: str, p: Path) -> str:
    """Dispatch by extension. Images return "" here (handled via VLM)."""
    ext = Path(rel).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(p)
    if ext == ".docx":
        return extract_docx(p)
    if ext == ".xlsx":
        return extract_xlsx(p)
    if ext == ".csv":
        return extract_csv(p)
    if ext in (".txt", ".log", ".py"):
        return extract_plaintext(p)
    return ""


def is_image(rel: str) -> bool:
    return Path(rel).suffix.lower() in (".png", ".jpg", ".jpeg")


def render_pdf_pages(p: Path, dpi: int = 150) -> list:
    """Render PDF pages to PIL images (for scanned-PDF OCR fallback).

    Returns [] on any failure or if pymupdf is unavailable.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(p))
        imgs = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            import io
            from PIL import Image
            imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return imgs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


def chunk_text(text: str, rel: str) -> list[dict]:
    """Split into overlapping chunks, each tagged with its source file."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        # try to break on a newline near the end
        if end < n:
            cut = text.rfind("\n", start + CHUNK_SIZE - 150, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append({"text": piece, "file": rel, "chunk": idx})
            idx += 1
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Retrieval helpers (pure python/numpy — no model needed)
# ---------------------------------------------------------------------------

def stale_penalty(rel: str) -> float:
    name = rel.lower()
    return 0.5 if any(m in name for m in STALE_MARKERS) else 1.0


def cosine_topk(query_vec, matrix, k: int):
    """matrix: (n, d) float32. Returns indices of top-k by cosine sim."""
    import numpy as np
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def normalize_answer(s: str) -> str:
    """Lowercase, strip units/degree signs/punctuation for comparison."""
    s = s.lower()
    s = s.replace("°c", "").replace("°", "").replace("degrees", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_in_text(answer: str, text: str) -> bool:
    """Does the (normalized) answer appear in the text?"""
    a = normalize_answer(answer)
    t = normalize_answer(text)
    if not a:
        return False
    # For short numeric/part-number answers require token-level match
    if re.fullmatch(r"[a-z0-9\-/#. ]+", a) and len(a) <= 24:
        return a in t
    return a in t


# ---------------------------------------------------------------------------
# Prompt + citation verification
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a precise information extraction system. Answer the question using ONLY the context below.

Rules:
- Output the answer VALUE ONLY on the first line (examples: "94", "Q3 FY27", "B14", "ORR-FAN-2214-B"). No sentences, no explanation, no units beyond what is part of the value.
- On the second line write CITATIONS: followed by the exact file paths you used, comma-separated (example: CITATIONS: specs/tq40_datasheet_r2.pdf).
- Cite a file ONLY if the answer (or a necessary intermediate fact in a multi-step chain) appears in that file's text. Never cite a file you did not use. At most 3 files.
- If several files give different values for the same fact, prefer the file that is NOT marked withdrawn, superseded, obsolete or draft.
- If the context does not contain the answer, output exactly this and nothing else:
NO_ANSWER

Context:
{context}

Question: {query}
"""


def build_context(ranked: list[tuple[dict, float]], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for ch, _score in ranked:
        block = f"[FILE: {ch['file']}]\n{ch['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def parse_model_output(raw: str) -> tuple[str, list[str]]:
    """Returns (answer, citations). ("", []) means refusal/malformed."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "", []
    if lines[0].upper() == "NO_ANSWER":
        return "", []
    answer = lines[0]
    citations: list[str] = []
    for ln in lines[1:]:
        m = re.match(r"(?i)^citations?\s*:\s*(.+)$", ln)
        if m:
            citations = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
            break
    # sanity: answer should be short (a value, not prose)
    if len(answer) > 120 or len(answer.split()) > 12:
        return "", []
    return answer, citations[:3]


ENTITY_RE = re.compile(
    r"\b([A-Z]{2,}[A-Z0-9]*-\d[\w\-]*|E\d{4,5}|REV-[A-Z0-9]+)\b")


def _entities(text: str) -> set[str]:
    return set(ENTITY_RE.findall(text.upper()))


def verify_citations(answer: str, citations: list[str],
                     chunks_by_file: dict[str, list[str]]) -> list[str]:
    """Keep only citations whose file plausibly supports the answer.

    A file is kept if EITHER:
      (a) the normalized answer appears in the file's retrieved text, OR
      (b) [multi-hop bridge] the file shares a distinctive entity
          (ticket/part/error/revision id) with a file that does contain
          the answer. E.g. the production log links the incident to
          ticket ORR-1847 while bug_database.csv links ORR-1847 to the
          firmware version -- both are necessary, only the csv holds
          the final value.
    Files with no supporting text, or with neither (a) nor (b), are
    dropped. Returns the filtered list; empty -> caller must refuse.
    """
    if not answer or not citations:
        return []
    norm_files = {c: normalize_answer("\n".join(chunks_by_file.get(c, [])))
                  for c in citations}
    ans_norm = normalize_answer(answer)
    tokens = [t for t in ans_norm.split() if len(t) >= 2]

    def holds_value(blob: str) -> bool:
        if not blob:
            return False
        if ans_norm and ans_norm in blob:
            return True
        return bool(tokens) and any(t in blob for t in tokens)

    answer_files = {c for c in citations if holds_value(norm_files[c])}
    if not answer_files:
        return []  # answer grounded nowhere -> refuse
    answer_entities: set[str] = set()
    for c in answer_files:
        answer_entities |= _entities(chunks_raw(c, chunks_by_file))

    kept = []
    for c in citations:
        if c in answer_files:
            kept.append(c)
            continue
        raw = chunks_raw(c, chunks_by_file)
        if raw and (answer_entities & _entities(raw)):
            kept.append(c)
    # de-dup, preserve order
    seen, out = set(), []
    for c in kept:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def chunks_raw(citation: str, chunks_by_file: dict[str, list[str]]) -> str:
    return "\n".join(chunks_by_file.get(citation, []))
