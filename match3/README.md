# MC3 (RAG) solution — FYF AI Lab

## What it is
Docker + ROCm submission for AMD Lab Program Mini-Challenge 3.
Container is handed a mixed corpus, asked 10 questions, and must output
`{"answer", "citations", "confidence"}` per question. 20 pts/question,
200 max, no partial credit — answer AND exact citation set must be right.

## Files
- `app/app.py` — contract plumbing; `index()` walks/extracts, starts the
  GPU daemon, transcribes images, persists chunks; `answer()` is a thin
  socket client.
- `app/rag.py` — defensive corpus walk (skips chmod-000, encrypted,
  unknown-type, empty dirs), per-format extractors (pdf/docx/xlsx/csv/
  txt/log/py), scanned-PDF OCR fallback, chunking, cosine retrieval,
  answer prompt, citation verifier (value match + entity-bridge for
  multi-hop), socket RPC framing.
- `app/daemon.py` — GPU model server on unix socket `/app/index/rag.sock`.
  Loads once: bge-small-en-v1.5 (embed) + Qwen2.5-VL-7B-Instruct (QA + OCR).
- `download_models.py` — build-time weight fetch into /models (~15.5 GB).
- `Dockerfile` / `app/requirements.txt` — ROCm base (29.2 GB) + deps.

## Key design decisions
- One VLM (Qwen2.5-VL-7B) does text QA and image transcription — fits the
  ~30 GB weight budget (base 29.2 + 15.5 ≈ 45 GB < 60 GB limit).
- Daemon pattern: model loads once at --index; 10 question processes reuse
  it via socket (a naive 10x reload blows the 30s/question budget).
- Citations verified, not trusted: value must appear in the cited file, or
  (multi-hop) the file must share a bridge entity (ticket/part/error id)
  with a value-bearing file. WITHDRAWN files down-ranked 0.5x at retrieval
  + prompt prefers non-withdrawn revisions.
- Refusal: `("", [], 0.0)` when nothing retrieved or verification fails
  (the encrypted-file question scores full marks on correct refusal).

## Local testing done (2026-10-07, CPU)
- Walk: 12 files, parsers OK on pdf/docx/xlsx/csv/log/py/txt.
- Trap handling: encrypted PDF -> "" (no crash); .dat excluded from walk.
- Citation verifier: Q1 r2 kept / r1-trap dropped / Q9 both files kept
  (either order) / hallucinated file dropped / refusal on empty.
- Socket framing: large-payload round-trip OK.

## Still needed (GPU machine, after $100 credits activate)
1. `docker build` on AMD GPU box (pulls 29 GB base + downloads weights).
2. `python3 selfcheck.py <image> <sample corpus>` from the starter kit.
3. Full 10-question dry run on the sample corpus; tune refusal threshold
   (0.18) and top-k if needed.
4. Record the demo video (required for submission).
5. Fill lablab.ai submission draft Step 1 (title/desc/categories/tech).
