# AMD AI League

My work for the AMD AI League (ACT III) and the AMD Lab Program — AI systems
built and verified on AMD Instinct GPUs with ROCm.

## Match 3 — RAG Question Answering (`match3/`)

A retrieval-augmented QA container that answers questions over a messy corpus
(PDFs, DOCX, images, spreadsheets) with exact answer text and exact
source-file citations.

**How it works**

- A persistent GPU daemon keeps both models resident, so per-question
  processes never reload weights.
- On `--index`, the corpus is chunked and embedded with
  `BAAI/bge-small-en-v1.5`; images are OCR'd by the vision model.
- On each query, the retriever pulls the top chunks, a stale-revision penalty
  demotes superseded documents (e.g. withdrawn datasheet revisions), and
  `Qwen/Qwen2.5-VL-7B-Instruct` produces an exact answer plus a verified
  citation set — every cited file is checked against the corpus before output.

**Contract**

```bash
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is ...?"
# -> /app/output/query_01_output.json  {answer, citations, confidence}
```

**Verified results** (AMD Instinct MI300X, ROCm 10.1)

- Sample corpus: 10/10 correct — **TOTAL 200/200**
- Container starts with no network in 0.1s; index in 24.9s; query in 0.4s
  (30s budget); VRAM inside the 1–48 GiB band
- Trap question (withdrawn r1 datasheet saying 105 C vs current r2) answered
  correctly: `94 C`, cited to `specs/tq40_datasheet_r2.pdf`

**Image**: `victorjob154/mc3-match3:latest` (Docker Hub)

**Stack**: ROCm 10.1, PyTorch, Python, Docker, sentence-transformers.
Models are baked in at build time — the container runs fully offline.

## Match 4 — Intelligent Web Information Retrieval (`match4/`)

Coming next. Evaluating a Gemma-based approach on AMD GPUs
(Google DeepMind joined ACT III with a bonus prize pool for Gemma builds).

---

Built by Victor (Aung Hein Kyaw) — AI Automation & Agent Workflow Specialist,
Bangkok.
