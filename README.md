# AI Content Detector — Bulk URL Analyzer

A self-hostable Streamlit web app that takes up to **1,000 URLs at a time**,
scrapes each page's main article content, runs it through the open-source
[`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01)
model (current RAID-benchmark leader, fine-tuned from DeBERTa-v3-large), and
labels every URL with one of:

> **Low · Moderate · High · Very High**

…corresponding to the likelihood that the content is AI-generated.

Built for academic/research use. No paid APIs, no vendor lock-in,
fully reproducible.

---

## Quickstart

```bash
# 1. clone / cd into this directory
cd ai-content-detector

# 2. create a fresh venv (Python 3.10–3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. install pinned dependencies
pip install -r requirements.txt

# 4. launch the app
streamlit run app.py
```

The first run will download the detector weights from Hugging Face (~1.5 GB).
Subsequent runs use the local Hugging Face cache.

GPU is optional — the app auto-detects CUDA, Apple Silicon (MPS), or CPU.

---

## How to run this tool again

Once the venv is built and the model weights are cached, restarting the app is
three lines:

```bash
cd /Users/akilanvaruman/projects/ai-content-detector
source .venv/bin/activate
streamlit run app.py
```

Or use the bundled launcher (which does all three in one step):

```bash
/Users/akilanvaruman/projects/ai-content-detector/run.sh
```

The app will be available at **http://localhost:8501**. To stop it, press
`Ctrl+C` in the terminal that launched it (or `lsof -tiTCP:8501 -sTCP:LISTEN | xargs kill`).

---

## Architecture

```
Streamlit UI  ──▶  Async URL Fetcher (httpx, 20 workers)
                       │
                       ▼
                  Trafilatura (main-article extraction)
                       │
                       ▼
                  Tokeniser-aware chunker (450-token windows, 50-token overlap)
                       │
                       ▼
                  Desklib AI detector (DeBERTa-v3-large)
                       │
                       ▼
                  Average chunk probabilities → Low / Moderate / High / Very High
```

Module map:

| File | Role |
| --- | --- |
| `detector/fetcher.py` | Async URL fetching (httpx, retries, custom UA) |
| `detector/extractor.py` | Trafilatura wrapper for main-article extraction |
| `detector/chunker.py` | 512-token windowing for the model context limit |
| `detector/model.py` | Loads the Desklib model, exposes `predict` / `predict_batch` |
| `detector/pipeline.py` | Orchestrates the full fetch → extract → analyze flow |
| `app.py` | Streamlit UI (input tabs, progress, results, downloads) |

---

## Output

The UI shows only the four labels (no numerical scores) so reviewers don't
overweight noisy probabilities. Two CSV exports are available:

**Default — `ai_detector_results.csv`:**

```
url,word_count,ai_label,status
https://example.com/post-1,1842,Moderate,success
https://example.com/post-3,87,,insufficient_content
```

**Detailed — `ai_detector_results_detailed.csv` (research-grade):**

```
url,word_count,ai_probability,ai_label,n_chunks,extraction_method,model_version,model_commit,scan_timestamp,status
https://example.com/post-1,1842,0.34,Moderate,4,trafilatura,desklib/ai-text-detector-v1.01,<sha>,2026-04-30T14:30:22Z,success
```

The detailed CSV preserves the underlying probability, the model commit hash,
and the scan timestamp — everything needed to cite the run in a paper.

### Status values

| Status | Meaning |
| --- | --- |
| `success` | Page fetched, extracted, scored |
| `fetch_failed: <reason>` | URL unreachable (timeout, 4xx/5xx, DNS, etc.) |
| `extraction_failed` | Trafilatura could not isolate main-article text |
| `insufficient_content` | Article is below the minimum word count threshold |
| `error: <message>` | Unexpected failure during inference |

---

## Label thresholds

Average AI-probability across chunks is binned as:

| Probability | Label |
| --- | --- |
| `< 0.25` | Low |
| `0.25–0.49` | Moderate |
| `0.50–0.74` | High |
| `≥ 0.75` | Very High |

These thresholds match the conventions used by Originality.ai and Ahrefs
Site Explorer.

---

## Citing this tool

When publishing research that relies on this pipeline, cite at minimum:

* **Model:** `desklib/ai-text-detector-v1.01` (Hugging Face)
* **Model commit hash:** captured in every detailed CSV row (`model_commit`)
* **Scan timestamp:** captured in every detailed CSV row (`scan_timestamp`)
* **Extraction:** Trafilatura `1.12.2`

The detailed CSV is sufficient on its own to reproduce the methodology section
of any paper using this tool.

---

## Limitations

This tool is honest about its uncertainty. Read these before drawing
conclusions from the labels.

1. **Training-distribution drift.** The Desklib detector was trained on outputs
   from 11 LLMs covered by the RAID benchmark — GPT-2, GPT-3, GPT-4, ChatGPT,
   Llama 2, Mistral, Cohere, MPT, and several chat variants. It generalises to
   newer families like GPT-5, Claude Opus 4+, Gemini 3+, and Llama 4 via
   shared statistical patterns, but accuracy on those generations has not been
   directly benchmarked.
2. **Editing destroys signal.** Light human editing of AI-generated text
   degrades detection accuracy significantly. A piece that was AI-drafted and
   then revised by a human will frequently score lower than expected.
3. **False positives on stylised writing.** Highly structured genres (legal,
   scientific abstracts, technical documentation, formal academic prose) can
   produce false positives even when entirely human-written, because their
   surface statistics resemble model output.
4. **Short content is unreliable.** Pages under the minimum word threshold
   (default 150) are skipped — detection on short text is unreliable per the
   RAID paper. Do not coerce a label onto short articles.
5. **Labels are coarse.** Two pages both labelled `High` may have meaningfully
   different underlying scores (e.g., 0.51 vs 0.74). For granular research,
   use the detailed CSV.
6. **Model versioning matters.** Detector accuracy can shift between commits
   of the same Hugging Face model. Always cite the commit hash recorded in
   the detailed CSV when publishing.

---

## Performance targets

| Hardware | 1,000 URLs |
| --- | --- |
| Apple Silicon / consumer CPU | < 15 minutes |
| Consumer GPU (RTX 3060 or better) | < 5 minutes |

Memory ceiling: ~4 GB RAM (excluding the ~1.5 GB model weights).

---

## Don'ts (project guard-rails)

* No paid APIs (OpenAI, Anthropic, Originality.ai, GPTZero, Copyleaks, etc.).
* No probabilities exposed in the main UI — labels only.
* No synchronous `requests` library — all fetching is `httpx` async.
* One bad URL never kills the batch.
* No silent truncation of long articles — chunk and average instead.
* No Selenium / Playwright. Trafilatura handles 90%+ of pages.
* No authentication, accounts, or database. This is a single-user research tool.

---

## Module-level smoke tests

Each module has a self-contained `__main__` block. After installing
dependencies you can sanity-check pieces in isolation:

```bash
python -m detector.model         # loads the detector and runs two short samples
python -m detector.chunker       # exercises the 512-token windowing
python -m detector.extractor     # tests trafilatura on a tiny HTML snippet
python -m detector.fetcher       # fetches a few URLs concurrently
python -m detector.pipeline      # runs the full pipeline on three sample URLs
```
