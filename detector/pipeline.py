"""Orchestrator: chains fetch → extract → chunk → predict → label.

Public entry point: `analyze_urls(urls, min_words, concurrency)`.

The function is async because URL fetching is async; model inference itself runs
synchronously on the calling thread (that's fine — the heavy work is GPU/CPU
bound, not I/O bound).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .chunker import chunk_text, word_count
from .extractor import extract
from .fetcher import fetch_all, FetchResult
from .model import get_detector, to_label

EXTRACTION_METHOD = "trafilatura"


# ---- progress event passed to the UI ----------------------------------------

@dataclass
class ProgressEvent:
    done: int
    total: int
    current_url: str
    stage: str  # "fetching" or "analyzing"


# ---- the row we ultimately emit per URL --------------------------------------

@dataclass
class AnalysisRow:
    url: str
    word_count: int = 0
    ai_probability: float | None = None
    ai_label: str = ""
    n_chunks: int = 0
    extraction_method: str = EXTRACTION_METHOD
    model_version: str = ""
    model_commit: str = ""
    scan_timestamp: str = ""
    status: str = ""

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "word_count": self.word_count,
            "ai_probability": self.ai_probability,
            "ai_label": self.ai_label,
            "n_chunks": self.n_chunks,
            "extraction_method": self.extraction_method,
            "model_version": self.model_version,
            "model_commit": self.model_commit,
            "scan_timestamp": self.scan_timestamp,
            "status": self.status,
        }


# ---- input cleanup -----------------------------------------------------------

def normalize_urls(raw: list[str]) -> list[str]:
    """Strip whitespace, drop blanks, dedupe (preserving order), keep only http(s)."""
    seen: set[str] = set()
    out: list[str] = []
    for line in raw:
        if not isinstance(line, str):
            continue
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


# ---- per-URL analysis (extract + chunk + predict + label) --------------------

def _analyze_one(
    fetch_result: FetchResult,
    detector,
    min_words: int,
    timestamp: str,
) -> AnalysisRow:
    info = detector.info()
    row = AnalysisRow(
        url=fetch_result.url,
        model_version=info.model_name,
        model_commit=info.commit_hash,
        scan_timestamp=timestamp,
    )

    if not fetch_result.ok:
        row.status = f"fetch_failed: {fetch_result.error or 'unknown'}"
        return row

    extracted = extract(fetch_result.html)
    if extracted.text is None:
        row.status = "extraction_failed"
        return row

    row.word_count = extracted.word_count

    if extracted.word_count < min_words:
        row.status = "insufficient_content"
        return row

    try:
        chunks = chunk_text(extracted.text, detector.tokenizer)
        if not chunks:
            row.status = "extraction_failed"
            return row

        # Record chunk count up front so error rows are still informative.
        row.n_chunks = len(chunks)

        probs = detector.predict_batch(chunks)
        if not probs:
            row.status = "error: no model output"
            return row

        avg = sum(probs) / len(probs)
        row.ai_probability = round(avg, 6)
        row.ai_label = to_label(avg)
        row.status = "success"
    except Exception as e:
        row.status = f"error: {type(e).__name__}: {e}"

    return row


# ---- public API --------------------------------------------------------------

ProgressCallback = Callable[[ProgressEvent], None]


async def analyze_urls(
    urls: list[str],
    min_words: int = 150,
    concurrency: int = 40,
    on_progress: ProgressCallback | None = None,
    skip_existing_csv: str | None = None,
) -> pd.DataFrame:
    """Run the full pipeline. Returns a DataFrame in detailed-CSV column order.

    If `skip_existing_csv` points to a prior results CSV, any URLs in it with
    status == "success" are skipped, and those prior rows are carried into the
    output so the result is the union of "already done" + "newly processed".
    """
    urls = normalize_urls(urls)

    prior_rows = pd.DataFrame(columns=list(AnalysisRow(url="").as_dict().keys()))
    if skip_existing_csv:
        prior = pd.read_csv(skip_existing_csv)
        if "url" in prior.columns and "status" in prior.columns:
            done_mask = prior["status"].astype(str) == "success"
            already_done = set(prior.loc[done_mask, "url"].astype(str).tolist())
            if already_done:
                urls = [u for u in urls if u not in already_done]
                prior_rows = prior.loc[done_mask].copy()

    total = len(urls)
    if total == 0:
        if not prior_rows.empty:
            return prior_rows.reset_index(drop=True)
        return pd.DataFrame(columns=list(AnalysisRow(url="").as_dict().keys()))

    # Stage 1 — fetch URLs concurrently.
    fetch_done = 0

    def fetch_progress(result: FetchResult, done: int, _total: int):
        nonlocal fetch_done
        fetch_done = done
        if on_progress is not None:
            on_progress(ProgressEvent(done=done, total=total, current_url=result.url, stage="fetching"))

    fetch_results = await fetch_all(urls, concurrency=concurrency, on_progress=fetch_progress)

    # Stage 2 — extract + run the detector per URL (sequential; model is heavy).
    detector = get_detector()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[AnalysisRow] = []
    for i, fr in enumerate(fetch_results, start=1):
        row = _analyze_one(fr, detector, min_words=min_words, timestamp=timestamp)
        rows.append(row)
        if on_progress is not None:
            on_progress(ProgressEvent(done=i, total=total, current_url=fr.url, stage="analyzing"))

    df = pd.DataFrame([r.as_dict() for r in rows])
    if not prior_rows.empty:
        df = pd.concat([prior_rows, df], ignore_index=True)
    return df


# ---- CSV column projections used by the UI download buttons -----------------

DEFAULT_CSV_COLUMNS = ["url", "word_count", "ai_label", "status"]
DETAILED_CSV_COLUMNS = [
    "url",
    "word_count",
    "ai_probability",
    "ai_label",
    "n_chunks",
    "extraction_method",
    "model_version",
    "model_commit",
    "scan_timestamp",
    "status",
]


def to_default_csv(df: pd.DataFrame) -> str:
    return df[DEFAULT_CSV_COLUMNS].to_csv(index=False)


def to_detailed_csv(df: pd.DataFrame) -> str:
    return df[DETAILED_CSV_COLUMNS].to_csv(index=False)


if __name__ == "__main__":
    sample = [
        "https://example.com",
        "not a url",
        "https://httpbin.org/status/404",
    ]

    def progress(ev: ProgressEvent):
        print(f"  [{ev.stage} {ev.done}/{ev.total}] {ev.current_url}")

    df = asyncio.run(analyze_urls(sample, min_words=50, concurrency=5, on_progress=progress))
    print(df[DEFAULT_CSV_COLUMNS].to_string(index=False))
