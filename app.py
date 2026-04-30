"""Streamlit UI for the AI Content Detector."""

from __future__ import annotations

import asyncio
import io
from typing import List

import nest_asyncio
import pandas as pd
import streamlit as st

from detector.model import MODEL_NAME, get_detector
from detector.pipeline import (
    DEFAULT_CSV_COLUMNS,
    DETAILED_CSV_COLUMNS,
    ProgressEvent,
    analyze_urls,
    normalize_urls,
    to_default_csv,
    to_detailed_csv,
)

MAX_BATCH = 1000

nest_asyncio.apply()  # let Streamlit's event loop host our asyncio.run()
st.set_page_config(page_title="AI Content Detector", page_icon=None, layout="wide")


# --------------------------------------------------------------------------- #
# Detector load (cached for the life of the Streamlit session)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading detector model (first run downloads ~1.5 GB)…")
def _load_detector():
    return get_detector()


# --------------------------------------------------------------------------- #
# Helpers for the three input modes
# --------------------------------------------------------------------------- #

def _urls_from_paste(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _urls_from_csv(file) -> List[str]:
    df = pd.read_csv(file)
    # Find the URL column case-insensitively; fallback to the first column.
    url_col = next((c for c in df.columns if c.strip().lower() == "url"), df.columns[0])
    return [str(x).strip() for x in df[url_col].dropna().tolist()]


def _urls_from_txt(file) -> List[str]:
    raw = file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return _urls_from_paste(raw)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("AI Content Detector")
    st.caption("Bulk URL analysis using the Desklib RAID-leader detector.")

    concurrency = st.slider("Concurrency (parallel fetchers)", min_value=1, max_value=50, value=20)
    min_words = st.slider("Minimum word count", min_value=50, max_value=500, value=150, step=10)

    st.divider()
    st.subheader("Model")
    st.write(f"**Name:** `{MODEL_NAME}`")
    st.write("**Architecture:** DeBERTa-v3-large + linear head")
    st.write("**Benchmark:** RAID leader (open-source AI text detection)")

    detector_obj = _load_detector()
    info = detector_obj.info()
    st.write(f"**Commit:** `{info.commit_hash[:12] if info.commit_hash != 'unknown' else 'unknown'}`")
    st.write(f"**Device:** `{info.device}`")
    st.write(f"**Head:** `{info.head}`")

    st.divider()
    st.caption(
        "Labels: Low (<0.25) · Moderate (<0.50) · High (<0.75) · Very High (≥0.75). "
        "Detailed CSV preserves the underlying probability for research."
    )


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #

st.header("Bulk URL AI Detection")
st.write(
    "Paste URLs, upload a CSV (with a `URL` column), or upload a `.txt` file with one URL per line. "
    f"Max **{MAX_BATCH}** URLs per batch."
)

paste_tab, csv_tab, txt_tab = st.tabs(["Paste URLs", "Upload CSV", "Upload TXT"])

with paste_tab:
    pasted = st.text_area("One URL per line", height=220, key="paste_input")

with csv_tab:
    csv_file = st.file_uploader("CSV with a URL column", type=["csv"], key="csv_input")

with txt_tab:
    txt_file = st.file_uploader("Plain text, one URL per line", type=["txt"], key="txt_input")


def _gather_urls() -> List[str]:
    sources: List[List[str]] = []
    if pasted:
        sources.append(_urls_from_paste(pasted))
    if csv_file is not None:
        try:
            csv_file.seek(0)
            sources.append(_urls_from_csv(csv_file))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    if txt_file is not None:
        try:
            txt_file.seek(0)
            sources.append(_urls_from_txt(txt_file))
        except Exception as e:
            st.error(f"Could not read TXT: {e}")
    flat: List[str] = []
    for s in sources:
        flat.extend(s)
    return normalize_urls(flat)


urls = _gather_urls()

count = len(urls)
over_limit = count > MAX_BATCH
if count == 0:
    st.info("Add URLs via paste, CSV upload, or TXT upload to enable analysis.")
elif over_limit:
    st.error(f"You have **{count}** URLs. The maximum batch size is **{MAX_BATCH}**. Please split your input.")
else:
    st.success(f"Ready to analyze **{count}** unique URL(s).")

analyze_disabled = count == 0 or over_limit
go = st.button("Analyze", type="primary", disabled=analyze_disabled, use_container_width=True)


# --------------------------------------------------------------------------- #
# Run analysis
# --------------------------------------------------------------------------- #

if go and not analyze_disabled:
    progress_bar = st.progress(0.0, text="Starting…")
    status_line = st.empty()

    def _on_progress(ev: ProgressEvent):
        frac = ev.done / max(ev.total, 1)
        # Fetch and analyze are roughly two halves of the total work.
        if ev.stage == "fetching":
            display = 0.5 * frac
        else:
            display = 0.5 + 0.5 * frac
        progress_bar.progress(min(display, 1.0), text=f"{ev.stage}: {ev.done}/{ev.total}")
        status_line.caption(f"{ev.stage} → {ev.current_url}")

    with st.spinner("Running pipeline…"):
        df = asyncio.run(
            analyze_urls(
                urls,
                min_words=min_words,
                concurrency=concurrency,
                on_progress=_on_progress,
            )
        )

    progress_bar.progress(1.0, text="Done.")
    status_line.empty()

    st.session_state["last_results"] = df


# --------------------------------------------------------------------------- #
# Results display + downloads
# --------------------------------------------------------------------------- #

results_df: pd.DataFrame | None = st.session_state.get("last_results")

if results_df is not None and not results_df.empty:
    st.subheader("Results")
    summary_cols = st.columns(5)
    label_counts = results_df["ai_label"].value_counts().to_dict()
    summary_cols[0].metric("Total", len(results_df))
    summary_cols[1].metric("Low", label_counts.get("Low", 0))
    summary_cols[2].metric("Moderate", label_counts.get("Moderate", 0))
    summary_cols[3].metric("High", label_counts.get("High", 0))
    summary_cols[4].metric("Very High", label_counts.get("Very High", 0))

    # Visible table — labels only, no probabilities (per spec).
    visible = results_df[DEFAULT_CSV_COLUMNS].rename(
        columns={"url": "URL", "word_count": "Word Count", "ai_label": "AI Label", "status": "Status"}
    )
    st.dataframe(visible, use_container_width=True, hide_index=True)

    download_cols = st.columns(2)
    download_cols[0].download_button(
        label="Download Results (CSV)",
        data=to_default_csv(results_df),
        file_name="ai_detector_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        label="Download Detailed CSV (with probabilities)",
        data=to_detailed_csv(results_df),
        file_name="ai_detector_results_detailed.csv",
        mime="text/csv",
        use_container_width=True,
    )
