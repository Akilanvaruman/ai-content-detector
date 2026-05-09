"""Split long article text into 450-token windows with 25-token overlap.

DeBERTa-v3-large has a 512-token context limit. We use 450 + 25 overlap so the
windowing is not flush against the model's hard limit (truncation in the
tokenizer still kicks in at 512 to be safe).
"""

from __future__ import annotations

from typing import Iterable

WINDOW_TOKENS = 450
OVERLAP_TOKENS = 25
MAX_MODEL_TOKENS = 512


def chunk_text(text: str, tokenizer) -> list[str]:
    """Return one or more decoded text chunks. Empty input returns []."""
    text = (text or "").strip()
    if not text:
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids: list[int] = encoded["input_ids"]
    if not ids:
        return []

    if len(ids) <= MAX_MODEL_TOKENS:
        return [text]

    step = WINDOW_TOKENS - OVERLAP_TOKENS  # 425
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + WINDOW_TOKENS, len(ids))
        window_ids = ids[start:end]
        decoded = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
        if end >= len(ids):
            break
        start += step
    return chunks


def word_count(text: str) -> int:
    return len((text or "").split())


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("desklib/ai-text-detector-v1.01")

    short = "Hello world. " * 20
    long = "The cat sat on the mat. " * 400  # ~2400 words, well past 512 tokens
    print(f"short: {len(chunk_text(short, tok))} chunk(s)")
    print(f"long:  {len(chunk_text(long, tok))} chunk(s)")
    print(f"empty: {len(chunk_text('', tok))} chunk(s)")
