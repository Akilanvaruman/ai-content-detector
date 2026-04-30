"""Trafilatura wrapper. Returns clean main-article text or None on failure."""

from __future__ import annotations

from dataclasses import dataclass

import trafilatura


@dataclass
class ExtractionResult:
    text: str | None
    word_count: int


def extract(html: str | None) -> ExtractionResult:
    """Extract main article text. Empty/None HTML returns ExtractionResult(None, 0)."""
    if not html:
        return ExtractionResult(text=None, word_count=0)

    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            deduplicate=True,
            favor_precision=True,
        )
    except Exception:
        return ExtractionResult(text=None, word_count=0)

    if not text:
        return ExtractionResult(text=None, word_count=0)

    text = text.strip()
    if not text:
        return ExtractionResult(text=None, word_count=0)

    return ExtractionResult(text=text, word_count=len(text.split()))


if __name__ == "__main__":
    sample_html = """
    <html><body>
      <header>Site nav goes here, ignored</header>
      <article>
        <h1>Why we love cats</h1>
        <p>Cats are graceful, mysterious, and very very fluffy.</p>
        <p>They sleep most of the day and judge you the rest.</p>
      </article>
      <footer>copyright stuff</footer>
    </body></html>
    """
    result = extract(sample_html)
    print(f"words={result.word_count}")
    print(f"text={result.text!r}")

    print(extract(""))
    print(extract(None))
    print(extract("<html><body><p>tiny</p></body></html>"))
