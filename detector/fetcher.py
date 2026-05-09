"""Async URL fetcher using httpx.

Per spec:
  * 30s timeout per URL
  * 2 retries with exponential backoff (1s, 4s)
  * Custom User-Agent
  * Up to 5 redirect hops
  * One failure must NEVER kill the batch
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; AIDetectorBot/1.0; research)"
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2
BACKOFF_SECONDS = (1.0, 4.0)
MAX_REDIRECTS = 5


@dataclass
class FetchResult:
    url: str
    html: str | None
    status_code: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.html is not None and self.error is None


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
        timeout=httpx.Timeout(TIMEOUT_SECONDS),
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
    )


async def fetch_one(client: httpx.AsyncClient, url: str) -> FetchResult:
    last_error: str | None = None
    last_status: int | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url)
            last_status = response.status_code
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}"
            else:
                ctype = response.headers.get("content-type", "")
                if "html" not in ctype.lower() and ctype:
                    return FetchResult(url, None, response.status_code, f"non-HTML content-type: {ctype}")
                return FetchResult(url, response.text, response.status_code, None)
        except httpx.TimeoutException:
            last_error = "timeout"
        except httpx.TooManyRedirects:
            last_error = "too many redirects"
            break
        except httpx.RequestError as e:
            last_error = f"{type(e).__name__}: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            break

        if attempt < MAX_RETRIES:
            await asyncio.sleep(BACKOFF_SECONDS[attempt])

    return FetchResult(url, None, last_status, last_error or "unknown error")


async def fetch_all(
    urls: list[str],
    concurrency: int = 40,
    on_progress=None,
) -> list[FetchResult]:
    """Fetch all URLs in parallel up to `concurrency` at a time.

    `on_progress` is an optional callback invoked as `on_progress(result, done, total)`
    each time a URL finishes — used by the Streamlit UI to update the progress bar.
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(urls)
    done = 0
    results: list[FetchResult] = [None] * total  # type: ignore[list-item]

    async with _build_client() as client:
        async def worker(idx: int, url: str):
            nonlocal done
            async with sem:
                result = await fetch_one(client, url)
            results[idx] = result
            done += 1
            if on_progress is not None:
                try:
                    on_progress(result, done, total)
                except Exception:
                    # Progress callbacks must never break the batch.
                    pass

        await asyncio.gather(*(worker(i, u) for i, u in enumerate(urls)))

    return results


if __name__ == "__main__":
    import sys

    test_urls = [
        "https://example.com",
        "https://httpbin.org/status/404",
        "https://this-domain-does-not-exist-aidetector.test",
    ]
    if len(sys.argv) > 1:
        test_urls = sys.argv[1:]

    async def main():
        def progress(r, done, total):
            print(f"  [{done}/{total}] {r.url}  ok={r.ok}  status={r.status_code}  err={r.error}")
        results = await fetch_all(test_urls, concurrency=5, on_progress=progress)
        for r in results:
            print(r.url, "OK" if r.ok else f"FAIL ({r.error})")

    asyncio.run(main())
