"""Download Berkshire Hathaway shareholder letters and cache them as plain text.

Builds the corpus that ``scripts/distill_buffett_principles.py`` reads to
produce the Buffett Researcher's principles document. Letters are written to
``{data_cache_dir}/buffett_letters/{year}.txt`` with no TTL, mirroring the
immutable-document caches in ``tradingagents/dataflows/sec_edgar.py``.

The letters are copyrighted material published by Berkshire Hathaway. They are
downloaded to the user's local cache for personal research and are never
committed to this repository or redistributed.

Three quirks of the source site are handled here:

- The URL scheme changes mid-corpus. 1977-1997 are ``{year}.html``, 2004
  onward are ``{year}ltr.pdf``.
- 1998-2003 serve a stub notice page that links to the real letter, which
  lives at a different path (``1998htm.html``, ``/2000ar/2000letter.html``,
  ``2002pdf.pdf``, ...). Stubs are detected by extracted length and followed.
- The site sits behind a Sucuri cache that serves ``content-encoding: br``
  from a warm entry even when the request asks for identity. A cache-busting
  query parameter forces a MISS and a plain response; ``brotli`` is also
  declared in the ``buffett`` extra so urllib3 can transparently decode a
  compressed response if one still arrives.

Usage:
    uv sync --extra buffett
    uv run python scripts/fetch_buffett_letters.py
    uv run python scripts/fetch_buffett_letters.py --years 1977-1990 --force
"""

from __future__ import annotations

import argparse
import io
import re
import secrets
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

from tradingagents.dataflows.config import get_config

BASE_URL = "https://www.berkshirehathaway.com/letters/"
INDEX_URL = urljoin(BASE_URL, "letters.html")
FIRST_LETTER_YEAR = 1977

REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.5
USER_AGENT = "TradingAgents/0.3 shareholder-letter corpus builder"

# A stub notice page extracts to a few hundred characters; the shortest real
# letter is over 15k. Anything under this is treated as a pointer to follow.
STUB_MAX_CHARS = 3_000
MAX_STUB_HOPS = 2

_HREF_RE = re.compile(r"""(?i)href\s*=\s*["']?([^"'> ]+)""")
_LETTER_HREF_RE = re.compile(r"(?i)^(\d{4})(?:ltr)?\.(?:html|pdf)$")

_SKIP_TAGS = {"script", "style", "head"}
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}


class _TextExtractor(HTMLParser):
    """Extract visible text from letter HTML, newline-separating block elements."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth and data:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _normalize(text: str) -> str:
    """Collapse the fixed-width padding of the older letters while keeping lines."""
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Convert a letter's HTML to normalized plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return _normalize(parser.text())


def pdf_to_text(payload: bytes) -> str:
    """Convert a letter PDF to normalized plain text."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "Reading the PDF-era letters needs pypdf. Install the extra first:\n"
            "    uv sync --extra buffett"
        ) from exc

    reader = PdfReader(io.BytesIO(payload))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _normalize("\n\n".join(pages))


def _cache_bust(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}nc={secrets.token_hex(4)}"


def _fetch(url: str) -> requests.Response:
    """GET ``url`` with a cache-busting parameter. Single HTTP seam for tests."""
    response = requests.get(
        _cache_bust(url),
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def _decode(payload: bytes) -> str:
    """Decode letter HTML, which is a mix of ASCII and Word-exported cp1252."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("cp1252", errors="replace")


def _is_pdf(url: str, response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "pdf" in content_type or url.lower().endswith(".pdf")


def _stub_target(html: str, page_url: str) -> str | None:
    """Return the real letter URL a notice page points at, preferring HTML."""
    candidates = [
        urljoin(page_url, href)
        for href in _HREF_RE.findall(html)
        if href.lower().endswith((".html", ".pdf")) and "adobe" not in href.lower()
    ]
    candidates = [url for url in candidates if url.split("?")[0] != page_url.split("?")[0]]
    if not candidates:
        return None
    html_first = [url for url in candidates if url.lower().endswith(".html")]
    return (html_first or candidates)[0]


def fetch_letter_text(url: str) -> str:
    """Fetch one letter, following notice-page redirects, and return plain text."""
    for _ in range(MAX_STUB_HOPS + 1):
        response = _fetch(url)
        if _is_pdf(url, response):
            return pdf_to_text(response.content)

        html = _decode(response.content)
        text = html_to_text(html)
        if len(text) >= STUB_MAX_CHARS:
            return text

        target = _stub_target(html, url)
        if target is None:
            return text
        print(f"    notice page, following {target}")
        url = target
        time.sleep(REQUEST_DELAY_SECONDS)

    raise RuntimeError(f"too many notice-page hops starting at {url}")


def discover_letter_urls() -> dict[int, str]:
    """Map year to letter URL, scraped from the index and extended past its lag."""
    index_html = _decode(_fetch(INDEX_URL).content)
    urls: dict[int, str] = {}
    for href in _HREF_RE.findall(index_html):
        match = _LETTER_HREF_RE.match(href)
        if match and int(match.group(1)) >= FIRST_LETTER_YEAR:
            urls[int(match.group(1))] = urljoin(BASE_URL, href)

    # The index page is relinked some time after a new letter goes up, so the
    # most recent year is often reachable but unlisted. Probe forward for it.
    for year in range(max(urls, default=FIRST_LETTER_YEAR) + 1, date.today().year + 1):
        candidate = urljoin(BASE_URL, f"{year}ltr.pdf")
        response = requests.head(
            _cache_bust(candidate),
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            break
        print(f"Found unlisted letter for {year}")
        urls[year] = candidate

    return urls


def corpus_dir() -> Path:
    """Return the on-disk letter corpus directory, creating it if needed."""
    path = Path(get_config()["data_cache_dir"]) / "buffett_letters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus() -> dict[int, str]:
    """Return every cached letter as year -> text, for the distillation script."""
    letters = {}
    for path in sorted(corpus_dir().glob("*.txt")):
        if not path.stem.isdigit():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            letters[int(path.stem)] = text
    return letters


def _parse_years(spec: str | None) -> set[int] | None:
    """Parse a ``1977-1990,2020`` year filter into a set, or None for all years."""
    if not spec:
        return None
    years: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, end = (int(part) for part in chunk.split("-", 1))
            years.update(range(start, end + 1))
        elif chunk:
            years.add(int(chunk))
    return years


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", help="Restrict to years, e.g. '1977-1990,2024'")
    parser.add_argument("--force", action="store_true", help="Re-download cached letters")
    args = parser.parse_args()

    wanted = _parse_years(args.years)
    destination = corpus_dir()
    print(f"Corpus directory: {destination}")

    urls = discover_letter_urls()
    years = sorted(year for year in urls if wanted is None or year in wanted)
    if not years:
        print("No letters matched the requested years.")
        return 1
    print(f"Discovered {len(urls)} letters; fetching {len(years)} ({years[0]}-{years[-1]})\n")

    written = skipped = failed = 0
    for year in years:
        path = destination / f"{year}.txt"
        if path.exists() and path.stat().st_size > 0 and not args.force:
            skipped += 1
            continue

        print(f"  {year}: {urls[year]}")
        try:
            text = fetch_letter_text(urls[year])
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed += 1
            continue

        if len(text) < STUB_MAX_CHARS:
            print(f"    FAILED: extracted only {len(text)} chars, looks incomplete")
            failed += 1
            continue

        path.write_text(
            f"# Berkshire Hathaway Shareholder Letter — {year}\n\n{text}\n", encoding="utf-8"
        )
        print(f"    wrote {len(text):,} chars")
        written += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone: {written} written, {skipped} already cached, {failed} failed.")
    if written or skipped:
        print("Next: uv run python scripts/distill_buffett_principles.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
