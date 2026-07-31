"""SEC EDGAR vendor: bounded access to 10-K/10-Q/8-K filings.

EDGAR is keyless, but SEC's fair-access policy (https://www.sec.gov/os/accessing-edgar-data)
requires a descriptive ``User-Agent`` identifying the client and caps traffic at
roughly 10 requests/second; throttling is signalled with HTTP 403. Complete
primary documents are converted to plain text and cached locally, but tool
responses are deliberately bounded: callers first receive an index, then use
search or offset-based reads for exact excerpts. This avoids provider tool-output
limits without giving an agent arbitrary filesystem access.

Filed documents are immutable, and CIK assignments are stable, so three layers
are cached to plain text files with NO TTL (mirroring the earnings-call cache in
``alpha_vantage_fundamentals.py``): the ticker->CIK mapping, the "latest filing
on or before date D" resolution, and the extracted filing text. A warm lookup
makes zero HTTP calls.
"""

import json
import logging
import os
import re
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

import requests

from .config import get_config
from .errors import VendorRateLimitError
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

_SEC_WWW = "https://www.sec.gov"
_SEC_DATA = "https://data.sec.gov"
_COMPANY_TICKERS_URL = f"{_SEC_WWW}/files/company_tickers.json"
REQUEST_TIMEOUT = 30

# SEC requires a User-Agent in "Name contact@email" form and rejects other
# shapes (a parenthesized or URL-bearing UA gets HTTP 403). Set
# SEC_EDGAR_USER_AGENT="Your Name your@email.com" to identify yourself per SEC
# policy; the default is a compliant framework placeholder.
_DEFAULT_UA = "TradingAgents/0.3 tradingagents@example.com"

# Exact form matches only — amendments (10-K/A, 10-Q/A) are excluded because
# they restate a filing the analyst may already have seen.
_SUPPORTED_FORMS = ("10-K", "10-Q")

_FORM_ALIASES = {
    "LATEST": "latest",
    "10-K": "10-K",
    "10K": "10-K",
    "10-Q": "10-Q",
    "10Q": "10-Q",
    "8-K": "8-K",
    "8K": "8-K",
}

# 8-K event filings are frequent and individually small, so a window of recent
# events is more useful than a single latest filing.
_8K_LOOKBACK_DAYS = 90
_MAX_8K_FILINGS = 4

# Claude Code and some chat APIs spill oversized tool results to files that the
# analysis agent cannot (and should not) read. Keep every public response below
# this provider-neutral ceiling while retaining the complete document in cache.
MAX_TOOL_OUTPUT_CHARS = 20_000
DEFAULT_PAGE_CHARS = 12_000
MAX_PAGE_CHARS = 16_000
MAX_SEARCH_QUERIES = 12
MAX_MATCHES_PER_QUERY = 3
MAX_SEARCH_CONTEXT_CHARS = 800
_MAX_INDEX_HEADINGS = 120

# Inline-XBRL filings hide machine-readable metadata inside <ix:header>; strip
# it before parsing because HTMLParser handles namespaced tags inconsistently.
_IX_HEADER_RE = re.compile(r"(?is)<ix:header.*?</ix:header>")

_SECTION_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:"
    r"PART\s+[IVXLC]+(?:\b|\.)|"
    r"ITEM\s+\d+[A-Z]?(?:\.\d+)?(?:\b|\.)|"
    r"8-K\s+FILED\b|"
    r"NOTE\s+\d+[A-Z]?(?:\b|\.)|"
    r"NOTES?\s+TO\s+(?:THE\s+)?(?:CONDENSED\s+)?CONSOLIDATED|"
    r"FINANCIAL\s+STATEMENTS(?:\s+AND\s+SUPPLEMENTARY\s+DATA)?"
    r")",
    re.IGNORECASE,
)

_SELECTION_KEYS = {"form", "filing_date", "report_date", "accession", "primary_doc"}


def _user_agent() -> str:
    """Return the User-Agent for EDGAR requests (env override, read per call)."""
    return os.environ.get("SEC_EDGAR_USER_AGENT", "").strip() or _DEFAULT_UA


def _edgar_get(url: str) -> str:
    """Fetch ``url`` from EDGAR. Single HTTP seam so tests can stub it.

    SEC signals fair-access throttling with HTTP 403 (not 429); map it to
    ``VendorRateLimitError`` so the router treats it as a transient throttle.
    """
    resp = requests.get(
        url,
        headers={"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 403:
        raise VendorRateLimitError(
            f"SEC EDGAR refused the request (HTTP 403) for {url}. SEC requires a "
            "'Name email' User-Agent; set SEC_EDGAR_USER_AGENT to identify your client."
        )
    resp.raise_for_status()
    return resp.text


def _sec_cache_path(kind: str, name: str) -> Path:
    """Build the on-disk path for a SEC-filings cache entry.

    ``kind`` is a subdirectory (``cik``/``selection``/``filings``) and ``name``
    the file stem. Config is read lazily so tests can redirect ``data_cache_dir``.
    """
    root = Path(get_config()["data_cache_dir"]) / "sec_filings"
    return root / kind / f"{name}.txt"


def _read_text_cache(path: Path) -> str | None:
    """Return the cached file's stripped contents, or ``None`` if missing/empty.

    Never serves an empty file (mirrors the poisoned-cache guard in
    ``stockstats_utils.py``); any read error degrades to a cache miss.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def _write_text_cache(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` (creating parents). Never writes empty content.

    A write failure is swallowed: caching is a best-effort optimization and must
    not break the fetch path.
    """
    if not text or not text.strip():
        return
    try:
        os.makedirs(path.parent, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


# In-process cache of the full ticker->CIK map. The company list is ~1MB and
# identical across every ticker, so it is downloaded and parsed at most once per
# process; per-ticker resolutions still persist to the file cache. Reset in tests
# via the autouse cache-isolation fixture.
_TICKER_MAP: dict[str, str] | None = None


def _load_ticker_map() -> dict[str, str] | None:
    """Return the ticker->CIK map, downloading and memoizing it once per process."""
    global _TICKER_MAP
    if _TICKER_MAP is not None:
        return _TICKER_MAP
    try:
        payload = json.loads(_edgar_get(_COMPANY_TICKERS_URL))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    by_ticker: dict[str, str] = {}
    for row in payload.values():
        if isinstance(row, dict) and row.get("ticker") and row.get("cik_str") is not None:
            by_ticker[str(row["ticker"]).upper()] = str(row["cik_str"])
    _TICKER_MAP = by_ticker
    return _TICKER_MAP


def _resolve_cik(ticker: str) -> str | None:
    """Resolve a ticker to its SEC CIK (decimal string), or ``None`` if unknown.

    CIK assignments are stable, so a resolved value is cached with no TTL. A
    miss is NOT cached — a transient bad download of the company list must not
    permanently mark a real filer as unknown. Class-share tickers are retried
    with the EDGAR dash convention (``BRK.B`` -> ``BRK-B``).
    """
    try:
        safe = safe_ticker_component(ticker).upper()
    except ValueError:
        return None

    cached = _read_text_cache(_sec_cache_path("cik", safe))
    if cached is not None and cached.isdigit():
        return cached

    by_ticker = _load_ticker_map()
    if by_ticker is None:
        return None

    cik = by_ticker.get(safe) or by_ticker.get(safe.replace(".", "-"))
    if cik is None or not cik.isdigit():
        return None
    _write_text_cache(_sec_cache_path("cik", safe), cik)
    return cik


def _pick_filing(submissions: dict, form_type: str, curr_date: str | None) -> dict | None:
    """Pick the newest qualifying filing from an EDGAR submissions payload.

    ``filings.recent`` holds parallel arrays ordered newest-first (~1000 most
    recent filings — many years of 10-K/10-Q coverage; the older paginated index
    files are intentionally not fetched). ``form_type`` is ``"10-K"``,
    ``"10-Q"``, or ``"latest"`` (either form). When ``curr_date`` is set, a row
    qualifies only if its filing date is non-empty and on or before it — a
    filing is public from its filing date, and an unconfirmable date is skipped
    so a not-yet-public filing can't leak into a backtest (no look-ahead).
    """
    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        return None
    forms = recent.get("form")
    dates = recent.get("filingDate")
    accessions = recent.get("accessionNumber")
    docs = recent.get("primaryDocument")
    reports = recent.get("reportDate")
    # reportDate is only a display field, so — like ``_pick_8k_filings`` — it is
    # accessed defensively rather than gating the whole resolution: a payload
    # missing/nulling it must not falsely resolve to "no filing found".
    if not all(isinstance(col, list) for col in (forms, dates, accessions, docs)):
        return None

    wanted = _SUPPORTED_FORMS if form_type == "latest" else (form_type,)
    for i, form in enumerate(forms):
        if form not in wanted:
            continue
        try:
            filing_date = dates[i]
            accession = accessions[i]
            primary_doc = docs[i]
        except IndexError:
            continue
        if not (filing_date and accession and primary_doc):
            continue
        if curr_date and filing_date > curr_date:
            continue
        report_date = reports[i] if isinstance(reports, list) and i < len(reports) else ""
        return {
            "form": form,
            "filing_date": filing_date,
            "report_date": report_date or "",
            "accession": accession,
            "primary_doc": primary_doc,
        }
    return None


def _pick_8k_filings(submissions: dict, curr_date: str | None) -> list[dict]:
    """Pick the recent 8-K event filings, newest first.

    Returns up to ``_MAX_8K_FILINGS`` 8-Ks filed on or before ``curr_date`` and
    within ``_8K_LOOKBACK_DAYS`` of it (no look-ahead, same discipline as
    ``_pick_filing``); without ``curr_date`` the newest filings qualify
    regardless of age. The index's ``items`` column (e.g. ``"2.02,9.01"``) is
    carried along so the digest can label each event without a document fetch.
    """
    filings = submissions.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form")
    dates = recent.get("filingDate")
    accessions = recent.get("accessionNumber")
    docs = recent.get("primaryDocument")
    reports = recent.get("reportDate")
    items_col = recent.get("items")
    if not all(isinstance(col, list) for col in (forms, dates, accessions, docs)):
        return []

    floor = None
    if curr_date:
        try:
            floor = (date.fromisoformat(curr_date) - timedelta(days=_8K_LOOKBACK_DAYS)).isoformat()
        except ValueError:
            floor = None

    picked: list[dict] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        try:
            filing_date = dates[i]
            accession = accessions[i]
            primary_doc = docs[i]
        except IndexError:
            continue
        if not (filing_date and accession and primary_doc):
            continue
        if curr_date and filing_date > curr_date:
            continue
        if floor and filing_date < floor:
            break  # rows are newest-first: everything after is older still
        report_date = reports[i] if isinstance(reports, list) and i < len(reports) else ""
        items = items_col[i] if isinstance(items_col, list) and i < len(items_col) else ""
        picked.append(
            {
                "form": form,
                "filing_date": filing_date,
                "report_date": report_date or "",
                "accession": accession,
                "primary_doc": primary_doc,
                "items": items or "",
            }
        )
        if len(picked) >= _MAX_8K_FILINGS:
            break
    return picked


def _load_cached_8k_selection(path: Path) -> list[dict] | None:
    """Read and validate a cached 8-K selection list; ``None`` on any doubt."""
    raw = _read_text_cache(path)
    if raw is None:
        return None
    try:
        metas = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(metas, list) or not metas:
        return None
    for meta in metas:
        if not (
            isinstance(meta, dict)
            and meta.keys() >= {"form", "filing_date", "accession", "primary_doc"}
            and meta.get("accession")
            and meta.get("primary_doc")
        ):
            return None
    return metas


def _load_cached_selection(path: Path) -> dict | None:
    """Read and validate a cached filing-selection entry; ``None`` on any doubt."""
    raw = _read_text_cache(path)
    if raw is None:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(meta, dict)
        and meta.keys() >= _SELECTION_KEYS
        and meta.get("form") in _SUPPORTED_FORMS
        and meta.get("accession")
        and meta.get("primary_doc")
    ):
        return meta
    return None


_SKIP_TAGS = {"script", "style"}
_BLOCK_TAGS = {
    "p",
    "div",
    "tr",
    "br",
    "li",
    "table",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
}
# Table cells separate on the same line, not a newline: a financial-statement
# row like <tr><td>Revenue</td><td>1,000</td><td>900</td></tr> must extract to
# "Revenue 1,000 900", never the glued "Revenue1,000900" that corrupts the
# numbers the analyst depends on.
_CELL_TAGS = {"td", "th"}


class _TextExtractor(HTMLParser):
    """Extract visible text from filing HTML, newline-separating block elements."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in _CELL_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in _CELL_TAGS:
            self._parts.append(" ")

    def handle_data(self, data):
        if not self._skip_depth and data:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    """Convert filing HTML to normalized plain text.

    A plain-text primary document (older filings) passes through unchanged
    apart from whitespace normalization — no tags means no extraction loss.
    """
    parser = _TextExtractor()
    parser.feed(_IX_HEADER_RE.sub(" ", html))
    parser.close()
    text = parser.text()
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_filing(ticker: str, meta: dict, full_text: str) -> str:
    """Render a filing's complete text into a readable report with a header."""
    header = [
        f"SEC Filing — {ticker} {meta['form']}",
        f"Filed: {meta['filing_date']} | Period: {meta.get('report_date') or 'n/a'} | "
        f"Accession: {meta['accession']}",
        "Source: SEC EDGAR (complete filing text)",
        "",
    ]
    body = full_text or "No text could be extracted from this filing."
    return "\n".join(header + [body]).strip()


def _bounded_output(text: str, suffix: str = "\n\n[Additional output omitted.]") -> str:
    """Return ``text`` within the hard public tool-output ceiling."""
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    available = max(0, MAX_TOOL_OUTPUT_CHARS - len(suffix))
    return text[:available].rstrip() + suffix


def _document_label(document: str) -> str:
    """Return the first non-empty document line for compact response headers."""
    return next((line.strip() for line in document.splitlines() if line.strip()), "SEC Filing")


def _document_header_lines(document: str) -> list[str]:
    """Return the compact source metadata prepended to a cached document."""
    return [line.strip() for line in document.splitlines()[:3] if line.strip()]


def _looks_like_heading(line: str) -> bool:
    """Recognize explicit SEC headings plus short all-caps statement headings."""
    if _SECTION_HEADING_RE.match(line):
        return True
    letters = [char for char in line if char.isalpha()]
    return (
        len(letters) >= 4
        and len(line) <= 140
        and len(line.split()) <= 18
        and all(char.isupper() for char in letters)
    )


def _section_index(document: str) -> list[tuple[int, str]]:
    """Return bounded ``(character offset, heading)`` entries for a filing."""
    entries: list[tuple[int, str]] = []
    offset = 0
    for raw_line in document.splitlines(keepends=True):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line and _looks_like_heading(line):
            entries.append((offset, line[:240]))
            if len(entries) >= _MAX_INDEX_HEADINGS:
                break
        offset += len(raw_line)
    return entries


def _format_filing_index(document: str) -> str:
    """Return metadata and a navigable heading index, never the full body."""
    label = _document_label(document)
    metadata = _document_header_lines(document)
    headings = _section_index(document)
    lines = [
        f"{label} — bounded index",
        *metadata[1:],
        f"Document length: {len(document):,} characters (complete text cached locally)",
        "The filing body is intentionally omitted from this response.",
        "Use search_sec_filing for topic discovery and read_sec_filing for exact excerpts.",
        "Offsets below refer to character positions in the cached complete text.",
        "",
        "Suggested searches: management discussion, revenue recognition, segment, legal "
        "proceedings, commitments, leases, related parties, goodwill, impairment, "
        "valuation allowance, subsequent events.",
        "",
        "Section/heading index:",
    ]
    if headings:
        lines.extend(f"- char {offset:,}: {heading}" for offset, heading in headings)
        if len(headings) >= _MAX_INDEX_HEADINGS:
            lines.append(
                f"- Index stopped after {_MAX_INDEX_HEADINGS} headings; use search for later topics."
            )
    else:
        lines.append("- No reliable headings were detected; use search_sec_filing.")
    return _bounded_output("\n".join(lines).strip())


def _clean_queries(queries: list[str] | str) -> list[str]:
    """Normalize and deduplicate literal search phrases."""
    raw_queries = [queries] if isinstance(queries, str) else queries
    if not isinstance(raw_queries, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        if not isinstance(query, str):
            continue
        value = re.sub(r"\s+", " ", query).strip()[:200]
        key = value.casefold()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
        if len(cleaned) >= MAX_SEARCH_QUERIES:
            break
    return cleaned


def _search_document(
    document: str,
    queries: list[str],
    max_matches: int,
    context_chars: int,
) -> list[tuple[str, int, int, int, str]]:
    """Find phrases, round-robin ordered so every query gets a first result."""
    spans_by_query: list[tuple[str, list[tuple[int, int]]]] = []
    for query in queries:
        spans = [
            match.span() for match in re.finditer(re.escape(query), document, flags=re.IGNORECASE)
        ][:max_matches]
        spans_by_query.append((query, spans))

    results: list[tuple[str, int, int, int, str]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for rank in range(max_matches):
        for query, spans in spans_by_query:
            if rank >= len(spans):
                continue
            match_start, match_end = spans[rank]
            excerpt_start = max(0, match_start - context_chars)
            excerpt_end = min(len(document), match_end + context_chars)
            range_key = (excerpt_start, excerpt_end)
            if range_key not in seen_ranges:
                results.append(
                    (
                        query,
                        match_start,
                        excerpt_start,
                        excerpt_end,
                        document[excerpt_start:excerpt_end],
                    )
                )
                seen_ranges.add(range_key)
    return results


def _format_search_results(
    document: str,
    queries: list[str],
    max_matches: int,
    context_chars: int,
) -> str:
    """Render search excerpts incrementally so no result is cut mid-block."""
    label = _document_label(document)
    matches = _search_document(document, queries, max_matches, context_chars)
    matched_queries = {query.casefold() for query, *_ in matches}
    unmatched = [query for query in queries if query.casefold() not in matched_queries]
    parts = [
        f"SEC filing search — {label}",
        f"Document length: {len(document):,} characters",
        f"Queries: {', '.join(queries)}",
        "Offsets are exact character positions for read_sec_filing.",
    ]
    if unmatched:
        parts.append(f"No literal matches for: {', '.join(unmatched)}")
    if not matches:
        parts.extend(["", "No literal matches found. Try shorter or alternate filing terminology."])
        return "\n".join(parts)

    omitted = 0
    for number, (query, match_start, excerpt_start, excerpt_end, excerpt) in enumerate(
        matches, start=1
    ):
        block = (
            f"\n\n## Match {number}: {query!r}\n"
            f"Match offset: {match_start:,} | Excerpt range: "
            f"[{excerpt_start:,}:{excerpt_end:,}) | Recommended read offset: "
            f"{max(0, match_start - 2_000):,}\n"
            "--- BEGIN EXCERPT ---\n"
            f"{excerpt}\n"
            "--- END EXCERPT ---"
        )
        reserve = 100
        if len("\n".join(parts)) + len(block) + reserve > MAX_TOOL_OUTPUT_CHARS:
            omitted = len(matches) - number + 1
            break
        parts.append(block)
    if omitted:
        parts.append(
            f"\n\n[{omitted} additional match(es) omitted to keep the tool response bounded. "
            "Narrow the queries or request fewer matches.]"
        )
    return _bounded_output("\n".join(parts))


def _format_document_page(document: str, offset: int, max_chars: int) -> str:
    """Render one exact, bounded filing page with a stable continuation offset."""
    start = min(offset, len(document))
    end = min(len(document), start + max_chars)
    chunk = document[start:end]
    continuation = str(end) if end < len(document) else "END"
    output = (
        f"SEC filing page — {_document_label(document)}\n"
        f"Document length: {len(document):,} characters\n"
        f"Exact range: [{start:,}:{end:,}) | Next offset: {continuation}\n"
        "--- BEGIN EXACT FILING TEXT ---\n"
        f"{chunk}\n"
        "--- END EXACT FILING TEXT ---"
    )
    return _bounded_output(output)


def _fetch_submissions(cik: str) -> dict | None:
    """Fetch and parse the EDGAR submissions index for ``cik``; ``None`` if unusable."""
    try:
        submissions = json.loads(_edgar_get(f"{_SEC_DATA}/submissions/CIK{int(cik):010d}.json"))
    except (json.JSONDecodeError, TypeError):
        return None
    return submissions if isinstance(submissions, dict) else None


def _get_8k_digest(
    ticker: str, safe: str | None, cik: str, curr_date: str | None
) -> tuple[str, bool]:
    """Build a digest of the recent 8-K event filings for ``ticker``.

    Each 8-K's full extracted text is included and cached per accession (a filed
    document never changes); the selection of recent 8-Ks per (ticker, date) is
    cached like the periodic-filing selection.
    """
    metas = None
    selection_path = (
        _sec_cache_path("selection", f"{safe}-8-K-{curr_date}")
        if safe is not None and curr_date
        else None
    )
    if selection_path is not None:
        metas = _load_cached_8k_selection(selection_path)

    if metas is None:
        submissions = _fetch_submissions(cik)
        if submissions is not None:
            metas = _pick_8k_filings(submissions, curr_date)
        # Only cache a non-empty resolution; an empty or failed one may be a
        # transient payload problem and must be retried next time.
        if metas and selection_path is not None:
            _write_text_cache(selection_path, json.dumps(metas))

    if not metas:
        return (
            (
                f"No 8-K filed for {ticker} in the {_8K_LOOKBACK_DAYS} days on or before "
                f"{curr_date or 'the current date'} in SEC EDGAR."
            ),
            False,
        )

    lines = [
        f"SEC Filings — {ticker} 8-K digest",
        f"Material-event filings in the {_8K_LOOKBACK_DAYS} days on or before "
        f"{curr_date or 'the current date'} (newest first; up to {_MAX_8K_FILINGS})",
        "Source: SEC EDGAR (complete filing text)",
        "",
    ]
    fetched = 0
    last_error: Exception | None = None
    for meta in metas:
        accession_nodash = meta["accession"].replace("-", "")
        filing_path = (
            _sec_cache_path("filings", f"{safe}-{accession_nodash}") if safe is not None else None
        )
        body = _read_text_cache(filing_path) if filing_path is not None else None
        if body is None:
            # Isolate each document fetch: one 8-K's HTTP error must not discard
            # the others already selected. Only if every fetch fails do we let
            # the error propagate so the router can degrade the whole category.
            try:
                body = _html_to_text(
                    _edgar_get(
                        f"{_SEC_WWW}/Archives/edgar/data/{int(cik)}/{accession_nodash}/"
                        f"{meta['primary_doc']}"
                    )
                )
            except Exception as exc:  # noqa: BLE001 — best-effort per filing
                last_error = exc
                logger.debug("Could not fetch 8-K %s for %s: %s", meta["accession"], ticker, exc)
                body = None
            else:
                if filing_path is not None:
                    _write_text_cache(filing_path, body)
        if body is not None:
            fetched += 1

        event = f" | Event date: {meta['report_date']}" if meta.get("report_date") else ""
        items = f" | Items: {meta['items']}" if meta.get("items") else ""
        lines.append(
            f"## 8-K filed {meta['filing_date']}{event}{items} | Accession: {meta['accession']}"
        )
        lines.append(body or "Could not retrieve this filing's text.")
        lines.append("")

    # If nothing was retrievable at all, surface the failure so the optional
    # category degrades to a sentinel rather than returning an all-empty digest.
    if fetched == 0 and last_error is not None:
        raise last_error
    return "\n".join(lines).strip(), True


def _get_periodic_document(
    ticker: str,
    safe: str,
    cik: str,
    curr_date: str | None,
    form_type: str,
) -> tuple[str, bool]:
    """Resolve, fetch, and cache one complete 10-K/10-Q primary document."""
    # Selection layer: "the latest <form> filed on or before curr_date" is
    # immutable, so cache it per (ticker, form, date). A warm hit skips the
    # submissions-index call. Only a real resolution is cached — a None may be
    # a transient failure and must be retried next time.
    meta = None
    selection_path = (
        _sec_cache_path("selection", f"{safe}-{form_type}-{curr_date}") if curr_date else None
    )
    if selection_path is not None:
        meta = _load_cached_selection(selection_path)

    if meta is None:
        submissions = _fetch_submissions(cik)
        if submissions is not None:
            meta = _pick_filing(submissions, form_type, curr_date)
        if meta is not None and selection_path is not None:
            _write_text_cache(selection_path, json.dumps(meta))

    if meta is None:
        wanted = form_type if form_type != "latest" else "10-K or 10-Q"
        return (
            f"No {wanted} found for {ticker} on or before "
            f"{curr_date or 'the current date'} in SEC EDGAR.",
            False,
        )

    # Filing layer: a filed document never changes, so the extracted text is
    # cached per accession number. A warm hit skips the document download.
    accession_nodash = meta["accession"].replace("-", "")
    filing_path = _sec_cache_path("filings", f"{safe}-{accession_nodash}")
    cached = _read_text_cache(filing_path)
    if cached is not None:
        return cached, True

    html = _edgar_get(
        f"{_SEC_WWW}/Archives/edgar/data/{int(cik)}/{accession_nodash}/{meta['primary_doc']}"
    )
    text = _html_to_text(html)
    formatted = _format_filing(ticker, meta, text)
    # Only cache when real text was extracted, never an empty-document report.
    if text:
        _write_text_cache(filing_path, formatted)
    return formatted, bool(text)


def _get_filing_document(ticker: str, curr_date: str | None, form_type: str) -> tuple[str, bool]:
    """Return a complete cached document internally plus an availability flag."""
    try:
        safe = safe_ticker_component(ticker).upper()
    except ValueError:
        return (
            f"No SEC filings found for {ticker}: invalid ticker symbol.",
            False,
        )

    cik = _resolve_cik(ticker)
    if cik is None:
        return (
            (
                f"No SEC filings found for {ticker}: not in the SEC EDGAR company list "
                "(EDGAR covers US-listed filers only)."
            ),
            False,
        )

    if form_type == "8-K":
        return _get_8k_digest(ticker, safe, cik, curr_date)
    return _get_periodic_document(ticker, safe, cik, curr_date, form_type)


def _normalized_form_type(form_type: str) -> str | None:
    """Return a canonical supported form name."""
    return _FORM_ALIASES.get(str(form_type or "latest").strip().upper())


def _unsupported_form_message(form_type: str) -> str:
    return f"Unsupported form_type {form_type!r}. Use '10-K', '10-Q', '8-K', or 'latest'."


def get_sec_filing(ticker: str, curr_date: str = None, form_type: str = "latest") -> str:
    """Return a bounded metadata/heading index for a recent SEC filing.

    The complete primary document is fetched and cached, but never emitted in
    this response. Use :func:`search_sec_filing` to locate relevant disclosures
    and :func:`read_sec_filing` to retrieve exact bounded pages.
    """
    canonical = _normalized_form_type(form_type)
    if canonical is None:
        return _unsupported_form_message(form_type)
    document, available = _get_filing_document(ticker, curr_date, canonical)
    return _format_filing_index(document) if available else document


def search_sec_filing(
    ticker: str,
    curr_date: str,
    queries: list[str] | str,
    form_type: str = "latest",
    max_matches: int = 2,
    context_chars: int = 500,
) -> str:
    """Search a complete cached filing and return bounded, offset-labeled excerpts."""
    canonical = _normalized_form_type(form_type)
    if canonical is None:
        return _unsupported_form_message(form_type)
    cleaned = _clean_queries(queries)
    if not cleaned:
        return "At least one non-empty literal search query is required."
    try:
        bounded_matches = max(1, min(int(max_matches), MAX_MATCHES_PER_QUERY))
        bounded_context = max(0, min(int(context_chars), MAX_SEARCH_CONTEXT_CHARS))
    except (TypeError, ValueError):
        return "max_matches and context_chars must be integers."

    document, available = _get_filing_document(ticker, curr_date, canonical)
    if not available:
        return document
    return _format_search_results(document, cleaned, bounded_matches, bounded_context)


def read_sec_filing(
    ticker: str,
    curr_date: str,
    offset: int = 0,
    form_type: str = "latest",
    max_chars: int = DEFAULT_PAGE_CHARS,
) -> str:
    """Read one exact page from a cached filing without arbitrary file access."""
    canonical = _normalized_form_type(form_type)
    if canonical is None:
        return _unsupported_form_message(form_type)
    try:
        bounded_offset = int(offset)
        bounded_chars = max(1, min(int(max_chars), MAX_PAGE_CHARS))
    except (TypeError, ValueError):
        return "offset and max_chars must be integers."
    if bounded_offset < 0:
        return "offset must be zero or greater."

    document, available = _get_filing_document(ticker, curr_date, canonical)
    if not available:
        return document
    if bounded_offset > len(document):
        return (
            f"offset {bounded_offset:,} is beyond the document length "
            f"of {len(document):,} characters."
        )
    return _format_document_page(document, bounded_offset, bounded_chars)
