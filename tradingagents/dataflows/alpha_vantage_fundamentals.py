import contextlib
import json
import os
import re
from pathlib import Path

from .alpha_vantage_common import _make_api_request
from .config import get_config
from .utils import safe_ticker_component

# Earnings-call data is immutable for a given (ticker, date): a closed quarter's
# transcript never changes, a company's fiscal-year-end is static, and "the
# latest quarter reported on or before date D" is deterministic. So these three
# layers are cached to plain text files with NO TTL, letting a warm lookup make
# zero Alpha Vantage calls and conserve the free tier's 25 requests/day.
_QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")

_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _earnings_cache_path(kind: str, name: str) -> Path:
    """Build the on-disk path for an earnings-call cache entry.

    ``kind`` is a subdirectory (``transcripts``/``fiscal_year_end``/``quarter``)
    and ``name`` the file stem. Config is read lazily so tests can redirect
    ``data_cache_dir``.
    """
    root = Path(get_config()["data_cache_dir"]) / "earnings_call"
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


def _fiscal_year_end_month(ticker: str) -> int:
    """Return the company's fiscal-year-end month (1–12), defaulting to 12.

    Alpha Vantage labels earnings-call transcripts by the issuer's *fiscal*
    quarter, not the calendar quarter, so a company with a non-December fiscal
    year end (e.g. Apple ends in September) needs its fiscal calendar to compute
    the right ``YYYYQn`` label. The OVERVIEW endpoint's ``FiscalYearEnd`` field
    supplies it. Falls back to December (calendar year) when unavailable, which
    is correct for the majority of issuers.

    The fiscal-year-end is static per company, so a resolved month is cached to
    disk and reused on later runs to skip the OVERVIEW call. The ``12`` fallback
    is NOT cached — it may mask a transient OVERVIEW failure, so we retry next
    time rather than persist a possibly-wrong default.
    """
    try:
        safe = safe_ticker_component(ticker).upper()
    except ValueError:
        safe = None

    if safe is not None:
        cached = _read_text_cache(_earnings_cache_path("fiscal_year_end", safe))
        if cached is not None:
            with contextlib.suppress(ValueError):
                month = int(cached)
                if 1 <= month <= 12:
                    return month

    try:
        payload = json.loads(_make_api_request("OVERVIEW", {"symbol": ticker}))
    except (json.JSONDecodeError, TypeError):
        return 12
    if not isinstance(payload, dict):
        return 12
    name = str(payload.get("FiscalYearEnd", "")).strip().lower()
    month = _MONTH_NAMES.get(name)
    if month is None:
        return 12
    if safe is not None:
        _write_text_cache(_earnings_cache_path("fiscal_year_end", safe), str(month))
    return month


def _fiscal_date_to_quarter(fiscal_date_ending: str, fye_month: int = 12) -> str | None:
    """Map an ``YYYY-MM-DD`` fiscal period end to Alpha Vantage's ``YYYYQn`` label.

    The EARNINGS_CALL_TRANSCRIPT endpoint requires an explicit quarter string
    (e.g. ``2024Q1``) labeled by the issuer's *fiscal* year and quarter; the
    EARNINGS endpoint reports periods only as fiscal end dates. Given the
    fiscal-year-end month, the quarter number counts from the start of the fiscal
    year and the fiscal-year label is the calendar year the fiscal year ends in.
    With the default ``fye_month=12`` this reduces to plain calendar quarters.
    Apple (September FYE): ``2024-12-31 -> 2025Q1``, ``2025-03-31 -> 2025Q2``.
    Returns ``None`` for an unparseable date so the caller can degrade rather than
    query a bogus quarter.
    """
    try:
        year_str, month_str, _ = fiscal_date_ending.split("-")
        month = int(month_str)
        year = int(year_str)
    except (ValueError, AttributeError):
        return None
    quarter = ((month - fye_month - 1) % 12) // 3 + 1
    fiscal_year = year if month <= fye_month else year + 1
    return f"{fiscal_year:04d}Q{quarter}"


def _resolve_latest_reported_quarter(ticker: str, curr_date: str | None) -> str | None:
    """Return the newest fiscal quarter reported on or before ``curr_date``.

    Uses the EARNINGS endpoint's ``quarterlyEarnings`` (newest-first) and honors
    ``curr_date`` to avoid look-ahead — a transcript for a call that had not yet
    happened as of the trading date must not leak into the analysis. The fiscal
    end date is mapped to Alpha Vantage's fiscal ``YYYYQn`` label using the
    company's fiscal-year-end month. Returns ``None`` when no reported quarter
    qualifies or the payload is unusable.
    """
    result = _make_api_request("EARNINGS", {"symbol": ticker})
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    quarterly = payload.get("quarterlyEarnings")
    if not isinstance(quarterly, list):
        return None

    for report in quarterly:
        if not isinstance(report, dict):
            continue
        reported_date = report.get("reportedDate") or ""
        # quarterlyEarnings is ordered newest-first, so the first entry already
        # reported as of curr_date is the most recent available transcript. When
        # curr_date is set, skip any period we cannot confirm was reported on or
        # before it — a future reportedDate OR a missing/null one — so a
        # not-yet-reported quarter can't leak into a backtest (look-ahead).
        if curr_date and (not reported_date or reported_date > curr_date):
            continue
        # Only fetch the fiscal calendar once we have a qualifying period.
        fye_month = _fiscal_year_end_month(ticker)
        quarter = _fiscal_date_to_quarter(report.get("fiscalDateEnding", ""), fye_month)
        if quarter:
            return quarter
    return None


def _format_transcript(payload: dict, ticker: str, quarter: str) -> str:
    """Render an EARNINGS_CALL_TRANSCRIPT payload into a readable report.

    Surfaces each segment's per-speaker sentiment (an Alpha Vantage-provided
    score) alongside the spoken content so the analyst can weigh management tone
    and Q&A pushback, plus an average sentiment header for a quick read.
    """
    segments = payload.get("transcript")
    if not isinstance(segments, list) or not segments:
        return f"No earnings call transcript available for {ticker} {quarter}."

    sentiments = []
    for seg in segments:
        if isinstance(seg, dict):
            with contextlib.suppress(TypeError, ValueError):
                sentiments.append(float(seg.get("sentiment")))
    avg_line = ""
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        avg_line = f"Average segment sentiment: {avg:.2f} (Alpha Vantage, higher = more positive)\n"

    lines = [
        f"Earnings Call Transcript — {ticker} {quarter}",
        avg_line.rstrip("\n") if avg_line else "",
        "",
    ]
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker") or "Unknown speaker"
        title = seg.get("title")
        sentiment = seg.get("sentiment")
        header = f"{speaker}" + (f" ({title})" if title else "")
        if sentiment not in (None, ""):
            header += f" [sentiment: {sentiment}]"
        content = (seg.get("content") or "").strip()
        lines.append(header)
        lines.append(content)
        lines.append("")

    return "\n".join(lines).strip()


def get_earnings_call_transcript(ticker: str, curr_date: str = None) -> str:
    """Retrieve the most recent earnings call transcript for a ticker.

    Resolves the latest fiscal quarter reported on or before ``curr_date`` (no
    look-ahead), fetches its transcript from Alpha Vantage, and formats it with
    per-segment sentiment so the fundamentals analyst can explain the reported
    numbers using management's own commentary and guidance.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd

    Returns:
        str: A formatted earnings call transcript report, or a message noting
        that no transcript is available.

    Caching: results are read from / written to a local text-file cache (see
    ``_earnings_cache_path``) so a warm lookup for the same (ticker, date) makes
    zero Alpha Vantage calls. The cache read happens before any API call, so a
    warm cache also sidesteps the daily rate limit.
    """
    try:
        safe = safe_ticker_component(ticker).upper()
    except ValueError:
        safe = None

    # Quarter layer: the "latest quarter reported on or before curr_date" is
    # immutable, so cache it per (ticker, date). A warm hit skips both the
    # OVERVIEW and EARNINGS calls made during resolution.
    quarter = None
    quarter_path = (
        _earnings_cache_path("quarter", f"{safe}-{curr_date}")
        if safe is not None and curr_date
        else None
    )
    if quarter_path is not None:
        cached_quarter = _read_text_cache(quarter_path)
        if cached_quarter is not None and _QUARTER_RE.fullmatch(cached_quarter):
            quarter = cached_quarter

    if quarter is None:
        quarter = _resolve_latest_reported_quarter(ticker, curr_date)
        # Only cache a real resolution; a None result may be a transient failure.
        if quarter and quarter_path is not None:
            _write_text_cache(quarter_path, quarter)

    if not quarter:
        return (
            f"No earnings call transcript available for {ticker} on or before "
            f"{curr_date or 'the current date'}."
        )

    # Transcript layer: a closed quarter's transcript never changes, so cache it
    # per (ticker, quarter). A warm hit skips the premium transcript call.
    transcript_path = (
        _earnings_cache_path("transcripts", f"{safe}-{quarter}") if safe is not None else None
    )
    if transcript_path is not None:
        cached_transcript = _read_text_cache(transcript_path)
        if cached_transcript is not None:
            return cached_transcript

    result = _make_api_request("EARNINGS_CALL_TRANSCRIPT", {"symbol": ticker, "quarter": quarter})
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return f"No earnings call transcript available for {ticker} {quarter}."
    if not isinstance(payload, dict):
        return f"No earnings call transcript available for {ticker} {quarter}."

    formatted = _format_transcript(payload, ticker, quarter)
    # Only cache a genuine transcript, never the "no transcript available"
    # sentinel (mirrors the never-cache-empty guard in stockstats_utils.py).
    if (
        transcript_path is not None
        and isinstance(payload.get("transcript"), list)
        and payload["transcript"]
    ):
        _write_text_cache(transcript_path, formatted)
    return formatted


def _filter_reports_by_date(result, curr_date: str):
    """Drop annual/quarterly reports dated after curr_date to prevent look-ahead.

    ``_make_api_request`` returns the fundamentals payload as a JSON string, so
    parse, filter, and re-serialize. A non-JSON body or an unset ``curr_date`` is
    returned unchanged.
    """
    if not curr_date or not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if isinstance(payload.get(key), list):
            payload[key] = [r for r in payload[key] if r.get("fiscalDateEnding", "") <= curr_date]
    return json.dumps(payload)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)
