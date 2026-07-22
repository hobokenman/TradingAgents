"""FINRA Query API vendor for short sale data.

Two datasets from the FINRA Query API (https://developer.finra.org):

- ``regShoDaily`` — Reg SHO daily short sale volume: aggregate short vs total
  share volume for trades reported to a FINRA facility (TRF/ADF), one row per
  symbol per reporting facility per day. Used to compute the daily Short
  Volume Ratio (SVR).
- ``consolidatedShortInterest`` — bi-monthly consolidated short interest:
  reported short positions per settlement cycle with change and days-to-cover.

Both datasets are public and work without credentials. If
``FINRA_API_CLIENT_ID`` / ``FINRA_API_CLIENT_SECRET`` are set (from the FINRA
API console, https://developer.finra.org), an OAuth2 client-credentials token
is fetched and sent instead, which raises the applicable rate limits.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests

from .errors import NoMarketDataError, VendorError, VendorRateLimitError

logger = logging.getLogger(__name__)

FINRA_API_BASE = "https://api.finra.org"
FINRA_TOKEN_URL = (
    "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
)

# Network timeout (seconds) so a stalled request can't hang the agents,
# mirroring the FRED / Alpha Vantage clients.
REQUEST_TIMEOUT = 30

# Generous row cap for a single-symbol query: a daily-volume window is at most
# ~4 facility rows per trading day and short interest is bi-monthly, so this is
# never the binding constraint in practice.
MAX_RECORDS = 5000

# Rows cap for the rendered tables, mirroring fred.py: recent values matter
# most for a decision, and look_back_days is LLM-supplied, so an oversized
# window must not flood the analyst's context with a huge markdown table.
MAX_ROWS = 40

DEFAULT_SHORT_VOLUME_LOOKBACK_DAYS = 30
# ~12 bi-monthly settlement cycles, enough to read a positioning trend.
DEFAULT_SHORT_INTEREST_LOOKBACK_DAYS = 180

# FINRA disseminates each short-interest cycle roughly two weeks after its
# settlement date. Settlement cycles newer than curr_date minus this lag were
# not yet public on curr_date, so they are excluded to keep historical
# (backtest) dates free of lookahead bias.
SHORT_INTEREST_PUBLICATION_LAG_DAYS = 14


class FinraDataError(VendorError):
    """FINRA returned an unusable response (HTTP error, bad token, bad body)."""


# OAuth token reused across calls within a process; FINRA tokens are
# short-lived, so we refresh shortly before expiry. Keyed by client id so
# rotating credentials in a long-lived process cannot reuse a stale token.
_token_cache: dict = {"client_id": None, "token": None, "expires_at": 0.0}


def _invalidate_token() -> None:
    _token_cache["token"] = None
    _token_cache["expires_at"] = 0.0


def _get_access_token() -> str | None:
    """Return a bearer token when credentials are configured, else None.

    Anonymous access is a supported mode for these public datasets, so missing
    credentials mean "query without auth", not "vendor unavailable".
    """
    client_id = os.getenv("FINRA_API_CLIENT_ID")
    client_secret = os.getenv("FINRA_API_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    now = time.monotonic()
    if (
        _token_cache["token"]
        and _token_cache["client_id"] == client_id
        and now < _token_cache["expires_at"]
    ):
        return _token_cache["token"]

    try:
        response = requests.post(
            FINRA_TOKEN_URL, auth=(client_id, client_secret), timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as e:
        raise FinraDataError(
            f"FINRA OAuth token request failed (check FINRA_API_CLIENT_ID / "
            f"FINRA_API_CLIENT_SECRET): {e}"
        ) from e

    token = payload.get("access_token")
    if not token:
        raise FinraDataError(
            "FINRA OAuth token response did not contain an access_token; "
            "check FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET."
        )
    expires_in = float(payload.get("expires_in", 1800))
    _token_cache["client_id"] = client_id
    _token_cache["token"] = token
    # Refresh a minute early so a token can't expire mid-request.
    _token_cache["expires_at"] = now + max(expires_in - 60, 60)
    return token


def _query_dataset(
    dataset: str,
    symbol_field: str,
    symbol: str,
    date_field: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """POST a filtered query for one symbol/date-window against a FINRA dataset."""
    payload = {
        "limit": MAX_RECORDS,
        "compareFilters": [
            {
                "compareType": "EQUAL",
                "fieldName": symbol_field,
                "fieldValue": symbol,
            }
        ],
        "dateRangeFilters": [
            {"fieldName": date_field, "startDate": start_date, "endDate": end_date}
        ],
    }
    retried = False
    while True:
        headers = {"Accept": "application/json"}
        token = _get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.post(
                f"{FINRA_API_BASE}/data/group/otcMarket/name/{dataset}",
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            raise FinraDataError(f"FINRA request for {dataset} failed: {e}") from e

        # A cached token can be revoked or expire early; refresh it once and
        # retry rather than failing the whole (optional) data pull.
        if token and response.status_code in (401, 403) and not retried:
            _invalidate_token()
            retried = True
            continue
        break

    if response.status_code == 429:
        raise VendorRateLimitError(f"FINRA rate limit hit for {dataset}")
    if response.status_code >= 400:
        raise FinraDataError(
            f"FINRA request for {dataset} failed with HTTP "
            f"{response.status_code}: {response.text[:300]}"
        )
    if not response.text.strip():
        return []
    try:
        rows = response.json()
    except ValueError as e:
        raise FinraDataError(f"FINRA returned non-JSON body for {dataset}") from e
    if not isinstance(rows, list):
        raise FinraDataError(f"FINRA returned unexpected payload shape for {dataset}")
    return rows


def _to_int(value) -> int:
    """FINRA serializes share counts as decimal strings (e.g. '414.000000')."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def get_short_volume(
    symbol: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch FINRA daily short sale volume and Short Volume Ratio as markdown.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        curr_date: End of the window (yyyy-mm-dd); no later rows are returned,
            so a past date never leaks future data.
        look_back_days: Trailing window length; ``None`` uses
            DEFAULT_SHORT_VOLUME_LOOKBACK_DAYS.

    Returns:
        A markdown report with a per-day table (short volume, total volume,
        SVR) and summary statistics, or raises ``NoMarketDataError`` when the
        symbol has no rows in the window.
    """
    if look_back_days is None:
        look_back_days = DEFAULT_SHORT_VOLUME_LOOKBACK_DAYS
    sym = symbol.strip().upper()
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    rows = _query_dataset(
        "regShoDaily",
        "securitiesInformationProcessorSymbolIdentifier",
        sym,
        "tradeReportDate",
        start_date,
        curr_date,
    )
    if not rows:
        raise NoMarketDataError(
            sym,
            detail=(
                f"no FINRA daily short sale volume between {start_date} and "
                f"{curr_date}; the symbol may not be a US-listed equity"
            ),
        )

    # One row per reporting facility (NYSE TRF, NASDAQ TRFs, ADF) per day —
    # sum them into a single daily record before computing ratios.
    daily: dict[str, dict[str, int]] = {}
    for row in rows:
        date = row.get("tradeReportDate")
        if not date:
            continue
        agg = daily.setdefault(date, {"short": 0, "exempt": 0, "total": 0})
        agg["short"] += _to_int(row.get("shortParQuantity"))
        agg["exempt"] += _to_int(row.get("shortExemptParQuantity"))
        agg["total"] += _to_int(row.get("totalParQuantity"))

    # One record per day with reported volume: (date, agg, svr). The summary
    # statistics, the table, and the header day count all derive from this one
    # list so they can never disagree (zero-total days are dropped everywhere).
    points = [
        (date, agg, agg["short"] / agg["total"])
        for date, agg in sorted(daily.items())
        if agg["total"] > 0
    ]
    if not points:
        raise NoMarketDataError(
            sym, detail=f"FINRA reported zero total volume between {start_date} and {curr_date}"
        )

    ratios = [svr for _, _, svr in points]
    latest_date, _, latest_svr = points[-1]
    mean_svr = sum(ratios) / len(ratios)
    recent = ratios[-5:]
    recent_mean = sum(recent) / len(recent)
    if recent_mean > mean_svr * 1.05:
        trend = "rising vs the period baseline"
    elif recent_mean < mean_svr * 0.95:
        trend = "falling vs the period baseline"
    else:
        trend = "in line with the period baseline"

    header = (
        f"## FINRA Daily Short Sale Volume: {sym}\n"
        f"- Window: {start_date} to {curr_date} ({len(points)} trading days)\n"
        f"- Source: FINRA Reg SHO daily files (trades reported to FINRA "
        f"facilities — off-exchange/TRF volume, not consolidated exchange volume)\n"
    )
    summary = (
        f"\n**Latest SVR:** {latest_svr:.1%} ({latest_date}) | "
        f"**Period mean:** {mean_svr:.1%} "
        f"(min {min(ratios):.1%}, max {max(ratios):.1%}) | "
        f"**5-day mean:** {recent_mean:.1%} — {trend}\n"
    )
    shown = points
    truncation_note = ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        truncation_note = (
            f"\n_(showing the most recent {MAX_ROWS} of {len(points)} trading days; "
            f"summary statistics cover the full window)_\n"
        )
    table = (
        "\n| Date | Short Volume | Short Exempt | Total Volume | Short Volume Ratio |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {date} | {agg['short']:,} | {agg['exempt']:,} | {agg['total']:,} | {svr:.1%} |"
            for date, agg, svr in shown
        )
        + "\n"
    )
    note = (
        "\n_Interpretation: SVR ~35-50% is a normal baseline for most liquid "
        "equities because market-maker liquidity provision is reported as short "
        "sales. Read the level relative to this symbol's own recent baseline: a "
        "sustained rise suggests growing short-side pressure; an elevated SVR "
        "into weakness can also precede short-covering bounces._\n"
    )
    return header + summary + truncation_note + table + note


def get_short_interest(
    symbol: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch FINRA bi-monthly consolidated short interest as markdown.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        curr_date: "Now" for the analysis (yyyy-mm-dd). Only cycles that were
            already publicly disseminated by this date are returned — i.e.
            settlement dates at least SHORT_INTEREST_PUBLICATION_LAG_DAYS
            earlier — so a past date never leaks then-unpublished data.
        look_back_days: Trailing window length; ``None`` uses
            DEFAULT_SHORT_INTEREST_LOOKBACK_DAYS (~12 bi-monthly cycles).

    Returns:
        A markdown report with a per-cycle table (short position, change %,
        days-to-cover) and a latest-cycle summary, or raises
        ``NoMarketDataError`` when the symbol has no cycles in the window.
    """
    if look_back_days is None:
        look_back_days = DEFAULT_SHORT_INTEREST_LOOKBACK_DAYS
    sym = symbol.strip().upper()
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # End the settlement-date window at the dissemination cutoff, not at
    # curr_date: a cycle settling within the publication lag was not yet
    # public on curr_date, and including it would leak future data into
    # historical (backtest) runs.
    end_date = (end_dt - timedelta(days=SHORT_INTEREST_PUBLICATION_LAG_DAYS)).strftime("%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    rows = _query_dataset(
        "consolidatedShortInterest",
        "symbolCode",
        sym,
        "settlementDate",
        start_date,
        end_date,
    )
    cycles = sorted(
        (row for row in rows if row.get("settlementDate")),
        key=lambda row: row["settlementDate"],
    )
    if not cycles:
        raise NoMarketDataError(
            sym,
            detail=(
                f"no FINRA short interest cycles settling between {start_date} "
                f"and {end_date} (cycles settling after {end_date} were not yet "
                f"published on {curr_date}); the symbol may not be a US-listed equity"
            ),
        )

    latest = cycles[-1]
    latest_position = _to_int(latest.get("currentShortPositionQuantity"))
    first_position = _to_int(cycles[0].get("currentShortPositionQuantity"))
    if len(cycles) < 2:
        window_change_text = "n/a (single cycle in window)"
    elif first_position > 0:
        window_change = (latest_position - first_position) / first_position
        window_change_text = f"{window_change:+.1%} across the window"
    else:
        window_change_text = "n/a across the window"

    def _fmt_change(row) -> str:
        value = row.get("changePercent")
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return "n/a"

    def _fmt_days(row) -> str:
        try:
            return f"{float(row.get('daysToCoverQuantity')):.1f}"
        except (TypeError, ValueError):
            return "n/a"

    issue_name = latest.get("issueName") or sym
    header = (
        f"## FINRA Consolidated Short Interest: {sym} ({issue_name})\n"
        f"- Window: settlements {start_date} to {end_date} ({len(cycles)} bi-monthly "
        f"cycles published as of {curr_date})\n"
        f"- Source: FINRA consolidated short interest (Rule 4560 member firm "
        f"reports)\n"
    )
    summary = (
        f"\n**Latest cycle ({latest.get('settlementDate')}):** "
        f"{latest_position:,} shares short | "
        f"**Change vs prior cycle:** {_fmt_change(latest)} | "
        f"**Days to cover:** {_fmt_days(latest)} | "
        f"**Trend:** {window_change_text}\n"
    )
    shown = cycles
    truncation_note = ""
    if len(cycles) > MAX_ROWS:
        shown = cycles[-MAX_ROWS:]
        truncation_note = (
            f"\n_(showing the most recent {MAX_ROWS} of {len(cycles)} cycles; "
            f"the window trend covers the full range)_\n"
        )
    table = (
        "\n| Settlement Date | Short Position | Change vs Prior | Days to Cover "
        "| Avg Daily Volume |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {row.get('settlementDate')} | "
            f"{_to_int(row.get('currentShortPositionQuantity')):,} | "
            f"{_fmt_change(row)} | {_fmt_days(row)} | "
            f"{_to_int(row.get('averageDailyVolumeQuantity')):,} |"
            for row in shown
        )
        + "\n"
    )
    note = (
        "\n_Interpretation: short interest settles bi-monthly and FINRA "
        "publishes each cycle roughly two weeks after its settlement date; "
        "cycles not yet published as of the analysis date are excluded, so "
        "the latest cycle shown may lag current positioning by a few weeks. "
        "Rising short interest with high days-to-cover signals crowded bearish "
        "conviction — a headwind for price, but also fuel for a short squeeze "
        "if the narrative turns._\n"
    )
    return header + summary + truncation_note + table + note
