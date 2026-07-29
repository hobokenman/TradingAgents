"""Earnings call transcript source for the fundamentals analyst.

Covers quarter resolution (latest reported quarter, no look-ahead past the
trading date), transcript formatting with per-segment sentiment, and the
optional-category routing degradation when Alpha Vantage is unavailable.
"""

import copy
import json
from unittest import mock

import pytest

import tradingagents.dataflows.alpha_vantage_fundamentals as avf
import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the earnings-call file cache to a tmp dir for every test.

    Keeps cache reads/writes out of the real ``~/.tradingagents/cache`` and gives
    each test a clean, empty cache so API-mocking behaves deterministically.
    """
    monkeypatch.setattr(avf, "get_config", lambda: {"data_cache_dir": str(tmp_path)})


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


_EARNINGS_JSON = json.dumps(
    {
        "symbol": "AAPL",
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2024-06-30", "reportedDate": "2024-08-01"},  # future
            {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-05-02"},  # latest <= curr_date
            {"fiscalDateEnding": "2023-12-31", "reportedDate": "2024-02-01"},
        ],
    }
)


@pytest.mark.unit
def test_fiscal_date_to_quarter_calendar_year():
    # Default December fiscal-year end -> plain calendar quarters.
    assert avf._fiscal_date_to_quarter("2024-01-31") == "2024Q1"
    assert avf._fiscal_date_to_quarter("2024-06-30") == "2024Q2"
    assert avf._fiscal_date_to_quarter("2024-09-30") == "2024Q3"
    assert avf._fiscal_date_to_quarter("2024-12-31") == "2024Q4"
    assert avf._fiscal_date_to_quarter("garbage") is None


@pytest.mark.unit
def test_fiscal_date_to_quarter_non_calendar_year():
    # Apple's September fiscal-year end: AV labels the Dec-2024 quarter 2025Q1.
    assert avf._fiscal_date_to_quarter("2024-12-31", fye_month=9) == "2025Q1"
    assert avf._fiscal_date_to_quarter("2025-03-31", fye_month=9) == "2025Q2"
    assert avf._fiscal_date_to_quarter("2025-06-30", fye_month=9) == "2025Q3"
    assert avf._fiscal_date_to_quarter("2025-09-30", fye_month=9) == "2025Q4"


def _earnings_or_overview(fye_name):
    """Fake _make_api_request: OVERVIEW returns the fiscal-year end, else EARNINGS."""

    def fake(fn, params):
        if fn == "OVERVIEW":
            return json.dumps({"FiscalYearEnd": fye_name})
        return _EARNINGS_JSON

    return fake


@pytest.mark.unit
def test_resolve_quarter_skips_unreported_future_quarter(monkeypatch):
    # The most recent quarter whose call had already happened as of curr_date.
    # Calendar-year company (December FYE): 2024-03-31 -> 2024Q1.
    monkeypatch.setattr(avf, "_make_api_request", _earnings_or_overview("December"))
    assert avf._resolve_latest_reported_quarter("AAPL", "2024-06-01") == "2024Q1"


@pytest.mark.unit
def test_resolve_quarter_uses_fiscal_year_end(monkeypatch):
    # September FYE shifts the label: 2024-03-31 is fiscal 2024Q2.
    monkeypatch.setattr(avf, "_make_api_request", _earnings_or_overview("September"))
    assert avf._resolve_latest_reported_quarter("AAPL", "2024-06-01") == "2024Q2"


@pytest.mark.unit
def test_resolve_quarter_no_curr_date_takes_newest(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", _earnings_or_overview("December"))
    assert avf._resolve_latest_reported_quarter("AAPL", None) == "2024Q2"


@pytest.mark.unit
def test_resolve_quarter_unusable_payload(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    assert avf._resolve_latest_reported_quarter("AAPL", "2024-06-01") is None


@pytest.mark.unit
def test_resolve_quarter_skips_entry_with_missing_reported_date(monkeypatch):
    # Look-ahead guard: a newest-first entry whose reportedDate is null/missing
    # cannot be confirmed as reported <= curr_date, so it must be skipped (not
    # returned) to avoid leaking a not-yet-reported quarter into a backtest.
    payload = json.dumps(
        {
            "quarterlyEarnings": [
                {"fiscalDateEnding": "2024-06-30", "reportedDate": None},  # null -> skip
                {"fiscalDateEnding": "2024-03-31", "reportedDate": "2024-05-02"},  # -> 2024Q1
            ]
        }
    )

    def fake(fn, params):
        return json.dumps({"FiscalYearEnd": "December"}) if fn == "OVERVIEW" else payload

    monkeypatch.setattr(avf, "_make_api_request", fake)
    assert avf._resolve_latest_reported_quarter("AAPL", "2024-06-01") == "2024Q1"


@pytest.mark.unit
def test_resolve_quarter_no_curr_date_allows_missing_reported_date(monkeypatch):
    # Without curr_date there is no look-ahead concern; the newest entry is used
    # even if its reportedDate is absent.
    payload = json.dumps({"quarterlyEarnings": [{"fiscalDateEnding": "2024-06-30"}]})

    def fake(fn, params):
        return json.dumps({"FiscalYearEnd": "December"}) if fn == "OVERVIEW" else payload

    monkeypatch.setattr(avf, "_make_api_request", fake)
    assert avf._resolve_latest_reported_quarter("AAPL", None) == "2024Q2"


@pytest.mark.unit
def test_get_transcript_formats_with_sentiment(monkeypatch):
    transcript = json.dumps(
        {
            "symbol": "AAPL",
            "quarter": "2024Q1",
            "transcript": [
                {
                    "speaker": "Tim Cook",
                    "title": "CEO",
                    "content": "Revenue grew.",
                    "sentiment": "0.8",
                },
                {
                    "speaker": "Analyst",
                    "title": "",
                    "content": "Margin outlook?",
                    "sentiment": "0.1",
                },
            ],
        }
    )

    def fake_request(fn, params):
        return _EARNINGS_JSON if fn == "EARNINGS" else transcript

    monkeypatch.setattr(avf, "_make_api_request", fake_request)
    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "AAPL 2024Q1" in out
    assert "Tim Cook (CEO)" in out
    assert "Revenue grew." in out
    assert "sentiment: 0.8" in out
    assert "Average segment sentiment: 0.45" in out


@pytest.mark.unit
def test_get_transcript_none_available(monkeypatch):
    def fake_request(fn, params):
        return _EARNINGS_JSON if fn == "EARNINGS" else json.dumps({"transcript": []})

    monkeypatch.setattr(avf, "_make_api_request", fake_request)
    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "No earnings call transcript available" in out


@pytest.mark.unit
def test_get_transcript_no_quarter_resolved(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "No earnings call transcript available" in out


@pytest.mark.unit
def test_routing_degrades_when_vendor_unavailable():
    # earnings_transcripts is optional: an Alpha Vantage failure must not abort
    # the fundamentals node — the router returns a sentinel instead.
    _reset_config()
    try:
        set_config({"data_vendors": {"earnings_transcripts": "alpha_vantage"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_earnings_call_transcript": {"alpha_vantage": _raise_boom}},
            clear=False,
        ):
            result = interface.route_to_vendor("get_earnings_call_transcript", "AAPL", "2024-06-01")
        assert "DATA_UNAVAILABLE" in result
        assert "earnings_transcripts" in result
    finally:
        _reset_config()


def _raise_boom(*a, **k):
    raise ValueError("AV down")


# --- Local file cache -------------------------------------------------------

_TRANSCRIPT_JSON = json.dumps(
    {
        "symbol": "AAPL",
        "quarter": "2024Q1",
        "transcript": [
            {"speaker": "Tim Cook", "title": "CEO", "content": "Revenue grew.", "sentiment": "0.8"},
        ],
    }
)


def _cache_file(tmp_path, kind, name):
    return tmp_path / "earnings_call" / kind / f"{name}.txt"


def _seed(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _recording_request(handlers):
    """Fake _make_api_request that records requested function names and dispatches.

    ``handlers`` maps an AV function name to the string body to return. A request
    for a function absent from ``handlers`` raises (so tests can assert an
    endpoint is never hit).
    """
    calls = []

    def fake(fn, params):
        calls.append(fn)
        if fn not in handlers:
            raise AssertionError(f"unexpected API call: {fn}")
        return handlers[fn]

    fake.calls = calls
    return fake


@pytest.mark.unit
def test_warm_cache_makes_zero_api_calls(tmp_path, monkeypatch):
    # Both the resolved quarter and the transcript are cached -> no API at all.
    _seed(_cache_file(tmp_path, "quarter", "AAPL-2024-06-01"), "2024Q1")
    _seed(_cache_file(tmp_path, "transcripts", "AAPL-2024Q1"), "CACHED TRANSCRIPT TEXT")

    def boom(fn, params):
        raise AssertionError(f"no API call expected, got {fn}")

    monkeypatch.setattr(avf, "_make_api_request", boom)
    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert out == "CACHED TRANSCRIPT TEXT"


@pytest.mark.unit
def test_transcript_cache_hit_skips_premium_call(tmp_path, monkeypatch):
    # Transcript cached but quarter not: resolution runs, but the premium
    # EARNINGS_CALL_TRANSCRIPT call must be skipped.
    _seed(_cache_file(tmp_path, "transcripts", "AAPL-2024Q1"), "CACHED TRANSCRIPT TEXT")
    fake = _recording_request(
        {"OVERVIEW": json.dumps({"FiscalYearEnd": "December"}), "EARNINGS": _EARNINGS_JSON}
    )
    monkeypatch.setattr(avf, "_make_api_request", fake)

    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert out == "CACHED TRANSCRIPT TEXT"
    assert "EARNINGS_CALL_TRANSCRIPT" not in fake.calls


@pytest.mark.unit
def test_cold_cache_writes_all_three_layers(tmp_path, monkeypatch):
    fake = _recording_request(
        {
            "OVERVIEW": json.dumps({"FiscalYearEnd": "December"}),
            "EARNINGS": _EARNINGS_JSON,
            "EARNINGS_CALL_TRANSCRIPT": _TRANSCRIPT_JSON,
        }
    )
    monkeypatch.setattr(avf, "_make_api_request", fake)

    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "Revenue grew." in out

    # All three cache layers were written.
    assert _cache_file(tmp_path, "fiscal_year_end", "AAPL").read_text().strip() == "12"
    assert _cache_file(tmp_path, "quarter", "AAPL-2024-06-01").read_text().strip() == "2024Q1"
    transcript_file = _cache_file(tmp_path, "transcripts", "AAPL-2024Q1")
    assert transcript_file.exists() and transcript_file.read_text().strip()

    # A second call is served entirely from cache — no further API calls.
    fake.calls.clear()
    out2 = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert out2 == out
    assert fake.calls == []


@pytest.mark.unit
def test_empty_transcript_not_cached(tmp_path, monkeypatch):
    fake = _recording_request(
        {
            "OVERVIEW": json.dumps({"FiscalYearEnd": "December"}),
            "EARNINGS": _EARNINGS_JSON,
            "EARNINGS_CALL_TRANSCRIPT": json.dumps({"transcript": []}),
        }
    )
    monkeypatch.setattr(avf, "_make_api_request", fake)

    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "No earnings call transcript available" in out
    # The sentinel is never persisted, but the resolved quarter still is.
    assert not _cache_file(tmp_path, "transcripts", "AAPL-2024Q1").exists()
    assert _cache_file(tmp_path, "quarter", "AAPL-2024-06-01").exists()


@pytest.mark.unit
def test_unresolvable_quarter_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    out = avf.get_earnings_call_transcript("AAPL", curr_date="2024-06-01")
    assert "No earnings call transcript available" in out
    assert not _cache_file(tmp_path, "quarter", "AAPL-2024-06-01").exists()


@pytest.mark.unit
def test_fiscal_year_end_cache_hit_skips_overview(tmp_path, monkeypatch):
    _seed(_cache_file(tmp_path, "fiscal_year_end", "AAPL"), "9")

    def boom(fn, params):
        raise AssertionError(f"no API call expected, got {fn}")

    monkeypatch.setattr(avf, "_make_api_request", boom)
    assert avf._fiscal_year_end_month("AAPL") == 9
