"""SEC EDGAR 10-K/10-Q/8-K filing source for the fundamentals analyst.

Covers CIK resolution, filing selection (latest form, no look-ahead past the
trading date, amendments excluded), full-document HTML-to-text extraction and
local caching, bounded index/search/pagination access, the 8-K event digest,
EDGAR fair-access behavior, and optional-category routing degradation.
"""

import copy
import json
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.dataflows.sec_edgar as se
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorRateLimitError


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the SEC-filings file cache to a tmp dir for every test.

    Also resets the in-process ticker->CIK memo so a map parsed in one test can't
    satisfy another test that expects a fresh download.
    """
    monkeypatch.setattr(se, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(se, "_TICKER_MAP", None)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


_TICKERS_JSON = json.dumps(
    {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway Inc"},
    }
)

_SUBMISSIONS_JSON = json.dumps(
    {
        "cik": 320193,
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "8-K", "10-K/A", "10-Q", "8-K", "10-K"],
                "filingDate": [
                    "2024-08-02",
                    "2024-07-15",
                    "2024-05-28",
                    "2024-05-20",
                    "2024-05-03",
                    "2024-02-15",
                    "2023-11-03",
                ],
                "accessionNumber": [
                    "0000320193-24-000081",
                    "0000320193-24-000070",
                    "0000320193-24-000065",
                    "0000320193-24-000060",
                    "0000320193-24-000050",
                    "0000320193-24-000020",
                    "0000320193-23-000106",
                ],
                "primaryDocument": [
                    "q3-2024.htm",
                    "future8k.htm",
                    "event8k.htm",
                    "amend.htm",
                    "q2-2024.htm",
                    "stale8k.htm",
                    "annual-10k.htm",
                ],
                "reportDate": [
                    "2024-06-29",
                    "2024-07-15",
                    "2024-05-28",
                    "2023-09-30",
                    "2024-03-30",
                    "",
                    "2023-09-30",
                ],
                "items": ["", "1.01,9.01", "2.02,9.01", "", "", "8.01", ""],
            }
        },
    }
)

_TENK_HTML = """<html><head><title>10-K</title></head>
<body>
<ix:header><p>HIDDEN-XBRL-METADATA</p></ix:header>
<script>var secret = "SCRIPT-CONTENT";</script>
<div>
<p>TABLE OF CONTENTS</p>
<p>Item 1. Business 5</p>
<p>Item 1A. Risk Factors 12</p>
<p>Item 3. Legal Proceedings 30</p>
<p>Item 7. Management's Discussion and Analysis 35</p>
<p>Item 8. Financial Statements and Supplementary Data 60</p>
</div>
<p>PART I</p>
<p>Item 1. Business</p>
<p>We design and sell widgets worldwide to industrial customers.</p>
<p>Item 1A. Risk Factors</p>
<p>RISK-BODY: Competition may reduce widget margins materially over time.</p>
<p>Item 3. Legal Proceedings</p>
<p>LEGAL-BODY: A patent infringement suit is pending in Delaware federal court.</p>
<p>PART II</p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
<p>MDNA-BODY: Revenue grew nine percent on higher widget volume and pricing.</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>FIN-BODY: Consolidated balance sheets and the accompanying notes follow.</p>
</body></html>"""

_EIGHTK_HTML = """<html><body>
<p>Item 2.02 Results of Operations and Financial Condition</p>
<p>EVENT-BODY: The company announced quarterly results and furnished a press release.</p>
</body></html>"""

_TENQ_HTML = """<html><body>
<p>PART I — FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p>
<p>Q-FIN-BODY: Unaudited condensed consolidated statements are included herein.</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>Q-MDNA-BODY: Sequential revenue improved on services strength this quarter.</p>
<p>PART II — OTHER INFORMATION</p>
<p>Item 1. Legal Proceedings</p>
<p>Q-LEGAL-BODY: No material developments occurred in pending litigation.</p>
<p>Item 1A. Risk Factors</p>
<p>Q-RISK-BODY: There have been material changes to previously disclosed risks.</p>
</body></html>"""


def _dispatching_get(handlers):
    """Fake ``_edgar_get`` that dispatches on URL substring and records calls.

    ``handlers`` maps a URL fragment to the body to return. A URL matching no
    fragment raises, so tests can assert an endpoint is never hit.
    """
    calls = []

    def fake(url):
        calls.append(url)
        for fragment, body in handlers.items():
            if fragment in url:
                return body
        raise AssertionError(f"unexpected EDGAR URL: {url}")

    fake.calls = calls
    return fake


_DEFAULT_HANDLERS = {
    "company_tickers.json": _TICKERS_JSON,
    "/submissions/": _SUBMISSIONS_JSON,
    "q2-2024.htm": _TENQ_HTML,
    "annual-10k.htm": _TENK_HTML,
    "event8k.htm": _EIGHTK_HTML,
}


def _cache_file(tmp_path, kind, name):
    return tmp_path / "sec_filings" / kind / f"{name}.txt"


def _seed(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _raise_boom(*a, **k):
    raise ValueError("EDGAR down")


# --- CIK resolution ---------------------------------------------------------


@pytest.mark.unit
def test_resolve_cik_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    assert se._resolve_cik("AAPL") == "320193"
    assert _cache_file(tmp_path, "cik", "AAPL").read_text().strip() == "320193"

    # Warm hit: no HTTP at all.
    monkeypatch.setattr(se, "_edgar_get", _raise_boom)
    assert se._resolve_cik("AAPL") == "320193"


@pytest.mark.unit
def test_resolve_cik_dot_class_ticker(monkeypatch):
    # EDGAR lists class shares with a dash (BRK-B); the dot form must resolve.
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    assert se._resolve_cik("BRK.B") == "1067983"


@pytest.mark.unit
def test_unknown_ticker_returns_message_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    out = se.get_sec_filing("ZZZZ", curr_date="2024-06-01")
    assert "No SEC filings found" in out
    # A miss may be transient (bad download) and must not poison the cache.
    assert not _cache_file(tmp_path, "cik", "ZZZZ").exists()


@pytest.mark.unit
def test_company_list_downloaded_once_across_tickers(monkeypatch):
    # The ~1MB company list is identical for every ticker, so the in-process memo
    # must download it at most once even when resolving several fresh tickers.
    fake = _dispatching_get(_DEFAULT_HANDLERS)
    monkeypatch.setattr(se, "_edgar_get", fake)
    assert se._resolve_cik("AAPL") == "320193"
    assert se._resolve_cik("BRK.B") == "1067983"
    downloads = [u for u in fake.calls if "company_tickers.json" in u]
    assert len(downloads) == 1


# --- Filing selection -------------------------------------------------------


@pytest.mark.unit
def test_pick_filing_no_look_ahead():
    # The 2024-08-02 10-Q postdates the trading date and must not be selected.
    meta = se._pick_filing(json.loads(_SUBMISSIONS_JSON), "latest", "2024-06-01")
    assert meta["form"] == "10-Q"
    assert meta["filing_date"] == "2024-05-03"


@pytest.mark.unit
def test_pick_filing_skips_amendments():
    # The 10-K/A filed 2024-05-20 is newer but must never satisfy "10-K".
    meta = se._pick_filing(json.loads(_SUBMISSIONS_JSON), "10-K", "2024-06-01")
    assert meta["form"] == "10-K"
    assert meta["filing_date"] == "2023-11-03"


@pytest.mark.unit
def test_pick_filing_explicit_form():
    submissions = json.loads(_SUBMISSIONS_JSON)
    assert se._pick_filing(submissions, "10-Q", "2024-06-01")["filing_date"] == "2024-05-03"
    assert se._pick_filing(submissions, "10-K", "2024-06-01")["filing_date"] == "2023-11-03"


@pytest.mark.unit
def test_pick_filing_tolerates_missing_report_date():
    # reportDate is only a display field; a payload without it must still resolve
    # a real filing rather than falsely reporting none found.
    submissions = json.loads(_SUBMISSIONS_JSON)
    del submissions["filings"]["recent"]["reportDate"]
    meta = se._pick_filing(submissions, "10-K", "2024-06-01")
    assert meta is not None
    assert meta["form"] == "10-K"
    assert meta["report_date"] == ""


@pytest.mark.unit
def test_pick_filing_no_curr_date_takes_newest():
    meta = se._pick_filing(json.loads(_SUBMISSIONS_JSON), "latest", None)
    assert meta["filing_date"] == "2024-08-02"
    assert meta["form"] == "10-Q"


# --- Extraction and bounded retrieval ---------------------------------------


@pytest.mark.unit
def test_get_returns_bounded_index_not_full_body(monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    out = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="10-K")
    assert out.startswith("SEC Filing — AAPL 10-K — bounded index")
    assert "complete text cached locally" in out
    assert "Use search_sec_filing" in out
    assert "Item 7. Management's Discussion" in out
    assert "RISK-BODY" not in out
    assert "LEGAL-BODY" not in out
    assert len(out) <= se.MAX_TOOL_OUTPUT_CHARS


@pytest.mark.unit
def test_search_returns_exact_excerpts_and_offsets(monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    out = se.search_sec_filing(
        "AAPL",
        "2024-06-01",
        ["RISK-BODY", "LEGAL-BODY", "not-in-this-filing"],
        form_type="10-K",
        context_chars=80,
    )
    assert "RISK-BODY" in out
    assert "LEGAL-BODY" in out
    assert "Match offset:" in out
    assert "Excerpt range:" in out
    assert "Recommended read offset:" in out
    assert "No literal matches for: not-in-this-filing" in out
    assert len(out) <= se.MAX_TOOL_OUTPUT_CHARS


@pytest.mark.unit
def test_read_pagination_is_lossless(monkeypatch):
    document = "SEC Filing — TEST 10-Q\n" + "".join(f"line-{i:03d}\n" for i in range(80))
    monkeypatch.setattr(se, "_get_filing_document", lambda *args: (document, True))

    chunks = []
    offset = 0
    while offset < len(document):
        out = se.read_sec_filing("TEST", "2024-06-01", offset=offset, max_chars=73)
        chunk = out.split("--- BEGIN EXACT FILING TEXT ---\n", 1)[1].split(
            "\n--- END EXACT FILING TEXT ---", 1
        )[0]
        chunks.append(chunk)
        offset += len(chunk)

    assert "".join(chunks) == document


@pytest.mark.unit
def test_all_public_document_responses_are_bounded(monkeypatch):
    document = "SEC Filing — TEST 10-Q\n" + ("MATERIAL LEGAL PROCEEDINGS\n" * 30_000)
    monkeypatch.setattr(se, "_get_filing_document", lambda *args: (document, True))

    index = se.get_sec_filing("TEST", "2024-06-01")
    search = se.search_sec_filing(
        "TEST",
        "2024-06-01",
        ["material", "legal", "proceedings"],
        max_matches=3,
        context_chars=800,
    )
    page = se.read_sec_filing("TEST", "2024-06-01", max_chars=999_999)
    assert all(len(output) <= se.MAX_TOOL_OUTPUT_CHARS for output in (index, search, page))


@pytest.mark.unit
def test_hidden_content_stripped():
    text = se._html_to_text(_TENK_HTML)
    assert "HIDDEN-XBRL-METADATA" not in text
    assert "SCRIPT-CONTENT" not in text


@pytest.mark.unit
def test_table_cells_are_separated():
    # Financial-statement cells must not glue together (Revenue1,000900) — that
    # would corrupt the numbers. Each cell is separated on the same line.
    html = (
        "<table><tr><td>Revenue</td><td>1,000</td><td>900</td></tr>"
        "<tr><td>Cost</td><td>400</td><td>350</td></tr></table>"
    )
    text = se._html_to_text(html)
    assert "Revenue 1,000 900" in text
    assert "Cost 400 350" in text
    assert "Revenue1,000" not in text
    assert "1,000900" not in text


# --- Local file cache -------------------------------------------------------


@pytest.mark.unit
def test_warm_cache_makes_zero_http_calls(tmp_path, monkeypatch):
    _seed(_cache_file(tmp_path, "cik", "AAPL"), "320193")
    _seed(
        _cache_file(tmp_path, "selection", "AAPL-latest-2024-06-01"),
        json.dumps(
            {
                "form": "10-Q",
                "filing_date": "2024-05-03",
                "report_date": "2024-03-30",
                "accession": "0000320193-24-000050",
                "primary_doc": "q2-2024.htm",
            }
        ),
    )
    _seed(_cache_file(tmp_path, "filings", "AAPL-000032019324000050"), "CACHED FILING TEXT")

    monkeypatch.setattr(se, "_edgar_get", _raise_boom)
    index = se.get_sec_filing("AAPL", curr_date="2024-06-01")
    page = se.read_sec_filing("AAPL", curr_date="2024-06-01")
    assert "bounded index" in index
    assert "CACHED FILING TEXT" in page


@pytest.mark.unit
def test_cold_cache_writes_all_layers_and_second_call_is_free(tmp_path, monkeypatch):
    fake = _dispatching_get(_DEFAULT_HANDLERS)
    monkeypatch.setattr(se, "_edgar_get", fake)

    index = se.get_sec_filing("AAPL", curr_date="2024-06-01")
    assert "bounded index" in index
    assert "Q-MDNA-BODY" not in index
    assert _cache_file(tmp_path, "cik", "AAPL").exists()
    assert _cache_file(tmp_path, "selection", "AAPL-latest-2024-06-01").exists()
    assert _cache_file(tmp_path, "filings", "AAPL-000032019324000050").exists()

    # Index, search, and read are now served entirely from cache.
    fake.calls.clear()
    search = se.search_sec_filing("AAPL", "2024-06-01", ["Q-MDNA-BODY"])
    page = se.read_sec_filing("AAPL", "2024-06-01")
    assert "Q-MDNA-BODY" in search
    assert "Q-MDNA-BODY" in page
    assert fake.calls == []


@pytest.mark.unit
def test_no_filing_resolution_not_cached(tmp_path, monkeypatch):
    only_8k = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2024-02-15"],
                    "accessionNumber": ["0000320193-24-000020"],
                    "primaryDocument": ["eightk.htm"],
                    "reportDate": [""],
                }
            }
        }
    )
    handlers = {"company_tickers.json": _TICKERS_JSON, "/submissions/": only_8k}
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(handlers))
    out = se.get_sec_filing("AAPL", curr_date="2024-06-01")
    assert "No 10-K or 10-Q found" in out
    assert not _cache_file(tmp_path, "selection", "AAPL-latest-2024-06-01").exists()


@pytest.mark.unit
def test_unsupported_form_type_returns_message(monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _raise_boom)
    out = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="S-1")
    assert "Unsupported form_type" in out


# --- 8-K event digest -------------------------------------------------------


@pytest.mark.unit
def test_8k_digest_window_and_no_look_ahead(monkeypatch):
    fake = _dispatching_get(_DEFAULT_HANDLERS)
    monkeypatch.setattr(se, "_edgar_get", fake)
    index = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="8-K")
    assert "8-K digest" in index
    # The 2024-05-28 8-K is in the 90-day window and its index items are shown.
    assert "8-K filed 2024-05-28" in index
    assert "Items: 2.02,9.01" in index
    assert "EVENT-BODY" not in index
    # The 2024-07-15 8-K postdates the trading date (no look-ahead) and the
    # 2024-02-15 one is older than the lookback window.
    assert "2024-07-15" not in index
    assert "2024-02-15" not in index

    # Search and a second index call are served entirely from cache.
    fake.calls.clear()
    search = se.search_sec_filing("AAPL", "2024-06-01", ["EVENT-BODY"], form_type="8-K")
    index2 = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="8-K")
    assert "EVENT-BODY" in search
    assert index2 == index
    assert fake.calls == []


@pytest.mark.unit
def test_8k_none_in_window_returns_message(tmp_path, monkeypatch):
    # As of 2023-12-01 no 8-K has been filed yet at all.
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    out = se.get_sec_filing("AAPL", curr_date="2023-12-01", form_type="8-K")
    assert "No 8-K filed" in out
    # An empty resolution is never cached (may be a transient payload problem).
    assert not _cache_file(tmp_path, "selection", "AAPL-8-K-2023-12-01").exists()


@pytest.mark.unit
def test_8k_alias_accepted(monkeypatch):
    monkeypatch.setattr(se, "_edgar_get", _dispatching_get(_DEFAULT_HANDLERS))
    out = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="8k")
    assert "8-K digest" in out


_TWO_8K_SUBS = json.dumps(
    {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K"],
                "filingDate": ["2024-05-28", "2024-05-20"],
                "accessionNumber": ["0000320193-24-000065", "0000320193-24-000064"],
                "primaryDocument": ["good8k.htm", "bad8k.htm"],
                "reportDate": ["2024-05-28", "2024-05-20"],
                "items": ["2.02,9.01", "5.02"],
            }
        }
    }
)


@pytest.mark.unit
def test_8k_digest_survives_one_fetch_error(monkeypatch):
    # One 8-K document erroring must not discard the others already selected.
    def fake(url):
        if "company_tickers.json" in url:
            return _TICKERS_JSON
        if "/submissions/" in url:
            return _TWO_8K_SUBS
        if "good8k.htm" in url:
            return "<html><body><p>GOOD-EVENT-BODY present.</p></body></html>"
        if "bad8k.htm" in url:
            raise VendorRateLimitError("throttled")
        raise AssertionError(f"unexpected EDGAR URL: {url}")

    monkeypatch.setattr(se, "_edgar_get", fake)
    index = se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="8-K")
    search = se.search_sec_filing(
        "AAPL",
        "2024-06-01",
        ["GOOD-EVENT-BODY", "Could not retrieve"],
        form_type="8-K",
    )
    assert "8-K filed 2024-05-28" in index
    assert "8-K filed 2024-05-20" in index
    assert "GOOD-EVENT-BODY present." in search
    assert "Could not retrieve this filing's text." in search


@pytest.mark.unit
def test_8k_digest_all_fetches_fail_raises(monkeypatch):
    # If every document fetch fails, propagate so the router degrades the whole
    # optional category rather than returning an all-empty digest.
    def fake(url):
        if "company_tickers.json" in url:
            return _TICKERS_JSON
        if "/submissions/" in url:
            return _TWO_8K_SUBS
        raise VendorRateLimitError("throttled")

    monkeypatch.setattr(se, "_edgar_get", fake)
    with pytest.raises(VendorRateLimitError):
        se.get_sec_filing("AAPL", curr_date="2024-06-01", form_type="8-K")


# --- Tool contract ----------------------------------------------------------


@pytest.mark.unit
def test_tool_requires_curr_date():
    # The LLM-facing tool must require curr_date so a backtest cannot silently
    # omit it and receive the newest filing overall (look-ahead).
    import inspect

    from tradingagents.agents.utils.fundamental_data_tools import (
        get_sec_filing,
        read_sec_filing,
        search_sec_filing,
    )

    for tool in (get_sec_filing, search_sec_filing, read_sec_filing):
        sig = inspect.signature(tool.func)
        assert sig.parameters["curr_date"].default is inspect.Parameter.empty
    assert inspect.signature(get_sec_filing.func).parameters["form_type"].default == "latest"


# --- EDGAR fair-access behavior ---------------------------------------------


@pytest.mark.unit
def test_403_raises_rate_limit():
    resp = mock.MagicMock(status_code=403)
    with (
        mock.patch.object(se.requests, "get", return_value=resp),
        pytest.raises(VendorRateLimitError),
    ):
        se._edgar_get("https://data.sec.gov/submissions/CIK0000320193.json")


@pytest.mark.unit
def test_user_agent_env_override(monkeypatch):
    resp = mock.MagicMock(status_code=200, text="ok")

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test Person test@example.com")
    with mock.patch.object(se.requests, "get", return_value=resp) as mget:
        assert se._edgar_get("https://www.sec.gov/files/company_tickers.json") == "ok"
    assert mget.call_args.kwargs["headers"]["User-Agent"] == "Test Person test@example.com"

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    with mock.patch.object(se.requests, "get", return_value=resp) as mget:
        se._edgar_get("https://www.sec.gov/files/company_tickers.json")
    assert mget.call_args.kwargs["headers"]["User-Agent"] == se._DEFAULT_UA


# --- Routing ----------------------------------------------------------------


@pytest.mark.unit
def test_routing_degrades_when_vendor_unavailable():
    # sec_filings is optional: an EDGAR failure must not abort the fundamentals
    # node — the router returns a sentinel instead.
    _reset_config()
    try:
        set_config({"data_vendors": {"sec_filings": "sec_edgar"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_sec_filing": {"sec_edgar": _raise_boom}},
            clear=False,
        ):
            result = interface.route_to_vendor("get_sec_filing", "AAPL", "2024-06-01", "latest")
        assert "DATA_UNAVAILABLE" in result
        assert "sec_filings" in result
    finally:
        _reset_config()
