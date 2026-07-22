"""FINRA short sale vendor: facility aggregation, SVR math, short interest
formatting, lookahead-safe windowing, HTTP/token handling, and router
integration.

All API access is mocked, so these run without a network connection or a key.
"""

import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import finra, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorRateLimitError

# Two trading days, each split across reporting facilities, so aggregation
# is exercised: 2026-07-01 -> short 600 / total 1000 (SVR 60%),
# 2026-07-02 -> short 300 / total 1000 (SVR 30%).
_SHORT_VOLUME_ROWS = [
    {
        "tradeReportDate": "2026-07-01",
        "securitiesInformationProcessorSymbolIdentifier": "AAPL",
        "shortParQuantity": "400.000000",
        "shortExemptParQuantity": "10.000000",
        "totalParQuantity": "700.000000",
        "reportingFacilityCode": "N",
    },
    {
        "tradeReportDate": "2026-07-01",
        "securitiesInformationProcessorSymbolIdentifier": "AAPL",
        "shortParQuantity": "200.000000",
        "shortExemptParQuantity": "5.000000",
        "totalParQuantity": "300.000000",
        "reportingFacilityCode": "Q",
    },
    {
        "tradeReportDate": "2026-07-02",
        "securitiesInformationProcessorSymbolIdentifier": "AAPL",
        "shortParQuantity": "300.000000",
        "shortExemptParQuantity": "0.000000",
        "totalParQuantity": "1000.000000",
        "reportingFacilityCode": "N",
    },
]

_SHORT_INTEREST_ROWS = [
    {
        "settlementDate": "2026-06-30",
        "symbolCode": "AAPL",
        "issueName": "Apple Inc.",
        "currentShortPositionQuantity": "120000000.000000",
        "changePercent": "20.00",
        "daysToCoverQuantity": "2.50",
        "averageDailyVolumeQuantity": "48000000.000000",
    },
    {
        "settlementDate": "2026-06-15",
        "symbolCode": "AAPL",
        "issueName": "Apple Inc.",
        "currentShortPositionQuantity": "100000000.000000",
        "changePercent": "-5.00",
        "daysToCoverQuantity": "2.10",
        "averageDailyVolumeQuantity": "47000000.000000",
    },
]


@pytest.mark.unit
class FinraShortVolumeTests(unittest.TestCase):
    def test_aggregates_facilities_and_computes_svr(self):
        with mock.patch.object(finra, "_query_dataset", return_value=_SHORT_VOLUME_ROWS):
            out = finra.get_short_volume("aapl", "2026-07-03", 30)
        self.assertIn("## FINRA Daily Short Sale Volume: AAPL", out)
        # 2026-07-01 across N + Q facilities: 600 short / 1000 total = 60.0%
        self.assertIn("| 2026-07-01 | 600 | 15 | 1,000 | 60.0% |", out)
        self.assertIn("| 2026-07-02 | 300 | 0 | 1,000 | 30.0% |", out)
        # Latest day (sorted by date, not row order) and period mean (45%).
        self.assertIn("**Latest SVR:** 30.0% (2026-07-02)", out)
        self.assertIn("**Period mean:** 45.0%", out)

    def test_window_is_lookahead_safe(self):
        captured = {}

        def _capture(dataset, symbol_field, symbol, date_field, start, end):
            captured.update(start=start, end=end, dataset=dataset)
            return _SHORT_VOLUME_ROWS

        with mock.patch.object(finra, "_query_dataset", side_effect=_capture):
            finra.get_short_volume("AAPL", "2026-07-03", 30)
        self.assertEqual(captured["dataset"], "regShoDaily")
        self.assertEqual(captured["end"], "2026-07-03")
        self.assertEqual(captured["start"], "2026-06-03")  # 30d back

    def test_empty_result_raises_no_market_data(self):
        with (
            mock.patch.object(finra, "_query_dataset", return_value=[]),
            self.assertRaises(NoMarketDataError),
        ):
            finra.get_short_volume("ZZZZ", "2026-07-03", 30)

    def test_long_window_table_is_truncated_but_stats_use_full_range(self):
        # One facility row per day, MAX_ROWS + 10 days: the table must cap at
        # MAX_ROWS recent rows while summary stats still cover every day.
        rows = [
            {
                "tradeReportDate": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "shortParQuantity": "500",
                "shortExemptParQuantity": "0",
                "totalParQuantity": "1000",
            }
            for i in range(finra.MAX_ROWS + 10)
        ]
        with mock.patch.object(finra, "_query_dataset", return_value=rows):
            out = finra.get_short_volume("AAPL", "2026-12-31", 365)
        self.assertIn(f"most recent {finra.MAX_ROWS} of {finra.MAX_ROWS + 10}", out)
        self.assertIn(f"({finra.MAX_ROWS + 10} trading days)", out)
        body_rows = [ln for ln in out.splitlines() if ln.startswith("| 2026")]
        self.assertEqual(len(body_rows), finra.MAX_ROWS)

    def test_zero_total_volume_raises_no_market_data(self):
        rows = [
            {
                "tradeReportDate": "2026-07-01",
                "shortParQuantity": "0",
                "shortExemptParQuantity": "0",
                "totalParQuantity": "0",
            }
        ]
        with (
            mock.patch.object(finra, "_query_dataset", return_value=rows),
            self.assertRaises(NoMarketDataError),
        ):
            finra.get_short_volume("AAPL", "2026-07-03", 30)


@pytest.mark.unit
class FinraShortInterestTests(unittest.TestCase):
    def test_cycles_sorted_and_summary_uses_latest(self):
        # Rows arrive newest-first; the report must sort by settlement date.
        with mock.patch.object(finra, "_query_dataset", return_value=_SHORT_INTEREST_ROWS):
            out = finra.get_short_interest("aapl", "2026-07-03", 180)
        self.assertIn("## FINRA Consolidated Short Interest: AAPL (Apple Inc.)", out)
        self.assertIn("**Latest cycle (2026-06-30):** 120,000,000 shares short", out)
        self.assertIn("**Change vs prior cycle:** +20.00%", out)
        self.assertIn("**Days to cover:** 2.5", out)
        # 100M -> 120M across the window = +20%
        self.assertIn("+20.0% across the window", out)
        rows = [ln for ln in out.splitlines() if ln.startswith("| 2026")]
        self.assertEqual(rows[0].split("|")[1].strip(), "2026-06-15")
        self.assertEqual(rows[1].split("|")[1].strip(), "2026-06-30")

    def test_window_is_lookahead_safe(self):
        captured = {}

        def _capture(dataset, symbol_field, symbol, date_field, start, end):
            captured.update(start=start, end=end, dataset=dataset, symbol_field=symbol_field)
            return _SHORT_INTEREST_ROWS

        with mock.patch.object(finra, "_query_dataset", side_effect=_capture):
            finra.get_short_interest("AAPL", "2026-07-03", 180)
        self.assertEqual(captured["dataset"], "consolidatedShortInterest")
        self.assertEqual(captured["symbol_field"], "symbolCode")
        # The settlement window ends at the dissemination cutoff, not at
        # curr_date: cycles settling within the ~2-week publication lag were
        # not yet public on curr_date and must not leak into a backtest.
        self.assertEqual(captured["end"], "2026-06-19")  # curr - 14d lag
        self.assertEqual(captured["start"], "2026-01-04")  # 180d back

    def test_single_cycle_reports_no_trend(self):
        # One cycle in the window: latest == first, so a computed "+0.0%"
        # would fake a flat trend from a single data point.
        with mock.patch.object(finra, "_query_dataset", return_value=[_SHORT_INTEREST_ROWS[0]]):
            out = finra.get_short_interest("AAPL", "2026-07-20", 30)
        self.assertIn("**Trend:** n/a (single cycle in window)", out)
        self.assertNotIn("+0.0% across the window", out)

    def test_empty_result_raises_no_market_data(self):
        with (
            mock.patch.object(finra, "_query_dataset", return_value=[]),
            self.assertRaises(NoMarketDataError),
        ):
            finra.get_short_interest("ZZZZ", "2026-07-03", 180)


class _FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text if text else ("[]" if body is None else "body")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise finra.requests.HTTPError(f"HTTP {self.status_code}")


@pytest.mark.unit
class FinraHttpTests(unittest.TestCase):
    def setUp(self):
        finra._token_cache.update(client_id=None, token=None, expires_at=0.0)

    tearDown = setUp

    def test_http_error_raises_finra_data_error(self):
        response = _FakeResponse(status_code=500, text="boom")
        with (
            mock.patch.object(finra.requests, "post", return_value=response),
            self.assertRaises(finra.FinraDataError),
        ):
            finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")

    def test_rate_limit_raises_vendor_rate_limit(self):
        response = _FakeResponse(status_code=429, text="slow down")
        with (
            mock.patch.object(finra.requests, "post", return_value=response),
            self.assertRaises(VendorRateLimitError),
        ):
            finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")

    def test_anonymous_when_no_credentials(self):
        captured = {}

        def _post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return _FakeResponse(body=[])

        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(finra.requests, "post", side_effect=_post),
        ):
            rows = finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")
        self.assertEqual(rows, [])
        self.assertNotIn("Authorization", captured["headers"])
        self.assertIn("regShoDaily", captured["url"])

    def test_bearer_token_attached_when_credentials_set(self):
        calls = []

        def _post(url, **kwargs):
            calls.append((url, kwargs))
            if url == finra.FINRA_TOKEN_URL:
                return _FakeResponse(body={"access_token": "tok123", "expires_in": 1800})
            return _FakeResponse(body=[])

        env = {"FINRA_API_CLIENT_ID": "id", "FINRA_API_CLIENT_SECRET": "secret"}
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch.object(finra.requests, "post", side_effect=_post),
        ):
            finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")
            # Second data call must reuse the cached token, not re-authenticate.
            finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")
        token_calls = [c for c in calls if c[0] == finra.FINRA_TOKEN_URL]
        data_calls = [c for c in calls if c[0] != finra.FINRA_TOKEN_URL]
        self.assertEqual(len(token_calls), 1)
        self.assertEqual(len(data_calls), 2)
        for _, kwargs in data_calls:
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tok123")

    def test_bad_token_response_raises_finra_data_error(self):
        env = {"FINRA_API_CLIENT_ID": "id", "FINRA_API_CLIENT_SECRET": "secret"}
        response = _FakeResponse(body={"unexpected": "shape"})
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch.object(finra.requests, "post", return_value=response),
            self.assertRaises(finra.FinraDataError),
        ):
            finra._get_access_token()

    def test_revoked_token_is_refreshed_and_retried_once(self):
        # A cached token rejected with 401 must be invalidated and the data
        # request retried once with a freshly fetched token.
        tokens = iter(["stale", "fresh"])
        calls = []

        def _post(url, **kwargs):
            calls.append((url, kwargs))
            if url == finra.FINRA_TOKEN_URL:
                return _FakeResponse(body={"access_token": next(tokens), "expires_in": 1800})
            auth = kwargs["headers"].get("Authorization")
            if auth == "Bearer stale":
                return _FakeResponse(status_code=401, text="revoked")
            return _FakeResponse(body=[])

        env = {"FINRA_API_CLIENT_ID": "id", "FINRA_API_CLIENT_SECRET": "secret"}
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch.object(finra.requests, "post", side_effect=_post),
        ):
            rows = finra._query_dataset("regShoDaily", "f", "AAPL", "d", "2026-06-01", "2026-07-01")
        self.assertEqual(rows, [])
        data_calls = [c for c in calls if c[0] != finra.FINRA_TOKEN_URL]
        self.assertEqual(len(data_calls), 2)
        self.assertEqual(data_calls[1][1]["headers"]["Authorization"], "Bearer fresh")

    def test_token_cache_is_keyed_by_client_id(self):
        # Rotating credentials in a long-lived process must not reuse the
        # previous client's still-unexpired token.
        calls = []

        def _post(url, **kwargs):
            calls.append(url)
            client = kwargs.get("auth", (None,))[0]
            return _FakeResponse(body={"access_token": f"tok-{client}", "expires_in": 1800})

        with mock.patch.object(finra.requests, "post", side_effect=_post):
            env_a = {"FINRA_API_CLIENT_ID": "client-a", "FINRA_API_CLIENT_SECRET": "s"}
            with mock.patch.dict("os.environ", env_a, clear=True):
                self.assertEqual(finra._get_access_token(), "tok-client-a")
            env_b = {"FINRA_API_CLIENT_ID": "client-b", "FINRA_API_CLIENT_SECRET": "s"}
            with mock.patch.dict("os.environ", env_b, clear=True):
                self.assertEqual(finra._get_access_token(), "tok-client-b")


@pytest.mark.unit
class FinraRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_short_sale_category_routes_to_finra(self):
        self.assertEqual(interface.get_category_for_method("get_short_volume"), "short_sale_data")
        self.assertEqual(
            interface.get_category_for_method("get_short_interest"),
            "short_sale_data",
        )
        set_config({"data_vendors": {"short_sale_data": "finra"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_short_volume": {"finra": lambda *a, **k: "SVR_OK"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_short_volume", "AAPL", "2026-07-03", 30)
        self.assertEqual(out, "SVR_OK")

    def test_vendor_failure_degrades_gracefully(self):
        # short_sale_data is optional: a FINRA outage degrades to a sentinel
        # instead of aborting the analysis.
        set_config({"data_vendors": {"short_sale_data": "finra"}})

        def _broken(*a, **k):
            raise finra.FinraDataError("FINRA request failed with HTTP 503")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_short_volume": {"finra": _broken}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_short_volume", "AAPL", "2026-07-03", 30)
        self.assertIn("DATA_UNAVAILABLE", out)

    def test_rate_limit_degrades_gracefully(self):
        # A 429 on the sole vendor of an optional category must degrade to the
        # sentinel like any other failure, not fall through to a RuntimeError.
        set_config({"data_vendors": {"short_sale_data": "finra"}})

        def _throttled(*a, **k):
            raise VendorRateLimitError("FINRA rate limit hit for regShoDaily")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_short_volume": {"finra": _throttled}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_short_volume", "AAPL", "2026-07-03", 30)
        self.assertIn("DATA_UNAVAILABLE", out)

    def test_no_data_returns_explicit_sentinel(self):
        set_config({"data_vendors": {"short_sale_data": "finra"}})

        def _no_rows(*a, **k):
            raise NoMarketDataError("ZZZZ", detail="no FINRA rows")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_short_interest": {"finra": _no_rows}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_short_interest", "ZZZZ", "2026-07-03", 180)
        self.assertIn("NO_DATA_AVAILABLE", out)


if __name__ == "__main__":
    unittest.main()
