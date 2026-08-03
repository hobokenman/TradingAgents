"""Buffett Researcher as an Analysts Team checkbox option."""

from unittest import mock

import pytest

import cli.main as m
from cli.models import AnalystType
from cli.utils import BUFFETT_SELECTION_VALUE, partition_analyst_selection


@pytest.mark.unit
class TestPartitionAnalystSelection:
    def test_analysts_only(self):
        analysts, buffett = partition_analyst_selection([AnalystType.MARKET, AnalystType.NEWS])
        assert analysts == [AnalystType.MARKET, AnalystType.NEWS]
        assert buffett is False

    def test_buffett_alone(self):
        analysts, buffett = partition_analyst_selection([BUFFETT_SELECTION_VALUE])
        assert analysts == []
        assert buffett is True

    def test_analysts_and_buffett(self):
        analysts, buffett = partition_analyst_selection(
            [AnalystType.MARKET, BUFFETT_SELECTION_VALUE, AnalystType.FUNDAMENTALS]
        )
        assert analysts == [AnalystType.MARKET, AnalystType.FUNDAMENTALS]
        assert buffett is True

    def test_empty(self):
        analysts, buffett = partition_analyst_selection([])
        assert analysts == []
        assert buffett is False


@pytest.mark.unit
class TestBuildRunConfigBuffett:
    def _selections(self, buffett_enabled: bool) -> dict:
        return {
            "research_depth": 3,
            "shallow_thinker": "gpt-5.4-mini",
            "deep_thinker": "gpt-5.5",
            "backend_url": None,
            "llm_provider": "openai",
            "google_thinking_level": None,
            "openai_reasoning_effort": None,
            "codex_reasoning_effort": None,
            "anthropic_effort": None,
            "output_language": "English",
            "buffett_researcher_enabled": buffett_enabled,
        }

    def test_selection_disables_buffett_without_env(self, monkeypatch):
        monkeypatch.delenv("TRADINGAGENTS_BUFFETT_RESEARCHER", raising=False)
        cfg = m._build_run_config(self._selections(False), checkpoint=None)
        assert cfg["buffett_researcher_enabled"] is False

    def test_selection_enables_buffett_without_env(self, monkeypatch):
        monkeypatch.delenv("TRADINGAGENTS_BUFFETT_RESEARCHER", raising=False)
        cfg = m._build_run_config(self._selections(True), checkpoint=None)
        assert cfg["buffett_researcher_enabled"] is True

    def test_env_wins_over_selection(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_BUFFETT_RESEARCHER", "true")
        patched = dict(m.DEFAULT_CONFIG, buffett_researcher_enabled=True)
        with mock.patch.object(m, "DEFAULT_CONFIG", patched):
            cfg = m._build_run_config(self._selections(False), checkpoint=None)
        assert cfg["buffett_researcher_enabled"] is True


@pytest.mark.unit
class TestGetUserSelectionsBuffett:
    def _run(self, env, fake_cfg, select_return):
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(m, "DEFAULT_CONFIG", fake_cfg),
            mock.patch.object(m, "fetch_announcements", return_value=None),
            mock.patch.object(m, "display_announcements"),
            mock.patch.object(m, "get_ticker", return_value="AAPL"),
            mock.patch.object(m, "get_analysis_date", return_value="2026-05-29"),
            mock.patch.object(m, "select_analysts", return_value=select_return),
            mock.patch.object(m, "select_research_depth", return_value=1),
            mock.patch.object(m, "ensure_api_key"),
            mock.patch.object(m, "select_llm_provider", return_value=("openai", None)),
            mock.patch.object(m, "ask_output_language", return_value="English"),
            mock.patch.object(m, "select_shallow_thinking_agent", return_value="gpt-5.4-mini"),
            mock.patch.object(m, "select_deep_thinking_agent", return_value="gpt-5.5"),
            mock.patch.object(m, "ask_openai_reasoning_effort", return_value=None),
        ):
            return m.get_user_selections()

    def test_checkbox_disables_buffett(self, monkeypatch):
        monkeypatch.delenv("TRADINGAGENTS_BUFFETT_RESEARCHER", raising=False)
        sel = self._run({}, dict(m.DEFAULT_CONFIG), ([AnalystType.MARKET], False))
        assert sel["buffett_researcher_enabled"] is False
        assert sel["analysts"] == [AnalystType.MARKET]

    def test_env_overrides_checkbox_false(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_BUFFETT_RESEARCHER", "false")
        fake_cfg = dict(m.DEFAULT_CONFIG, buffett_researcher_enabled=False)
        # Checkbox would have returned True, but env wins.
        sel = self._run(
            {"TRADINGAGENTS_BUFFETT_RESEARCHER": "false"},
            fake_cfg,
            ([AnalystType.MARKET], True),
        )
        assert sel["buffett_researcher_enabled"] is False
