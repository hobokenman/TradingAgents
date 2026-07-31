"""Buffett Researcher: prompt grounding, debate-state safety, and graph placement.

The agent runs once between the analysts and the Bull/Bear debate. The
load-bearing constraint is that it must seed the shared debate history
*without* touching `count` or `current_response` -- see
TestDebateStateIsNotDisturbed for why each one matters.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.buffett_researcher import (
    create_buffett_researcher,
    load_buffett_principles,
    principles_override_path,
)
from tradingagents.agents.schemas import (
    BuffettAssessment,
    BuffettVerdict,
    render_buffett_assessment,
)
from tradingagents.dataflows.config import set_config
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import DEBATE_PATH_MAP, GraphSetup


def _assessment(verdict=BuffettVerdict.WONDERFUL_FAIR_PRICE):
    return BuffettAssessment(
        verdict=verdict,
        moat="Entrenched brand with demonstrated pricing power; widening.",
        management="Candid letters; buybacks below intrinsic value.",
        valuation="Owner earnings well above reported; 30% discount to intrinsic value.",
        thesis="A wonderful business available at a fair price.",
    )


def _state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "instrument_context": "Ticker: AAPL (Apple Inc.)",
        "market_report": "MARKET_REPORT_SENTINEL",
        "sentiment_report": "SENTIMENT_REPORT_SENTINEL",
        "news_report": "NEWS_REPORT_SENTINEL",
        "fundamentals_report": "FUNDAMENTALS_REPORT_SENTINEL",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }
    state.update(overrides)
    return state


def _structured_llm(captured: dict, assessment: BuffettAssessment | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns a schema."""
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or (assessment or _assessment())
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _freetext_llm(captured: dict, content="FREE TEXT ASSESSMENT"):
    """MagicMock LLM with no structured-output support, forcing the fallback path."""
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")
    llm.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or MagicMock(content=content)
    )
    return llm


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderBuffettAssessment:
    def test_all_sections_present(self):
        md = render_buffett_assessment(_assessment())
        assert "**Verdict**: Wonderful Business, Fair Price" in md
        assert "**Moat**: Entrenched brand" in md
        assert "**Management**: Candid letters" in md
        assert "**Valuation**: Owner earnings" in md
        assert "**Thesis**: A wonderful business" in md

    def test_every_verdict_renders(self):
        for verdict in BuffettVerdict:
            md = render_buffett_assessment(_assessment(verdict))
            assert f"**Verdict**: {verdict.value}" in md


# ---------------------------------------------------------------------------
# Principles document
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrinciplesDocument:
    def test_baseline_ships_with_the_package(self):
        text = load_buffett_principles()
        assert "Circle of Competence" in text
        assert "Evaluation Checklist" in text

    def test_override_takes_precedence(self, tmp_path):
        override = tmp_path / "principles.md"
        override.write_text("USER DISTILLED PRINCIPLES", encoding="utf-8")
        set_config({"buffett_principles_path": str(override)})
        assert load_buffett_principles() == "USER DISTILLED PRINCIPLES"

    def test_falls_back_to_baseline_when_override_missing(self, tmp_path):
        set_config({"buffett_principles_path": str(tmp_path / "absent.md")})
        assert "Circle of Competence" in load_buffett_principles()

    def test_configured_path_wins_over_home_default(self, tmp_path):
        set_config({"buffett_principles_path": str(tmp_path / "p.md")})
        assert principles_override_path() == tmp_path / "p.md"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrompt:
    def test_prompt_carries_principles_and_every_analyst_report(self, tmp_path):
        override = tmp_path / "principles.md"
        override.write_text("PRINCIPLES_SENTINEL", encoding="utf-8")
        set_config({"buffett_principles_path": str(override)})

        captured = {}
        create_buffett_researcher(_structured_llm(captured))(_state())

        prompt = captured["prompt"]
        assert "PRINCIPLES_SENTINEL" in prompt
        for sentinel in (
            "MARKET_REPORT_SENTINEL",
            "SENTIMENT_REPORT_SENTINEL",
            "NEWS_REPORT_SENTINEL",
            "FUNDAMENTALS_REPORT_SENTINEL",
        ):
            assert sentinel in prompt
        assert "Apple Inc." in prompt

    def test_prompt_licenses_the_too_hard_verdict(self):
        captured = {}
        create_buffett_researcher(_structured_llm(captured))(_state())
        assert "Too Hard" in captured["prompt"]

    def test_structured_output_renders_to_markdown(self):
        result = create_buffett_researcher(_structured_llm({}))(_state())
        assert "**Verdict**: Wonderful Business, Fair Price" in result["buffett_report"]

    def test_free_text_fallback_when_provider_lacks_structured_output(self):
        captured = {}
        result = create_buffett_researcher(_freetext_llm(captured))(_state())
        assert result["buffett_report"] == "FREE TEXT ASSESSMENT"
        assert "PRINCIPLES" in captured["prompt"].upper()


# ---------------------------------------------------------------------------
# Debate-state safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDebateStateIsNotDisturbed:
    """The node contributes to the debate history but takes no turn in the debate."""

    def test_history_is_seeded_with_the_assessment(self):
        result = create_buffett_researcher(_structured_llm({}))(_state())
        history = result["investment_debate_state"]["history"]
        assert history.startswith("Buffett Researcher:")
        assert "Wonderful Business, Fair Price" in history

    def test_existing_history_is_preserved(self):
        state = _state()
        state["investment_debate_state"]["history"] = "Earlier note."
        result = create_buffett_researcher(_structured_llm({}))(state)
        history = result["investment_debate_state"]["history"]
        assert history.startswith("Earlier note.")
        assert "Buffett Researcher:" in history

    def test_count_is_untouched(self):
        """Incrementing count would end the debate one speaker early: the router
        stops at `count >= 2 * max_debate_rounds`, so starting at 1 means Bear
        never speaks."""
        result = create_buffett_researcher(_structured_llm({}))(_state())
        assert result["investment_debate_state"]["count"] == 0

    def test_current_response_is_untouched(self):
        """The Bull prompt reads current_response as 'Last bear argument', so
        writing the value thesis there would misattribute it."""
        result = create_buffett_researcher(_structured_llm({}))(_state())
        assert result["investment_debate_state"]["current_response"] == ""

    def test_bull_and_bear_histories_pass_through(self):
        state = _state()
        state["investment_debate_state"]["bull_history"] = "BULL"
        state["investment_debate_state"]["bear_history"] = "BEAR"
        debate = create_buffett_researcher(_structured_llm({}))(state)["investment_debate_state"]
        assert debate["bull_history"] == "BULL"
        assert debate["bear_history"] == "BEAR"

    def test_debate_router_still_alternates_after_the_node_runs(self):
        """End-to-end on the state the node emits: Bull, then Bear, then manager."""
        logic = ConditionalLogic(max_debate_rounds=1)
        debate = create_buffett_researcher(_structured_llm({}))(_state())["investment_debate_state"]

        # Bull speaks first and hands to Bear.
        after_bull = {**debate, "current_response": "Bull Analyst: ...", "count": 1}
        assert logic.should_continue_debate({"investment_debate_state": after_bull}) == (
            "Bear Researcher"
        )

        # Bear speaks and the debate ends at the round limit.
        after_bear = {**debate, "current_response": "Bear Analyst: ...", "count": 2}
        assert logic.should_continue_debate({"investment_debate_state": after_bear}) == (
            "Research Manager"
        )

    def test_speaker_prefix_does_not_collide_with_the_router(self):
        """The router switches speakers on a "Bull" prefix; ours must not match."""
        logic = ConditionalLogic(max_debate_rounds=1)
        state = {
            "investment_debate_state": {"current_response": "Buffett Researcher: ...", "count": 0}
        }
        assert logic.should_continue_debate(state) in DEBATE_PATH_MAP


# ---------------------------------------------------------------------------
# Non-stock assets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutOfCircleAbstention:
    def test_crypto_abstains_without_calling_the_model(self):
        llm = _structured_llm({})
        result = create_buffett_researcher(llm)(_state(asset_type="crypto"))

        llm.with_structured_output.return_value.invoke.assert_not_called()
        llm.invoke.assert_not_called()
        assert "**Verdict**: Too Hard" in result["buffett_report"]

    def test_abstention_reads_as_neutral_not_bearish(self):
        result = create_buffett_researcher(_structured_llm({}))(_state(asset_type="crypto"))
        assert "abstention" in result["buffett_report"].lower()

    def test_abstention_still_seeds_the_debate_history(self):
        result = create_buffett_researcher(_structured_llm({}))(_state(asset_type="crypto"))
        assert "Buffett Researcher:" in result["investment_debate_state"]["history"]


# ---------------------------------------------------------------------------
# State and graph wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphWiring:
    def test_initial_state_seeds_the_report_field(self):
        state = Propagator().create_initial_state("AAPL", "2026-01-15")
        assert state["buffett_report"] == ""

    def _nodes_and_edges(self, buffett_enabled):
        setup = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={key: MagicMock() for key in ("market", "social", "news", "fundamentals")},
            conditional_logic=ConditionalLogic(),
            buffett_enabled=buffett_enabled,
        )
        workflow = setup.setup_graph(["market", "fundamentals"])
        edges = {(start, end) for start, end in workflow.edges}
        return set(workflow.nodes), edges

    def test_enabled_inserts_the_node_between_analysts_and_bull(self):
        nodes, edges = self._nodes_and_edges(buffett_enabled=True)
        assert "Buffett Researcher" in nodes
        assert ("Msg Clear Fundamentals", "Buffett Researcher") in edges
        assert ("Buffett Researcher", "Bull Researcher") in edges
        assert ("Msg Clear Fundamentals", "Bull Researcher") not in edges

    def test_disabled_restores_the_direct_handoff(self):
        nodes, edges = self._nodes_and_edges(buffett_enabled=False)
        assert "Buffett Researcher" not in nodes
        assert ("Msg Clear Fundamentals", "Bull Researcher") in edges

    def test_graph_compiles_with_the_node_present(self):
        setup = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={key: MagicMock() for key in ("market", "social", "news", "fundamentals")},
            conditional_logic=ConditionalLogic(),
            buffett_enabled=True,
        )
        setup.setup_graph(["market", "social", "news", "fundamentals"]).compile()


# ---------------------------------------------------------------------------
# Downstream consumers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownstreamConsumers:
    def test_research_manager_gets_the_assessment_as_its_own_section(self):
        from tradingagents.agents.managers.research_manager import create_research_manager

        captured = {}
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("unsupported")
        llm.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or MagicMock(content="PLAN")
        )

        state = _state(buffett_report="BUFFETT_ASSESSMENT_SENTINEL")
        state["investment_debate_state"]["history"] = "DEBATE_HISTORY_SENTINEL"
        create_research_manager(llm)(state)

        prompt = captured["prompt"]
        assert "BUFFETT_ASSESSMENT_SENTINEL" in prompt
        assert "Independent Value-Investing Assessment" in prompt
        # Presented separately from the debate, not merged into it.
        assert prompt.index("BUFFETT_ASSESSMENT_SENTINEL") < prompt.index("DEBATE_HISTORY_SENTINEL")

    def test_research_manager_omits_the_section_when_disabled(self):
        from tradingagents.agents.managers.research_manager import create_research_manager

        captured = {}
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("unsupported")
        llm.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or MagicMock(content="PLAN")
        )
        create_research_manager(llm)(_state())
        assert "Independent Value-Investing Assessment" not in captured["prompt"]

    @pytest.mark.parametrize(
        "factory_path,factory_name",
        [
            ("tradingagents.agents.researchers.bull_researcher", "create_bull_researcher"),
            ("tradingagents.agents.researchers.bear_researcher", "create_bear_researcher"),
        ],
    )
    def test_debaters_see_the_assessment(self, factory_path, factory_name):
        import importlib

        factory = getattr(importlib.import_module(factory_path), factory_name)
        captured = {}
        llm = MagicMock()
        llm.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or MagicMock(content="ARG")
        )
        factory(llm)(_state(buffett_report="BUFFETT_ASSESSMENT_SENTINEL"))
        assert "BUFFETT_ASSESSMENT_SENTINEL" in captured["prompt"]

    def test_report_tree_writes_the_assessment(self, tmp_path):
        from tradingagents.reporting import write_report_tree

        write_report_tree(
            {
                "market_report": "MKT",
                "buffett_report": "BUFFETT REPORT",
                "investment_debate_state": {"judge_decision": "RM PLAN"},
            },
            "AAPL",
            tmp_path,
        )
        assert (tmp_path / "2_research" / "buffett.md").read_text() == "BUFFETT REPORT"
        assert "BUFFETT REPORT" in (tmp_path / "complete_report.md").read_text()


# ---------------------------------------------------------------------------
# CLI progress table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCliStatusTable:
    def _buffer(self, buffett_enabled=True):
        from cli.main import MessageBuffer

        buffer = MessageBuffer()
        buffer.init_for_analysis(["market", "fundamentals"], buffett_enabled=buffett_enabled)
        return buffer

    def test_agent_appears_first_in_the_research_team(self):
        from cli.main import research_team_agents

        assert research_team_agents(self._buffer())[0] == "Buffett Researcher"

    def test_agent_is_hidden_when_disabled(self):
        from cli.main import research_team_agents

        buffer = self._buffer(buffett_enabled=False)
        assert "Buffett Researcher" not in buffer.agent_status
        assert research_team_agents(buffer)[0] == "Bull Researcher"

    def test_analysts_finishing_starts_the_buffett_researcher(self):
        from cli.main import update_analyst_statuses

        buffer = self._buffer()
        chunk = {"market_report": "MKT", "fundamentals_report": "FUND"}
        update_analyst_statuses(buffer, chunk)
        assert buffer.agent_status["Buffett Researcher"] == "in_progress"
        assert buffer.agent_status["Bull Researcher"] == "pending"

    def test_analysts_finishing_starts_bull_when_disabled(self):
        from cli.main import update_analyst_statuses

        buffer = self._buffer(buffett_enabled=False)
        update_analyst_statuses(buffer, {"market_report": "MKT", "fundamentals_report": "FUND"})
        assert buffer.agent_status["Bull Researcher"] == "in_progress"

    def test_debate_transition_does_not_reopen_a_finished_agent(self, monkeypatch):
        """The debate's bulk in_progress sweep must not walk Buffett back."""
        import cli.main

        buffer = self._buffer()
        buffer.update_agent_status("Buffett Researcher", "completed")
        monkeypatch.setattr(cli.main, "message_buffer", buffer)

        cli.main.update_research_team_status("in_progress")

        assert buffer.agent_status["Buffett Researcher"] == "completed"
        assert buffer.agent_status["Bull Researcher"] == "in_progress"
