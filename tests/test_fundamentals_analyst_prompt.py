"""The fundamentals analyst must inject earnings-call and SEC-filing SOPs.

Guards that the SOP guidance actually reaches the LLM's system prompt and that
``system_message`` is a real ``str`` (a prior trailing comma made it a 1-element
tuple that only worked because LangChain stringified it).
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import tradingagents.agents.analysts.fundamentals_analyst as fa


def _run_node_and_capture_system_prompt():
    """Invoke the node with a mocked LLM; return the formatted system message text."""
    captured = {}

    bound = MagicMock()

    def fake_invoke(prompt_value):
        # ``prompt | llm`` wraps the callable bound LLM in a RunnableLambda, so
        # it is called directly with the formatted PromptValue (not .invoke).
        captured["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok")  # no tool_calls -> node finalizes the report

    bound.side_effect = fake_invoke

    llm = MagicMock()
    llm.bind_tools.return_value = bound

    node = fa.create_fundamentals_analyst(llm)
    state = {
        "trade_date": "2025-07-28",
        "instrument_context": "The instrument is AAPL (a US stock).",
        "messages": [],
    }
    node(state)
    return captured["messages"][0].content


@pytest.mark.unit
def test_sop_constant_is_nonempty_str():
    assert isinstance(fa.EARNINGS_CALL_SOP, str)
    assert fa.EARNINGS_CALL_SOP.strip()


@pytest.mark.unit
def test_sec_sop_constant_is_nonempty_str():
    assert isinstance(fa.SEC_FILINGS_SOP, str)
    assert fa.SEC_FILINGS_SOP.strip()


@pytest.mark.unit
def test_system_prompt_contains_sec_sop():
    system_text = _run_node_and_capture_system_prompt()
    # Distinctive SEC-filing SOP markers must survive into the system prompt.
    assert "get_sec_filing" in system_text
    assert "search_sec_filing" in system_text
    assert "read_sec_filing" in system_text
    assert "bounded metadata/heading index" in system_text
    assert "do not request that same form again" in system_text
    assert "Never claim the filing is unreviewable merely because it is large" in system_text
    assert "Executive investment conclusion" in system_text
    assert "Red flags and positive inflections" in system_text
    assert "five items to monitor in the next filing" in system_text
    # 8-K event guidance is present.
    assert "8-K event filings" in system_text
    assert "routine housekeeping or thesis-relevant" in system_text
    # Fact/statement/interpretation discipline is preserved.
    assert (
        "separate reported facts, management statements, and analyst interpretation" in system_text
    )


@pytest.mark.unit
def test_sec_filing_tool_is_bound():
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock(side_effect=lambda pv: AIMessage(content="ok"))
    node = fa.create_fundamentals_analyst(llm)
    node(
        {
            "trade_date": "2025-07-28",
            "instrument_context": "The instrument is AAPL (a US stock).",
            "messages": [],
        }
    )
    bound_names = [tool.name for tool in llm.bind_tools.call_args[0][0]]
    assert {"get_sec_filing", "search_sec_filing", "read_sec_filing"} <= set(bound_names)


@pytest.mark.unit
def test_system_prompt_contains_sop_checklist():
    system_text = _run_node_and_capture_system_prompt()
    # A few distinctive SOP markers must survive into the actual system prompt.
    assert "five most important new disclosures" in system_text
    assert "conservative, realistic, aggressive, or back-end loaded" in system_text
    assert "three most important things to monitor next quarter" in system_text
    # Fact/claim/interpretation discipline is preserved.
    assert "separate reported facts, management claims, and your own interpretation" in system_text


@pytest.mark.unit
def test_system_message_is_str_not_tuple():
    # Regression: the block used to end with a trailing comma, making
    # ``system_message`` a tuple. If that regresses, the SOP is still present
    # (LangChain stringifies the tuple) so this asserts the type directly.
    system_text = _run_node_and_capture_system_prompt()
    assert isinstance(system_text, str)
    # A stringified 1-tuple would render as "('...',)"; the real prompt won't.
    assert not system_text.rstrip().endswith("',)")
