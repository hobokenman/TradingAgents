"""Codex subscription provider contract, using an offline SDK stub."""

from __future__ import annotations

import json
import sys
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.codex_chat import (
    CodexAuthenticationError,
    CodexUnavailableError,
    CodexUsageLimitError,
    inspect_codex_subscription,
    render_messages,
)
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options


def _fake_sdk(
    *,
    response: str = "final report",
    account_type: str | None = "chatgpt",
    error: Exception | None = None,
):
    module = types.ModuleType("openai_codex")
    captured: dict = {}

    class CodexConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class Sandbox:
        read_only = "read-only"

    class Thread:
        id = "thread-1"

        def run(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["run"] = kwargs
            if error:
                raise error
            return types.SimpleNamespace(
                id="turn-1",
                final_response=response,
                usage={
                    "total": {
                        "inputTokens": 30,
                        "outputTokens": 12,
                        "totalTokens": 42,
                    }
                },
                duration_ms=125,
            )

    class Codex:
        def __init__(self, config=None):
            captured["codex_config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def account(self):
            if account_type is None:
                return types.SimpleNamespace(account=None)
            account = types.SimpleNamespace(
                type=account_type,
                email="trader@example.com",
                plan_type="pro",
            )
            return types.SimpleNamespace(
                account=types.SimpleNamespace(root=account),
                requires_openai_auth=True,
            )

        def models(self):
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(model="gpt-5.6-sol", hidden=False),
                    types.SimpleNamespace(model="gpt-5.6-luna", hidden=False),
                    types.SimpleNamespace(model="hidden-model", hidden=True),
                ]
            )

        def thread_start(self, **kwargs):
            captured["thread_start"] = kwargs
            return Thread()

    module.Codex = Codex
    module.CodexConfig = CodexConfig
    module.Sandbox = Sandbox
    module.captured = captured
    return module


@pytest.fixture
def sdk(monkeypatch):
    def install(**kwargs):
        module = _fake_sdk(**kwargs)
        monkeypatch.setitem(sys.modules, "openai_codex", module)
        inspect_codex_subscription.cache_clear()
        return module

    yield install
    inspect_codex_subscription.cache_clear()


def _llm(**kwargs):
    return create_llm_client("codex", kwargs.pop("model", "gpt-5.6-sol"), **kwargs).get_llm()


@pytest.mark.unit
def test_factory_routes_codex():
    client = create_llm_client("codex", "gpt-5.6-sol")
    assert type(client).__name__ == "CodexClient"


@pytest.mark.unit
def test_sdk_options_are_subscription_only_ephemeral_and_read_only(sdk):
    module = sdk()
    llm = _llm(
        effort="high",
        temperature=0.7,
        api_key="must-not-leak",
        reasoning_effort="max",
        max_retries=9,
    )
    assert llm.effort == "high"
    assert not hasattr(llm, "temperature")

    result = llm.invoke("analyze SPY")
    captured = module.captured
    assert captured["config"]["env"]["OPENAI_API_KEY"] == ""
    assert captured["config"]["env"]["CODEX_API_KEY"] == ""
    assert captured["thread_start"]["ephemeral"] is True
    assert captured["thread_start"]["sandbox"] == "read-only"
    assert captured["run"]["sandbox"] == "read-only"
    assert captured["run"]["effort"] == "high"
    assert "Do not inspect the workspace" in captured["thread_start"]["base_instructions"]
    assert result.content == "final report"
    assert result.response_metadata["thread_id"] == "thread-1"
    assert result.usage_metadata == {
        "input_tokens": 30,
        "output_tokens": 12,
        "total_tokens": 42,
    }


@pytest.mark.unit
def test_api_key_account_is_rejected_before_a_turn(sdk):
    module = sdk(account_type="apiKey")
    with pytest.raises(CodexAuthenticationError, match="API-key accounts"):
        _llm().invoke("hi")
    assert "thread_start" not in module.captured


@pytest.mark.unit
def test_logged_out_account_is_rejected(sdk):
    sdk(account_type=None)
    with pytest.raises(CodexAuthenticationError, match="codex login"):
        _llm().invoke("hi")


@pytest.mark.unit
def test_missing_sdk_has_actionable_install_and_login_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai_codex", None)
    with pytest.raises(CodexUnavailableError, match="uv sync --extra codex"):
        _llm().invoke("hi")


@pytest.mark.unit
def test_bound_tools_return_langchain_tool_calls(sdk):
    payload = {
        "kind": "tool_calls",
        "content": "",
        "tool_calls": [
            {
                "name": "get_stock_data",
                "arguments_json": '{"symbol": "SPY"}',
                "id": "call-1",
            }
        ],
    }
    module = sdk(response=json.dumps(payload))

    @tool
    def get_stock_data(symbol: str) -> str:
        """Fetch stock prices."""
        return f"prices for {symbol}"

    result = _llm().bind_tools([get_stock_data]).invoke("analyze SPY")
    assert result.tool_calls == [
        {
            "name": "get_stock_data",
            "args": {"symbol": "SPY"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]
    schema = module.captured["run"]["output_schema"]
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == [
        "get_stock_data"
    ]
    developer = module.captured["thread_start"]["developer_instructions"]
    assert "get_stock_data" in developer
    assert '"symbol"' in developer


@pytest.mark.unit
def test_tool_protocol_final_answer_has_no_tool_calls(sdk):
    sdk(response=json.dumps({"kind": "final", "content": "BUY", "tool_calls": []}))

    @tool
    def prices(symbol: str) -> str:
        """Fetch prices."""
        return symbol

    result = _llm().bind_tools([prices]).invoke("decide")
    assert result.content == "BUY"
    assert result.tool_calls == []


@pytest.mark.unit
def test_message_rendering_preserves_tool_request_and_result():
    messages = [
        HumanMessage("analyze SPY"),
        AIMessage(
            content="",
            tool_calls=[{"name": "prices", "args": {"symbol": "SPY"}, "id": "call-1"}],
        ),
        ToolMessage(content="100.00", tool_call_id="call-1"),
    ]
    _, prompt = render_messages(messages)
    assert "Requested tools:" in prompt
    assert '"symbol": "SPY"' in prompt
    assert "Tool result: 100.00" in prompt


class _Decision(BaseModel):
    action: str
    confidence: float


@pytest.mark.unit
def test_structured_output_uses_sdk_schema_and_parses(sdk):
    module = sdk(response='{"action": "BUY", "confidence": 0.8}')
    parsed = _llm().with_structured_output(_Decision).invoke("decide")
    assert parsed == _Decision(action="BUY", confidence=0.8)
    schema = module.captured["run"]["output_schema"]
    assert schema["properties"]["action"]["type"] == "string"


@pytest.mark.unit
def test_subscription_inspection_discovers_visible_models(sdk):
    sdk()
    info = inspect_codex_subscription()
    assert info.email == "trader@example.com"
    assert info.plan_type == "pro"
    assert info.models == ("gpt-5.6-sol", "gpt-5.6-luna")


@pytest.mark.unit
def test_catalog_contains_latest_role_specific_models():
    assert get_model_options("codex", "deep")[0][1] == "gpt-5.6-sol"
    assert get_model_options("codex", "quick")[0][1] == "gpt-5.6-luna"
    assert get_model_options("openai", "deep")[0][1] == "gpt-5.6-sol"
    assert get_model_options("openai", "quick")[0][1] == "gpt-5.6-luna"


@pytest.mark.unit
def test_account_model_filter_preserves_quick_and_deep_roles(sdk):
    from cli.utils import _available_codex_model_options

    sdk()
    quick = _available_codex_model_options(get_model_options("codex", "quick"))
    deep = _available_codex_model_options(get_model_options("codex", "deep"))
    assert [value for _, value in quick] == ["gpt-5.6-luna", "custom"]
    assert [value for _, value in deep] == ["gpt-5.6-sol", "custom"]


@pytest.mark.unit
def test_graph_maps_codex_effort_to_sdk_kwarg():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "codex",
        "codex_reasoning_effort": "xhigh",
        "temperature": None,
        "llm_max_retries": None,
    }
    assert graph._get_provider_kwargs() == {"effort": "xhigh"}


@pytest.mark.unit
def test_usage_limit_has_dedicated_error(sdk):
    sdk(error=RuntimeError("usage limit reached"))
    with pytest.raises(CodexUsageLimitError, match="usage limit"):
        _llm().invoke("hi")
