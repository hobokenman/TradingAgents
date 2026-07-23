"""Claude Code — run the pipeline on a subscription instead of a metered API key.

The Claude Agent SDK runs its own agentic loop, so this provider's contract
differs from every other one: bound tools are executed inside Claude Code and
the returned message never carries ``tool_calls``. These tests pin that
contract (the graph's termination logic depends on it), the option mapping,
and the structured-output path — all against a stubbed SDK, so the suite stays
offline and the optional extra stays optional.
"""

import sys
import types

import pytest
from pydantic import BaseModel

from tradingagents.llm_clients.claude_code_chat import (
    ClaudeCodeUnavailableError,
    ClaudeCodeUsageLimitError,
    render_messages,
)
from tradingagents.llm_clients.factory import create_llm_client


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _AssistantMessage:
    def __init__(self, content):
        self.content = content


class _ResultMessage:
    def __init__(self, result="", structured_output=None, is_error=False, subtype=None):
        self.result = result
        self.structured_output = structured_output
        self.is_error = is_error
        self.subtype = subtype
        self.usage = {"input_tokens": 10}
        self.num_turns = 2
        self.session_id = "sess-1"


class _RateLimitEvent:
    def __init__(self):
        self.rate_limit_info = {"status": "warning"}


class _ClaudeSDKError(Exception):
    pass


class _CLINotFoundError(_ClaudeSDKError):
    pass


class _SdkMcpTool:
    def __init__(self, name, description, input_schema, handler):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


def _fake_sdk(messages):
    """Build a stub ``claude_agent_sdk`` module that replays ``messages``.

    The returned module exposes ``captured`` so tests can assert on the options
    the adapter built and on the MCP servers it registered.
    """
    module = types.ModuleType("claude_agent_sdk")
    captured = {}

    class _Options:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["options"] = kwargs

    async def query(*, prompt, options, transport=None):
        captured["prompt"] = prompt
        for message in messages:
            yield message

    def tool(name, description, input_schema):
        def decorator(handler):
            return _SdkMcpTool(name, description, input_schema, handler)

        return decorator

    def create_sdk_mcp_server(name, version="1.0.0", tools=None):
        captured["mcp_tools"] = tools or []
        return {"type": "sdk", "name": name}

    module.query = query
    module.tool = tool
    module.create_sdk_mcp_server = create_sdk_mcp_server
    module.ClaudeAgentOptions = _Options
    module.AssistantMessage = _AssistantMessage
    module.TextBlock = _TextBlock
    module.ResultMessage = _ResultMessage
    module.RateLimitEvent = _RateLimitEvent
    module.ClaudeSDKError = _ClaudeSDKError
    module.CLINotFoundError = _CLINotFoundError
    module.captured = captured
    return module


@pytest.fixture()
def sdk(monkeypatch):
    """Install a stub SDK returning a plain text answer; yields the module."""

    def install(messages=None):
        if messages is None:
            messages = [
                _AssistantMessage([_TextBlock("thinking out loud")]),
                _ResultMessage(result="final report"),
            ]
        module = _fake_sdk(messages)
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
        return module

    return install


def _llm(**kwargs):
    return create_llm_client(
        "claude_code", kwargs.pop("model", "claude-opus-4-8"), **kwargs
    ).get_llm()


@pytest.mark.unit
def test_factory_routes_claude_code():
    client = create_llm_client("claude_code", "claude-opus-4-8")
    assert type(client).__name__ == "ClaudeCodeClient"


@pytest.mark.unit
def test_sdk_options_are_forwarded_and_api_kwargs_dropped(sdk):
    sdk()
    llm = _llm(effort="high", max_turns=5, temperature=0.7, api_key="unused", max_retries=3)
    assert llm.effort == "high"
    assert llm.max_turns == 5
    # Sampling/retry/auth kwargs have no SDK equivalent and must not leak into
    # ChatClaudeCode as stray attributes.
    assert not hasattr(llm, "temperature")
    llm.invoke("hi")
    options = sys.modules["claude_agent_sdk"].captured["options"]
    assert options["effort"] == "high"
    assert options["max_turns"] == 5
    assert options["model"] == "claude-opus-4-8"


@pytest.mark.unit
def test_missing_sdk_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # forces ImportError
    with pytest.raises(ClaudeCodeUnavailableError, match="claude setup-token"):
        _llm().invoke("hi")


@pytest.mark.unit
def test_render_messages_splits_system_and_flattens_turns():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    system, prompt = render_messages([SystemMessage("be terse"), HumanMessage("analyze SPY")])
    assert system == "be terse"
    # A lone human turn is passed through unlabeled.
    assert prompt == "analyze SPY"

    _, prompt = render_messages([HumanMessage("bull case"), AIMessage("bear case")])
    assert prompt == "Human: bull case\n\nAssistant: bear case"


@pytest.mark.unit
def test_no_builtin_tools_and_user_settings_ignored(sdk):
    sdk()
    _llm().invoke("hi")
    options = sys.modules["claude_agent_sdk"].captured["options"]
    # No Read/Write/Bash: this is an analysis run, not a coding session.
    assert options["tools"] == []
    # A run must not pick up the user's CLAUDE.md / settings / skills.
    assert options["setting_sources"] is None
    # An API key in the environment would put the CLI back on metered billing.
    assert options["env"]["ANTHROPIC_API_KEY"] == ""
    assert options["env"]["ANTHROPIC_BASE_URL"] == ""


@pytest.mark.unit
def test_bound_tools_become_mcp_tools_and_message_has_no_tool_calls(sdk):
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def get_stock_data(symbol: str) -> str:
        """Fetch prices."""
        return f"prices for {symbol}"

    module = sdk()
    result = _llm().bind_tools([get_stock_data]).invoke("analyze SPY")

    options = module.captured["options"]
    assert list(options["mcp_servers"]) == ["tradingagents"]
    assert options["allowed_tools"] == ["mcp__tradingagents__get_stock_data"]
    # Headless runs cannot answer a permission prompt.
    assert options["permission_mode"] == "bypassPermissions"
    # The graph invariant: Claude Code ran the tool loop itself, so the analyst
    # node sees a finished report and ConditionalLogic routes to the clear node
    # instead of a ToolNode.
    assert result.tool_calls == []
    assert result.content == "final report"


@pytest.mark.unit
def test_mcp_tool_handler_bridges_to_langchain_tool(sdk):
    import asyncio

    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def get_stock_data(symbol: str) -> str:
        """Fetch prices."""
        if symbol == "BAD":
            raise ValueError("no such ticker")
        return f"prices for {symbol}"

    module = sdk()
    _llm().bind_tools([get_stock_data]).invoke("analyze SPY")
    handler = module.captured["mcp_tools"][0].handler

    ok = asyncio.run(handler({"symbol": "SPY"}))
    assert ok["content"][0]["text"] == "prices for SPY"

    # A vendor failure is reported back to the model, not raised into the graph.
    failed = asyncio.run(handler({"symbol": "BAD"}))
    assert failed["is_error"] is True
    assert "no such ticker" in failed["content"][0]["text"]


class _Decision(BaseModel):
    action: str
    confidence: float


@pytest.mark.unit
def test_structured_output_uses_json_schema_and_parses(sdk):
    module = sdk(
        [
            _ResultMessage(
                result='{"action": "BUY", "confidence": 0.8}',
                structured_output={"action": "BUY", "confidence": 0.8},
            )
        ]
    )
    parsed = _llm().with_structured_output(_Decision).invoke("decide")

    assert module.captured["options"]["output_format"]["type"] == "json_schema"
    assert isinstance(parsed, _Decision)
    assert parsed.action == "BUY"


@pytest.mark.unit
def test_structured_output_falls_back_to_parsing_text(sdk):
    """A run that returns prose-wrapped JSON but no structured payload still parses."""
    sdk([_ResultMessage(result='{"action": "SELL", "confidence": 0.1}')])
    parsed = _llm().with_structured_output(_Decision).invoke("decide")
    assert parsed.action == "SELL"


@pytest.mark.unit
def test_usage_limit_raises_dedicated_error(sdk):
    sdk([_ResultMessage(result="Claude usage limit reached", is_error=True)])
    with pytest.raises(ClaudeCodeUsageLimitError):
        _llm().invoke("hi")


@pytest.mark.unit
def test_result_message_error_raises(sdk):
    sdk([_ResultMessage(result="transport exploded", is_error=True)])
    with pytest.raises(RuntimeError, match="transport exploded"):
        _llm().invoke("hi")


@pytest.mark.unit
def test_falls_back_to_assistant_text_when_result_empty(sdk):
    sdk([_AssistantMessage([_TextBlock("partial answer")]), _ResultMessage(result="")])
    assert _llm().invoke("hi").content == "partial answer"
