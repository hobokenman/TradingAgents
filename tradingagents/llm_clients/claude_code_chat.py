"""LangChain chat model backed by the Claude Agent SDK (Claude Code subscription).

Every other provider in this package speaks a chat-completion API: send
messages, get back either text or a ``tool_calls`` array that LangGraph's
``ToolNode`` executes. The Claude Agent SDK is different — it drives the local
``claude`` binary using the Claude Code subscription credentials and runs its
*own* agentic loop, executing tools itself and returning only the finished
answer.

Two consequences shape this adapter:

1. ``bind_tools`` cannot hand tool calls back to the graph. Instead the bound
   LangChain tools are registered as an in-process MCP server and Claude Code
   runs the whole tool loop internally. The returned ``AIMessage`` therefore
   always has an empty ``tool_calls`` list, which the analyst nodes and
   ``ConditionalLogic.should_continue_*`` already treat as "done" — so the
   ``tools_*`` branches simply never fire for this provider and the graph
   topology needs no changes.
2. ``with_structured_output`` maps onto the SDK's ``output_format`` json_schema
   option; the parsed object comes back on ``ResultMessage.structured_output``.

Auth comes from the Claude Code login (``claude setup-token`` /
``CLAUDE_CODE_OAUTH_TOKEN``), not from ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Name of the in-process MCP server that carries the bound LangChain tools.
# Claude Code addresses its tools as ``mcp__<server>__<tool>``.
MCP_SERVER_NAME = "tradingagents"

# A tool-using analyst needs one turn per tool call plus a final write-up.
# Unbounded loops burn subscription quota, so cap by default; override with
# ``max_turns``.
DEFAULT_MAX_TURNS = 40

# The CLI prefers an API key / gateway redirect over the claude.ai login when
# one is present in the environment — which would quietly put this provider back
# on metered API billing, the exact thing it exists to avoid. The SDK merges
# ``options.env`` over the inherited environment, so blanking these forces the
# subscription credential path. (An ``ANTHROPIC_API_KEY`` set for the
# ``anthropic`` provider is common in this project's ``.env``.)
_SUBSCRIPTION_ENV_OVERRIDES = {
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_AUTH_TOKEN": "",
    "ANTHROPIC_BASE_URL": "",
}

_INSTALL_HINT = (
    "The claude_code provider requires the Claude Agent SDK and the Claude Code "
    "CLI. Install with: uv sync --extra claude-code (or pip install "
    "'tradingagents[claude-code]'), then authenticate with `claude setup-token`."
)


class ClaudeCodeUnavailableError(RuntimeError):
    """The Claude Agent SDK or the ``claude`` CLI is not usable."""


class ClaudeCodeUsageLimitError(RuntimeError):
    """The Claude Code subscription usage limit was reached.

    Raised as a distinct type so callers can tell "you are out of quota for
    this window" apart from an ordinary provider failure — the two need very
    different responses and only one of them is worth retrying.
    """


def _import_sdk():
    """Import ``claude_agent_sdk``, or raise with an actionable message."""
    try:
        import claude_agent_sdk  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ClaudeCodeUnavailableError(_INSTALL_HINT) from exc
    return claude_agent_sdk


def _text_of(content: Any) -> str:
    """Flatten LangChain message content (str or typed blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


_ROLE_LABELS = {
    "human": "Human",
    "ai": "Assistant",
    "tool": "Tool result",
    "function": "Tool result",
}


def render_messages(messages: Sequence[BaseMessage]) -> tuple[str | None, str]:
    """Split LangChain messages into (system prompt, single user prompt).

    ``query()`` takes one prompt string, so the non-system turns are rendered
    into a transcript. Every agent node in this project is effectively one-shot
    — the context lives in the prompt, not in a multi-turn history — so the
    flattening is lossless in practice.
    """
    system_parts: list[str] = []
    turns: list[str] = []

    for message in messages:
        text = _text_of(message.content)
        if not text:
            continue
        if message.type == "system":
            system_parts.append(text)
        else:
            label = _ROLE_LABELS.get(message.type, message.type.capitalize())
            turns.append(f"{label}: {text}")

    system_prompt = "\n\n".join(system_parts) or None
    # A lone human turn needs no role label — it reads as a plain instruction.
    prompt = turns[0].split(": ", 1)[-1] if len(turns) == 1 else "\n\n".join(turns)
    return system_prompt, prompt


def _tool_input_schema(lc_tool: Any) -> dict[str, Any]:
    """Best-effort JSON Schema for a LangChain tool's arguments."""
    for attr in ("tool_call_schema", "args_schema"):
        schema = getattr(lc_tool, attr, None)
        if schema is None:
            continue
        if isinstance(schema, dict):
            return schema
        model_json_schema = getattr(schema, "model_json_schema", None)
        if callable(model_json_schema):
            return model_json_schema()
    return {"type": "object", "properties": {}}


def _to_sdk_tool(lc_tool: Any):
    """Wrap a LangChain tool as an in-process SDK MCP tool.

    The LangChain tools in this project are synchronous and do blocking I/O
    (HTTP calls to data vendors), so the handler offloads to a worker thread
    rather than stalling the SDK's event loop.
    """
    sdk = _import_sdk()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(lc_tool.invoke, args)
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as exc:  # surfaced to the model, not fatal to the run
            logger.warning("claude_code: tool %s failed: %s", lc_tool.name, exc)
            return {
                "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                "is_error": True,
            }

    return sdk.tool(lc_tool.name, lc_tool.description or lc_tool.name, _tool_input_schema(lc_tool))(
        handler
    )


@dataclass
class _SdkResult:
    """What one SDK query produced."""

    text: str = ""
    structured: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    num_turns: int | None = None
    session_id: str | None = None
    # Subscription window state (five-hour / weekly), reported on every call.
    rate_limit: Any = None


def _is_usage_limit(message: str) -> bool:
    lowered = message.lower()
    return "usage limit" in lowered or "rate limit" in lowered or "quota" in lowered


async def _query(prompt: str, options: Any) -> _SdkResult:
    """Run one SDK query to completion and collect the final answer."""
    sdk = _import_sdk()
    collected = _SdkResult()
    assistant_text: list[str] = []

    try:
        async for message in sdk.query(prompt=prompt, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        assistant_text.append(block.text)
            elif isinstance(message, sdk.RateLimitEvent):
                info = message.rate_limit_info
                collected.rate_limit = info
                # Every call reports the current subscription window; only an
                # abnormal status is worth surfacing above debug level.
                status = getattr(info, "status", None)
                level = logging.DEBUG if status == "allowed" else logging.WARNING
                logger.log(level, "claude_code: subscription rate limit: %s", info)
            elif isinstance(message, sdk.ResultMessage):
                if message.is_error:
                    detail = message.result or message.subtype or "unknown error"
                    if _is_usage_limit(str(detail)):
                        raise ClaudeCodeUsageLimitError(str(detail))
                    raise RuntimeError(f"Claude Code run failed: {detail}")
                collected.text = message.result or ""
                collected.structured = message.structured_output
                collected.usage = message.usage or {}
                collected.num_turns = message.num_turns
                collected.session_id = message.session_id
    except sdk.CLINotFoundError as exc:
        raise ClaudeCodeUnavailableError(_INSTALL_HINT) from exc
    except sdk.ClaudeSDKError as exc:
        if _is_usage_limit(str(exc)):
            raise ClaudeCodeUsageLimitError(str(exc)) from exc
        raise

    # ``ResultMessage.result`` is the authoritative final text; fall back to the
    # concatenated assistant turns if the CLI omitted it.
    if not collected.text:
        collected.text = "\n".join(assistant_text)
    return collected


def _run_sync(coro) -> _SdkResult:
    """Run an async SDK query from sync code.

    The graph invokes every node synchronously, and some hosts (notebooks,
    async CLIs) already have a running event loop — so the query always gets a
    fresh loop on a worker thread instead of assuming it can own the current one.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="claude-code") as pool:
        return pool.submit(asyncio.run, coro).result()


class ChatClaudeCode(BaseChatModel):
    """Chat model that runs prompts through the Claude Code subscription.

    ``bind_tools`` registers tools with an in-process MCP server rather than
    returning tool calls to the caller — see the module docstring.
    """

    model: str = "claude-sonnet-5"
    max_turns: int | None = DEFAULT_MAX_TURNS
    effort: str | None = None
    cwd: str | None = None
    # Bound LangChain tools, exposed to Claude Code as MCP tools.
    lc_tools: list[Any] = []
    # Pydantic schema for structured output, set by ``with_structured_output``.
    structured_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "max_turns": self.max_turns, "effort": self.effort}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ChatClaudeCode:
        """Attach tools for Claude Code to execute inside its own loop."""
        if kwargs:
            logger.debug("claude_code: ignoring unsupported bind_tools kwargs: %s", sorted(kwargs))
        return self.model_copy(update={"lc_tools": list(tools)})

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Return a runnable that emits a validated instance of ``schema``.

        Only Pydantic model schemas are supported, which is all this project
        uses (see ``agents/utils/structured.py``).
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise NotImplementedError(
                "claude_code structured output requires a Pydantic model schema"
            )
        if include_raw:
            raise NotImplementedError("claude_code does not support include_raw")

        bound = self.model_copy(update={"structured_schema": schema})

        def parse(message: AIMessage) -> BaseModel:
            payload = message.additional_kwargs.get("structured_output")
            if payload is None:
                # No structured payload came back (e.g. the model answered in
                # prose). Try the text as JSON, then let the caller's free-text
                # fallback handle it.
                payload = json.loads(_text_of(message.content))
            if isinstance(payload, str):
                payload = json.loads(payload)
            return schema.model_validate(payload)

        return bound | RunnableLambda(parse)

    def _build_options(self, system_prompt: str | None) -> Any:
        sdk = _import_sdk()
        options: dict[str, Any] = {
            "model": self.model,
            # No built-in Read/Write/Bash: this is an analysis run, not a coding
            # session, and the only capabilities it should have are our tools.
            "tools": [],
            # Ignore the user's CLAUDE.md, settings, and skills so a run behaves
            # the same regardless of the directory it is launched from.
            "setting_sources": None,
            "max_turns": self.max_turns,
            "env": dict(_SUBSCRIPTION_ENV_OVERRIDES),
        }
        if system_prompt:
            options["system_prompt"] = system_prompt
        if self.effort:
            options["effort"] = self.effort
        if self.cwd:
            options["cwd"] = self.cwd
        if self.lc_tools:
            sdk_tools = [_to_sdk_tool(t) for t in self.lc_tools]
            options["mcp_servers"] = {
                MCP_SERVER_NAME: sdk.create_sdk_mcp_server(MCP_SERVER_NAME, tools=sdk_tools)
            }
            options["allowed_tools"] = [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in self.lc_tools]
            # Headless: nothing can answer a permission prompt, and the tool set
            # is limited to our own read-only data tools.
            options["permission_mode"] = "bypassPermissions"
        if self.structured_schema is not None:
            options["output_format"] = {
                "type": "json_schema",
                "schema": self.structured_schema.model_json_schema(),
            }
        return sdk.ClaudeAgentOptions(**options)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            logger.debug("claude_code: `stop` sequences are not supported; ignoring")

        system_prompt, prompt = render_messages(messages)
        result = _run_sync(_query(prompt, self._build_options(system_prompt)))

        additional_kwargs: dict[str, Any] = {}
        if result.structured is not None:
            additional_kwargs["structured_output"] = result.structured

        message = AIMessage(
            content=result.text,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "model": self.model,
                "num_turns": result.num_turns,
                "session_id": result.session_id,
                "usage": result.usage,
                "rate_limit": result.rate_limit,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__: list[str] = [
    "ChatClaudeCode",
    "ClaudeCodeUnavailableError",
    "ClaudeCodeUsageLimitError",
    "MCP_SERVER_NAME",
    "render_messages",
]
