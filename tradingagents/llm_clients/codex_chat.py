"""LangChain chat model backed by an OpenAI Codex subscription.

The official ``openai-codex`` SDK launches the bundled Codex runtime and uses
the credentials created by ``codex login``.  This adapter deliberately rejects
API-key accounts and blanks API-key environment variables in the subprocess so
selecting the ``codex`` provider cannot silently fall back to metered API use.

Codex turns are ephemeral and read-only.  Bound LangChain tools are described
to the model through a small structured-output protocol; returned calls are
converted to normal ``AIMessage.tool_calls`` and executed by the project's
existing LangGraph ``ToolNode`` instances.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SUBSCRIPTION_ENV_OVERRIDES = {
    "OPENAI_API_KEY": "",
    "CODEX_API_KEY": "",
}

_INSTALL_HINT = (
    "The codex provider requires the official Codex Python SDK. Install it with "
    "`uv sync --extra codex` (or `pip install 'tradingagents[codex]'`). It reuses "
    "the ChatGPT login from a Codex-capable IDE or Codex CLI; if signed out, "
    "install the Codex CLI separately and run `codex login`."
)

_BASE_INSTRUCTIONS = (
    "You are the language-model engine for TradingAgents, not a coding agent. "
    "Answer only from the conversation supplied in the turn. Do not inspect the "
    "workspace, run commands, edit files, browse the web, or invoke built-in Codex "
    "tools. TradingAgents executes any explicitly listed data tools outside Codex."
)


class CodexUnavailableError(RuntimeError):
    """The Codex SDK or its bundled runtime is not available."""


class CodexAuthenticationError(RuntimeError):
    """Codex is not authenticated with a ChatGPT subscription."""


class CodexUsageLimitError(RuntimeError):
    """The ChatGPT/Codex subscription usage limit was reached."""


@dataclass(frozen=True)
class CodexSubscriptionInfo:
    """Authenticated subscription metadata and its currently available models."""

    email: str | None
    plan_type: str | None
    models: tuple[str, ...]


def _import_sdk():
    try:
        import openai_codex  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise CodexUnavailableError(_INSTALL_HINT) from exc
    return openai_codex


def _value(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from a Pydantic object or a plain test/dict object."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True)
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _langchain_usage_metadata(usage: Any) -> dict[str, int] | None:
    """Map Codex's cumulative token breakdown to LangChain's common fields."""
    if not isinstance(usage, dict):
        return None
    total = usage.get("total", usage)
    if not isinstance(total, dict):
        return None
    input_tokens = total.get("inputTokens", total.get("input_tokens"))
    output_tokens = total.get("outputTokens", total.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    total_tokens = total.get(
        "totalTokens", total.get("total_tokens", int(input_tokens) + int(output_tokens))
    )
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def _subscription_account(response: Any) -> tuple[str | None, str | None]:
    """Validate and unpack ``Codex.account()``.

    The generated SDK wraps the account union in a Pydantic ``RootModel``.
    Keeping this extraction tolerant of dicts also makes the optional provider
    straightforward to test without installing its runtime.
    """
    account = _value(response, "account")
    if account is None:
        raise CodexAuthenticationError(
            "Codex is not logged in. Run `codex login` and choose ChatGPT "
            "subscription authentication."
        )
    account = _value(account, "root", account)
    account_type = _value(account, "type")
    if account_type != "chatgpt":
        label = account_type or "unknown"
        raise CodexAuthenticationError(
            f"The codex provider requires a ChatGPT subscription login, but "
            f"Codex reported account type {label!r}. Run `codex logout`, then "
            "`codex login` and select ChatGPT. OpenAI API-key accounts are "
            "intentionally rejected."
        )
    plan = _value(account, "plan_type", _value(account, "planType"))
    plan_value = _value(plan, "value", plan)
    return _value(account, "email"), str(plan_value) if plan_value is not None else None


def _sdk_config(sdk: Any, cwd: str | None = None) -> Any:
    return sdk.CodexConfig(
        cwd=cwd or os.getcwd(),
        env=dict(_SUBSCRIPTION_ENV_OVERRIDES),
    )


@lru_cache(maxsize=1)
def inspect_codex_subscription() -> CodexSubscriptionInfo:
    """Check subscription authentication and query account-visible models."""
    sdk = _import_sdk()
    try:
        with sdk.Codex(config=_sdk_config(sdk)) as codex:
            email, plan_type = _subscription_account(codex.account())
            response = codex.models()
    except CodexAuthenticationError:
        raise
    except FileNotFoundError as exc:
        raise CodexUnavailableError(_INSTALL_HINT) from exc
    except Exception as exc:
        raise _translate_error(exc) from exc

    models: list[str] = []
    for entry in _value(response, "data", ()) or ():
        if _value(entry, "hidden", False):
            continue
        model = _value(entry, "model") or _value(entry, "id")
        if model and model not in models:
            models.append(str(model))
    return CodexSubscriptionInfo(email, plan_type, tuple(models))


def _text_of(content: Any) -> str:
    """Flatten LangChain message content (string or typed blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


_ROLE_LABELS = {
    "human": "Human",
    "ai": "Assistant",
    "tool": "Tool result",
    "function": "Tool result",
}


def render_messages(messages: Sequence[BaseMessage]) -> tuple[str | None, str]:
    """Split LangChain messages into a developer prompt and turn transcript."""
    system_parts: list[str] = []
    turns: list[str] = []

    for message in messages:
        text = _text_of(message.content)
        if message.type == "system":
            if text:
                system_parts.append(text)
            continue

        label = _ROLE_LABELS.get(message.type, message.type.capitalize())
        parts: list[str] = []
        if text:
            parts.append(text)
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = [
                {"name": call["name"], "args": call["args"], "id": call.get("id")}
                for call in message.tool_calls
            ]
            parts.append(f"Requested tools: {json.dumps(calls, ensure_ascii=False)}")
        if parts:
            turns.append(f"{label}: {' '.join(parts)}")

    system_prompt = "\n\n".join(system_parts) or None
    prompt = turns[0].split(": ", 1)[-1] if len(turns) == 1 else "\n\n".join(turns)
    return system_prompt, prompt


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    for attr in ("tool_call_schema", "args_schema"):
        schema = getattr(tool, attr, None)
        if schema is None:
            continue
        if isinstance(schema, dict):
            return schema
        model_json_schema = getattr(schema, "model_json_schema", None)
        if callable(model_json_schema):
            return model_json_schema()
    return {"type": "object", "properties": {}}


def _tool_protocol_schema(tools: Sequence[Any]) -> dict[str, Any]:
    names = [str(tool.name) for tool in tools]
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["final", "tool_calls"]},
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "arguments_json": {"type": "string"},
                        "id": {"type": "string"},
                    },
                    "required": ["name", "arguments_json", "id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["kind", "content", "tool_calls"],
        "additionalProperties": False,
    }


def _tool_instructions(tools: Sequence[Any]) -> str:
    specs = [
        {
            "name": tool.name,
            "description": tool.description or tool.name,
            "input_schema": _tool_input_schema(tool),
        }
        for tool in tools
    ]
    return (
        "TradingAgents can execute the following external data tools:\n"
        f"{json.dumps(specs, ensure_ascii=False)}\n\n"
        "Return the required JSON object. Use kind=tool_calls when data is "
        "needed, put each tool's arguments in arguments_json as a JSON object "
        "string, and do not invent tool results. Use kind=final only when no "
        "more tool data is needed; then put the answer in content and return an "
        "empty tool_calls array."
    )


def _is_usage_limit(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered for marker in ("usage limit", "rate limit", "quota", "limit reached")
    )


def _is_auth_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ("not logged in", "unauthorized", "authentication", "login required")
    )


def _translate_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    if _is_usage_limit(message):
        return CodexUsageLimitError(message)
    if _is_auth_error(message):
        return CodexAuthenticationError(
            f"{message}. Run `codex login` and authenticate with ChatGPT."
        )
    return RuntimeError(f"Codex run failed: {message}")


@dataclass
class _CodexResult:
    text: str
    thread_id: str
    turn_id: str
    usage: Any
    duration_ms: int | None


def _run_codex(
    *,
    model: str,
    effort: str | None,
    cwd: str | None,
    system_prompt: str | None,
    prompt: str,
    output_schema: dict[str, Any] | None,
) -> _CodexResult:
    sdk = _import_sdk()
    developer_parts = [part for part in (system_prompt,) if part]

    try:
        with sdk.Codex(config=_sdk_config(sdk, cwd)) as codex:
            _subscription_account(codex.account())
            thread = codex.thread_start(
                base_instructions=_BASE_INSTRUCTIONS,
                developer_instructions="\n\n".join(developer_parts) or None,
                ephemeral=True,
                model=model,
                sandbox=sdk.Sandbox.read_only,
            )
            run_kwargs: dict[str, Any] = {
                "output_schema": output_schema,
                "sandbox": sdk.Sandbox.read_only,
            }
            if effort:
                run_kwargs["effort"] = effort
            result = thread.run(prompt, **run_kwargs)
    except (CodexAuthenticationError, CodexUsageLimitError, CodexUnavailableError):
        raise
    except FileNotFoundError as exc:
        raise CodexUnavailableError(_INSTALL_HINT) from exc
    except Exception as exc:
        raise _translate_error(exc) from exc

    text = _value(result, "final_response") or ""
    if not text:
        raise RuntimeError("Codex run completed without a final response.")
    return _CodexResult(
        text=text,
        thread_id=str(_value(thread, "id", "")),
        turn_id=str(_value(result, "id", "")),
        usage=_model_dump(_value(result, "usage")),
        duration_ms=_value(result, "duration_ms"),
    )


class ChatCodex(BaseChatModel):
    """LangChain chat model backed by a ChatGPT-authenticated Codex runtime."""

    model: str = "gpt-5.6-sol"
    effort: str | None = None
    cwd: str | None = None
    lc_tools: list[Any] = Field(default_factory=list)
    structured_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "codex"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ChatCodex:
        """Bind tools while leaving their execution in the LangGraph ToolNodes."""
        if kwargs:
            logger.debug("codex: ignoring unsupported bind_tools kwargs: %s", sorted(kwargs))
        return self.model_copy(update={"lc_tools": list(tools)})

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise NotImplementedError("codex structured output requires a Pydantic model schema")
        if include_raw:
            raise NotImplementedError("codex does not support include_raw")
        if kwargs:
            logger.debug("codex: ignoring unsupported structured-output kwargs: %s", sorted(kwargs))

        bound = self.model_copy(update={"structured_schema": schema})

        def parse(message: AIMessage) -> BaseModel:
            payload = message.additional_kwargs.get("structured_output")
            if payload is None:
                payload = json.loads(_text_of(message.content))
            if isinstance(payload, str):
                payload = json.loads(payload)
            return schema.model_validate(payload)

        return bound | RunnableLambda(parse)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            logger.debug("codex: `stop` sequences are not supported; ignoring")

        system_prompt, prompt = render_messages(messages)
        output_schema: dict[str, Any] | None = None
        if self.lc_tools:
            if self.structured_schema is not None:
                raise NotImplementedError(
                    "codex cannot combine bind_tools() and with_structured_output()"
                )
            tool_prompt = _tool_instructions(self.lc_tools)
            system_prompt = "\n\n".join(part for part in (system_prompt, tool_prompt) if part)
            output_schema = _tool_protocol_schema(self.lc_tools)
        elif self.structured_schema is not None:
            output_schema = self.structured_schema.model_json_schema()

        result = _run_codex(
            model=self.model,
            effort=self.effort,
            cwd=self.cwd,
            system_prompt=system_prompt,
            prompt=prompt,
            output_schema=output_schema,
        )

        content = result.text
        tool_calls: list[dict[str, Any]] = []
        additional_kwargs: dict[str, Any] = {}

        if self.lc_tools:
            try:
                payload = json.loads(result.text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Codex returned an invalid tool protocol response.") from exc
            content = str(payload.get("content", ""))
            if payload.get("kind") == "tool_calls":
                for call in payload.get("tool_calls", []):
                    try:
                        args = json.loads(call["arguments_json"])
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"Codex returned invalid arguments for tool {call.get('name')!r}."
                        ) from exc
                    if not isinstance(args, dict):
                        raise RuntimeError("Codex tool arguments must decode to a JSON object.")
                    tool_calls.append(
                        {
                            "name": call["name"],
                            "args": args,
                            "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                            "type": "tool_call",
                        }
                    )
                if not tool_calls:
                    raise RuntimeError(
                        "Codex requested tool execution without returning any tool calls."
                    )
        elif self.structured_schema is not None:
            additional_kwargs["structured_output"] = json.loads(result.text)

        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
            usage_metadata=_langchain_usage_metadata(result.usage),
            response_metadata={
                "model": self.model,
                "thread_id": result.thread_id,
                "turn_id": result.turn_id,
                "duration_ms": result.duration_ms,
                "usage": result.usage,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__ = [
    "ChatCodex",
    "CodexAuthenticationError",
    "CodexSubscriptionInfo",
    "CodexUnavailableError",
    "CodexUsageLimitError",
    "inspect_codex_subscription",
    "render_messages",
]
