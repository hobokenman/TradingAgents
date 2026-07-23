"""Client for OpenAI models accessed through a Codex subscription.

Authentication comes from ``codex login`` (a ChatGPT plan), not an OpenAI API
key.  ``ChatCodex`` adapts the local Codex runtime to LangChain's chat-model
contract while keeping tool execution in the existing LangGraph workflow.
"""

from typing import Any

from .base_client import BaseLLMClient
from .validators import validate_model

_PASSTHROUGH_KWARGS = ("effort", "cwd")


class CodexClient(BaseLLMClient):
    """Client for models served by the local Codex subscription runtime."""

    provider = "codex"

    def get_llm(self) -> Any:
        """Return a configured ``ChatCodex`` instance."""
        from .codex_chat import ChatCodex  # noqa: PLC0415 - optional dependency

        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and self.kwargs[key] is not None:
                llm_kwargs[key] = self.kwargs[key]
        return ChatCodex(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the Codex provider."""
        return validate_model("codex", self.model)
