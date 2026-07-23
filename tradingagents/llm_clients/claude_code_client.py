"""Client for Claude models accessed through a Claude Code subscription.

Authentication is the Claude Code login (``claude setup-token`` /
``CLAUDE_CODE_OAUTH_TOKEN``), not ``ANTHROPIC_API_KEY`` — no per-token API
billing. See ``claude_code_chat`` for how the SDK's agentic loop is mapped
onto the LangChain chat interface.
"""

from typing import Any

from .base_client import BaseLLMClient
from .validators import validate_model

# Options the graph forwards that map onto SDK options. Cross-provider kwargs
# with no SDK equivalent (``temperature``, ``timeout``, ``max_retries``,
# ``api_key``, ``max_tokens``) are dropped rather than passed through: the CLI
# owns sampling and retry behaviour for a subscription session.
_PASSTHROUGH_KWARGS = ("max_turns", "effort", "cwd")


class ClaudeCodeClient(BaseLLMClient):
    """Client for Claude models served by the local Claude Code CLI."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        """Return a configured ``ChatClaudeCode`` instance."""
        from .claude_code_chat import ChatClaudeCode  # noqa: PLC0415 - lazy: optional dep

        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {"model": self.model}
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and self.kwargs[key] is not None:
                llm_kwargs[key] = self.kwargs[key]
        return ChatClaudeCode(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the claude_code provider."""
        return validate_model("claude_code", self.model)
