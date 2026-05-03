"""LLM provider abstraction.

The strategy + replay code never imports `anthropic` directly — it
goes through `LLMProvider`. That gives us a single seam to swap
backends (OpenAI, Bedrock, OpenRouter) in v0.3 without touching the
strategies or prompts.

For v0.x, the only built-in provider is Anthropic Sonnet, which is
the default. Users who want a different backend implement
`LLMProvider` and pass an instance into the pipeline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """One LLM call's result, model-agnostic."""
    text: str                          # the message body
    input_tokens: int | None = None    # if the provider reports them
    output_tokens: int | None = None
    model: str | None = None           # actual model used (provider may default)


# ── Base class ──────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Single-method interface for chat-completion-style LLM calls.

    Implementations should be stateless wrappers — config (API key,
    model, timeout) is set in `__init__`, the call itself is pure.

    The library only ever uses the system+user-message pattern; we
    don't expose tool-use, streaming, multi-turn chat, or embeddings.
    Strategies that need richer behavior would subclass this with
    additional methods, but v0.x doesn't.
    """

    @abstractmethod
    def call(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Send one (system, user) pair and return the response.

        Raises on API/network errors — the orchestrator catches and
        treats failures as per-step errors (skip that target, move
        on to the next).
        """


# ── Anthropic default ───────────────────────────────────────────────────────

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicLLMProvider(LLMProvider):
    """Default Anthropic Sonnet provider.

    Reads `ANTHROPIC_API_KEY` from the environment. Pass a custom
    `model` to swap to Haiku or Opus. The library defaults to
    Sonnet 4.5 because most calls (program / propose / critic /
    judge) want the strongest reasoning available; Haiku is a
    reasonable fallback for the cheapest steps once you've tuned.
    """

    def __init__(
        self,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        client=None,                    # for tests; defaults to a real anthropic.Anthropic
    ):
        self.model = model

        if client is None:
            import anthropic   # imported lazily so tests can stub
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Either set the env var or pass api_key= to AnthropicLLMProvider."
                )
            self._client = anthropic.Anthropic(api_key=key)
        else:
            self._client = client

    def call(self, *, system: str, user: str, max_tokens: int) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            model=getattr(resp, "model", self.model),
        )


# ── Default factory used by the orchestrator when no provider is supplied ──

def default_llm_provider() -> LLMProvider:
    """Build the default provider from environment.

    Currently always Anthropic. v0.3 will look at an
    `AUTORESEARCH_LLM_PROVIDER` env var to pick between providers.
    """
    return AnthropicLLMProvider()
