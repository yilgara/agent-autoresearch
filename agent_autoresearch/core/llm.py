"""LLM provider abstraction.

The strategy + replay code never imports `anthropic` or `openai` directly
— it goes through `LLMProvider`. That gives us a single seam to swap
backends (or use mocks in tests) without touching strategies or prompts.

Two providers ship with the library:
  - `AnthropicLLMProvider`  (default — Sonnet 4.5)
  - `OpenAILLMProvider`     (opt-in — GPT-4o)

Pick which one runs by either:
  - passing `--llm-provider openai` on the CLI, or
  - constructing the provider directly and passing `llm=` to
    `run_pipeline()` from Python.

To add a new provider (Bedrock, OpenRouter, etc.), subclass
`LLMProvider` and register it in `_PROVIDER_ALIASES`.
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


# ── Anthropic provider ──────────────────────────────────────────────────────

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicLLMProvider(LLMProvider):
    """Default Anthropic Sonnet provider.

    Reads `ANTHROPIC_API_KEY` from the environment. Pass a custom
    `model` to swap to Haiku or Opus.
    """

    def __init__(
        self,
        model: str = _DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        client=None,                    # for tests; defaults to anthropic.Anthropic
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


# ── OpenAI provider ─────────────────────────────────────────────────────────

_DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAILLMProvider(LLMProvider):
    """OpenAI GPT-4o provider (opt-in).

    Reads `OPENAI_API_KEY` from the environment. To use this provider:

        pip install agent-autoresearch[openai]    # installs the openai SDK

        # then either pass --llm-provider openai on the CLI, or
        # instantiate directly:
        from agent_autoresearch.core.llm import OpenAILLMProvider
        provider = OpenAILLMProvider(model="gpt-4o-mini")

    The system + user pair is sent as the standard two-message chat
    completion. Token usage is mapped from OpenAI's
    `prompt_tokens`/`completion_tokens` to the library's
    `input_tokens`/`output_tokens` for cross-provider compatibility.

    Note: this provider uses the standard `chat.completions.create`
    endpoint with a `max_tokens` parameter — it works for gpt-4o,
    gpt-4-turbo, gpt-4, gpt-3.5-turbo. The newer reasoning models
    (o1-preview, o1-mini) use `max_completion_tokens` instead and
    aren't supported here without a custom client; pass `client=`
    if you need to call those.
    """

    def __init__(
        self,
        model: str = _DEFAULT_OPENAI_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client=None,                    # for tests; defaults to openai.OpenAI
    ):
        self.model = model

        if client is None:
            try:
                import openai   # imported lazily — optional dep
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAILLMProvider requires the `openai` package. "
                    "Install it with: pip install agent-autoresearch[openai]"
                ) from exc
            key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Either set the env var or pass api_key= to OpenAILLMProvider."
                )
            kwargs: dict = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
        else:
            self._client = client

    def call(self, *, system: str, user: str, max_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        # OpenAI response: choices[0].message.content
        choice = resp.choices[0] if resp.choices else None
        text = choice.message.content if choice and choice.message else ""
        usage = getattr(resp, "usage", None)
        # Map OpenAI's prompt_tokens/completion_tokens to the library's
        # input_tokens/output_tokens for cross-provider consistency.
        return LLMResponse(
            text=text or "",
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            model=getattr(resp, "model", self.model),
        )


# ── Default factory ─────────────────────────────────────────────────────────

# Names accepted in env / CLI for each provider — comparison is
# case-insensitive and aliases are forgiving.
_PROVIDER_ALIASES: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicLLMProvider,
    "claude":    AnthropicLLMProvider,
    "openai":    OpenAILLMProvider,
    "gpt":       OpenAILLMProvider,
}


def default_llm_provider(name: str | None = None) -> LLMProvider:
    """Build a provider by name, defaulting to Anthropic.

    Pass `name="openai"` (or `"anthropic"`) to pick a specific
    provider. With no argument, returns Anthropic. The CLI's
    `--llm-provider` flag is the user-facing way to override this;
    no env var is consulted.

    Returns a freshly-constructed provider — repeated calls don't
    share the underlying SDK client.
    """
    chosen = (name or "").strip().lower() or "anthropic"
    cls = _PROVIDER_ALIASES.get(chosen)
    if cls is None:
        valid = sorted(set(_PROVIDER_ALIASES.keys()))
        raise RuntimeError(
            f"Unknown LLM provider {chosen!r}. "
            f"Valid options: {', '.join(valid)}."
        )
    return cls()
