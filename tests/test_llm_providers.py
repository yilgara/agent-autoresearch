"""Tests for the LLM provider layer.

Both Anthropic and OpenAI providers are exercised with FAKE clients so
no API key or network is required. Validates:

  - call shape (system, user, max_tokens) → LLMResponse mapping
  - token-usage normalization across providers
  - construction errors when API key is missing
  - `default_llm_provider()` dispatch via name + env var
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from agent_autoresearch.core.llm import (
    AnthropicLLMProvider,
    LLMResponse,
    OpenAILLMProvider,
    default_llm_provider,
)


# ── Fake clients ────────────────────────────────────────────────────────────

class _FakeAnthropicMessages:
    def __init__(self, response_text="hello", model="claude-fake"):
        self.response_text = response_text
        self.model = model
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.response_text)],
            usage=SimpleNamespace(input_tokens=11, output_tokens=22),
            model=self.model,
        )


class _FakeAnthropicClient:
    def __init__(self, response_text="hello"):
        self.messages = _FakeAnthropicMessages(response_text)


class _FakeOpenAICompletions:
    def __init__(self, response_text="hi", model="gpt-fake"):
        self.response_text = response_text
        self.model = model
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.response_text),
            )],
            usage=SimpleNamespace(prompt_tokens=33, completion_tokens=44),
            model=self.model,
        )


class _FakeOpenAIClient:
    def __init__(self, response_text="hi"):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(response_text))


# ── Anthropic provider ──────────────────────────────────────────────────────

class TestAnthropicProvider:
    def test_call_returns_llm_response_with_tokens(self):
        provider = AnthropicLLMProvider(client=_FakeAnthropicClient("anthropic-says-hi"))
        resp = provider.call(system="sys", user="usr", max_tokens=100)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "anthropic-says-hi"
        assert resp.input_tokens == 11
        assert resp.output_tokens == 22
        assert resp.model == "claude-fake"

    def test_call_passes_args_to_client(self):
        client = _FakeAnthropicClient("ok")
        provider = AnthropicLLMProvider(client=client)
        provider.call(system="my system", user="my user", max_tokens=200)
        kw = client.messages.calls[0]
        assert kw["max_tokens"] == 200
        assert kw["system"] == "my system"
        assert kw["messages"] == [{"role": "user", "content": "my user"}]

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicLLMProvider()


# ── OpenAI provider ─────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_call_returns_llm_response_with_tokens(self):
        provider = OpenAILLMProvider(client=_FakeOpenAIClient("openai-says-hi"))
        resp = provider.call(system="sys", user="usr", max_tokens=100)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "openai-says-hi"
        # Token usage normalised: prompt_tokens → input_tokens, completion_tokens → output_tokens
        assert resp.input_tokens == 33
        assert resp.output_tokens == 44
        assert resp.model == "gpt-fake"

    def test_messages_format_matches_openai_chat_api(self):
        client = _FakeOpenAIClient("ok")
        provider = OpenAILLMProvider(client=client)
        provider.call(system="my system", user="my user", max_tokens=300)
        kw = client.chat.completions.calls[0]
        assert kw["max_tokens"] == 300
        # OpenAI uses BOTH a system and a user message in the messages array
        assert kw["messages"] == [
            {"role": "system", "content": "my system"},
            {"role": "user",   "content": "my user"},
        ]

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # We pass client=None to force the openai-import path... but we
        # still want to trip the API-key check, not the import. Simulate by
        # also setting client=None and intercepting the openai import.
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAILLMProvider()

    def test_empty_choices_returns_empty_text(self):
        """Defensive: if OpenAI returns no choices, we shouldn't crash."""
        class _EmptyOpenAI:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        return SimpleNamespace(
                            choices=[],
                            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                            model="x",
                        )
                completions = _Completions()
            chat = _Chat()
        provider = OpenAILLMProvider(client=_EmptyOpenAI())
        resp = provider.call(system="s", user="u", max_tokens=10)
        assert resp.text == ""


# ── default_llm_provider dispatch ───────────────────────────────────────────

class TestDefaultLLMProvider:
    def test_no_arg_returns_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = default_llm_provider()
        assert isinstance(provider, AnthropicLLMProvider)

    def test_explicit_name_picks_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = default_llm_provider("anthropic")
        assert isinstance(provider, AnthropicLLMProvider)

    def test_env_var_is_NOT_consulted(self, monkeypatch):
        """Regression test: AUTORESEARCH_LLM_PROVIDER must not influence
        which provider gets picked. CLI flag is the only override."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        monkeypatch.setenv("AUTORESEARCH_LLM_PROVIDER", "openai")  # ignored
        provider = default_llm_provider()
        assert isinstance(provider, AnthropicLLMProvider)

    def test_unknown_provider_raises_with_valid_options(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        with pytest.raises(RuntimeError, match="Unknown LLM provider"):
            default_llm_provider("bedrock")

    def test_alias_claude_resolves_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = default_llm_provider("claude")
        assert isinstance(provider, AnthropicLLMProvider)

    def test_alias_gpt_resolves_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        try:
            provider = default_llm_provider("gpt")
        except RuntimeError as exc:
            if "openai" in str(exc).lower():
                pytest.skip("openai SDK not installed in this environment")
            raise
        assert isinstance(provider, OpenAILLMProvider)

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        provider = default_llm_provider("ANTHROPIC")
        assert isinstance(provider, AnthropicLLMProvider)
