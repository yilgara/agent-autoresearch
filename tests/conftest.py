"""Shared pytest fixtures.

Most tests need a fake `LLMProvider` (since the real one needs an API
key + network) and small data fixtures. We bake those here so every
test file can `pytest` without external setup.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pytest

from agent_autoresearch.core.data import (
    Conversation,
    Evidence,
    Target,
    ToolCall,
    Turn,
)
from agent_autoresearch.core.llm import LLMProvider, LLMResponse


# ── Fake LLM provider ───────────────────────────────────────────────────────

@dataclass
class FakeLLM(LLMProvider):
    """`LLMProvider` that returns canned responses in order.

    Pass an iterable of strings (or `LLMResponse`s) at construction;
    each `call()` pops the next one. Records every call in `.calls`
    so tests can assert on prompts.

    Once exhausted, raises so an unexpected extra call doesn't pass
    silently.
    """
    queued: deque[str | LLMResponse] = field(default_factory=deque)
    calls: list[dict] = field(default_factory=list)

    def __init__(self, responses: Iterable[str | LLMResponse] = ()):
        self.queued = deque(responses)
        self.calls = []

    def push(self, response: str | LLMResponse) -> None:
        """Append a response to the queue (useful mid-test)."""
        self.queued.append(response)

    def call(self, *, system: str, user: str, max_tokens: int) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if not self.queued:
            raise AssertionError(
                f"FakeLLM ran out of canned responses after {len(self.calls)} call(s). "
                f"Last user prompt (truncated): {user[:200]}"
            )
        nxt = self.queued.popleft()
        if isinstance(nxt, LLMResponse):
            return nxt
        return LLMResponse(text=nxt, input_tokens=10, output_tokens=20, model="fake")


@pytest.fixture
def fake_llm():
    """Empty FakeLLM — tests push responses as needed."""
    return FakeLLM()


# ── Data factories ──────────────────────────────────────────────────────────

@pytest.fixture
def make_conversation():
    """Factory for tiny Conversation objects."""
    def _make(session_id: str, n_turns: int = 1) -> Conversation:
        return Conversation(
            session_id=session_id,
            turns=[
                Turn(turn=i, user=f"q{i}", agent=f"a{i}")
                for i in range(1, n_turns + 1)
            ],
        )
    return _make


@pytest.fixture
def make_target():
    """Factory for Target objects with sane defaults."""
    def _make(
        skill_name: str,
        *,
        fix_ids: list[str] = None,
        baseline_ids: list[str] = None,
        evidence_summary: str = "demo failure",
        focus_turn: int = 1,
    ) -> Target:
        fix_ids = fix_ids or []
        return Target(
            skill_name=skill_name,
            fix_session_ids=fix_ids,
            regression_baseline_ids=baseline_ids or [],
            evidence=[
                Evidence(
                    category="demo",
                    details={
                        "summary": evidence_summary,
                        "session_id": fix_ids[0] if fix_ids else "",
                        "focus_turn": focus_turn,
                    },
                ),
            ] if fix_ids else [],
        )
    return _make


# ── CSV + transcripts on disk ───────────────────────────────────────────────

@pytest.fixture
def tiny_csv_data(tmp_path: Path) -> dict:
    """Minimal CSV + 4-session transcripts on disk.

    Returns a dict with keys: `csv`, `transcripts_dir`, `tmp_path`.

      sess_001 — find-restaurant fail (vegan ignored, turn 2)
      sess_002 — find-restaurant fail (refused, turn 1)
      sess_003 — find-restaurant pass (baseline)
      sess_004 — multi-skill: find-restaurant pass + book-table fail (turn 2)

    Five rows total. Designed to exercise the multi-skill code path.
    """
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()

    # sess_001 — fail on turn 2
    (transcripts_dir / "sess_001.jsonl").write_text(
        json.dumps({"turn": 1, "user": "find dinner", "agent": "preferences?"}) + "\n"
        + json.dumps({"turn": 2, "user": "vegan", "agent": "Bob's Steakhouse",
                      "tool_calls": [{"name": "search", "args": {"q": "downtown"}}]}) + "\n",
        encoding="utf-8",
    )

    # sess_002 — fail on turn 1
    (transcripts_dir / "sess_002.jsonl").write_text(
        json.dumps({"turn": 1, "user": "find food", "agent": "check Yelp"}) + "\n",
        encoding="utf-8",
    )

    # sess_003 — pass (single turn). Use .json this time to cover both formats.
    (transcripts_dir / "sess_003.json").write_text(
        json.dumps({"session_id": "sess_003", "turns": [
            {"turn": 1, "user": "vegan dinner?", "agent": "Green Leaf",
             "tool_calls": [{"name": "search", "args": {"q": "vegan"}}]},
        ]}),
        encoding="utf-8",
    )

    # sess_004 — multi-skill, fail on book-table turn 2
    (transcripts_dir / "sess_004.jsonl").write_text(
        json.dumps({"turn": 1, "user": "find vegan", "agent": "Green Leaf"}) + "\n"
        + json.dumps({"turn": 2, "user": "book it", "agent": "Booked!",
                      "tool_calls": [{"name": "book", "args": {"date": "05/05/2026"},
                                      "error": "iso required"}]}) + "\n",
        encoding="utf-8",
    )

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "session_id,skill,score,failure_category,failure_summary,failure_turn,transcript_path\n"
        "sess_001,find-restaurant,fail,ignored_constraint,vegan ignored,2,\n"
        "sess_002,find-restaurant,fail,refused,refused yelp,1,\n"
        "sess_003,find-restaurant,pass,,,,\n"
        "sess_004,find-restaurant,pass,,,,\n"
        "sess_004,book-table,fail,wrong_args,date format,2,\n",
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "transcripts_dir": transcripts_dir,
        "tmp_path": tmp_path,
    }


__all__ = ["FakeLLM"]
