"""Tests for v2's atomic-mutation propose loop.

Covers the per-evidence retry budget, the recursive-rollback chain on
final failure, and the LLM-driven `done` early-exit. Validators are
injected as test fakes so we never call critic/replay directly here —
this file tests the LOOP, not the validators themselves.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_autoresearch.core.data import Conversation, Evidence, Target, Turn
from agent_autoresearch.strategies.v2.propose import (
    MAX_ATTEMPTS_PER_EVIDENCE,
    AtomicAttempt,
    propose,
)


# ── Builders ────────────────────────────────────────────────────────────────

def _evidence(idx: int, session_id: str | None = None) -> Evidence:
    sid = session_id or f"s{idx}"
    return Evidence(
        category=f"cat_{idx}",
        details={
            "summary": f"failure {idx}",
            "session_id": sid,
            "focus_turn": 1,
        },
    )


def _conv(session_id: str) -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(turn=1, user="u", agent="a")],
    )


def _target(n_evidence: int) -> Target:
    return Target(
        skill_name="my-skill",
        evidence=[_evidence(i, f"s{i}") for i in range(n_evidence)],
        fix_session_ids=[f"s{i}" for i in range(n_evidence)],
    )


def _conversations(n: int) -> dict[str, Conversation]:
    return {f"s{i}": _conv(f"s{i}") for i in range(n)}


def _atomic_response(action: str, body: str = "edited content " * 20) -> str:
    """Build an LLM response in the propose XML format."""
    if action == "edit":
        return (
            "<action>edit</action>\n"
            "<reasoning>do the thing</reasoning>\n"
            f"<new_skill_md>\n{body}\n</new_skill_md>"
        )
    elif action == "skip":
        return (
            "<action>skip</action>\n"
            "<reasoning>not needed</reasoning>"
        )
    elif action == "done":
        return (
            "<action>done</action>\n"
            "<reasoning>nothing more to add</reasoning>"
        )
    raise ValueError(action)


def _validators_always_pass():
    """(critic_per_attempt, final_replay) — both pass unconditionally."""
    return (
        lambda cand, cur, ev: (True, ""),
        lambda cand, target, convs: (True, ""),
    )


# ── Happy path ──────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_one_evidence_one_attempt_accepted(self, fake_llm):
        target = _target(1)
        fake_llm.push(_atomic_response("edit", body="v1 " * 100))

        crit_per, rep_final = _validators_always_pass()
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(1),
            critic_per_attempt=crit_per,
            final_replay=rep_final,
            llm=fake_llm,
        )
        assert result.action == "edit"
        assert len(result.accepted_log) == 1
        assert len(result.attempts_log) == 1
        assert "v1" in (result.new_skill_md or "")

    def test_multiple_evidence_each_addressed(self, fake_llm):
        target = _target(3)
        for i in range(3):
            fake_llm.push(_atomic_response("edit", body=f"after-ev{i} " * 50))

        crit_per, rep_final = _validators_always_pass()
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(3),
            critic_per_attempt=crit_per,
            final_replay=rep_final,
            llm=fake_llm,
        )
        assert result.action == "edit"
        assert len(result.accepted_log) == 3
        # Each accepted attempt targeted a different evidence index
        assert sorted(a.evidence_index for a in result.accepted_log) == [0, 1, 2]
        # State accumulated — last attempt's body is the final
        assert "after-ev2" in (result.new_skill_md or "")


# ── Per-evidence retries ────────────────────────────────────────────────────

class TestRetryLogic:
    def test_critic_fails_twice_then_passes(self, fake_llm):
        target = _target(1)
        for _ in range(3):
            fake_llm.push(_atomic_response("edit", body="x " * 100))

        # Critic fails on attempts 1+2, passes on 3
        critic_calls = {"n": 0}
        def crit_per(cand, cur, ev):
            critic_calls["n"] += 1
            if critic_calls["n"] < 3:
                return (False, "concern X")
            return (True, "")

        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(1),
            critic_per_attempt=crit_per,
            final_replay=lambda *a, **kw: (True, ""),
            llm=fake_llm,
        )
        assert result.action == "edit"
        assert len(result.accepted_log) == 1
        assert len(result.attempts_log) == 3        # 2 retries + 1 success
        assert result.attempts_log[0].failure_reason.startswith("Critic rejected")
        assert result.attempts_log[2].accepted is True

    def test_all_three_attempts_fail_moves_to_next_evidence(self, fake_llm):
        target = _target(2)
        # 3 failed attempts for ev0 + 1 successful for ev1 = 4 LLM calls
        for _ in range(4):
            fake_llm.push(_atomic_response("edit", body="z " * 100))

        # Critic always fails for ev0, always passes for ev1
        def crit_per(cand, cur, ev):
            if ev.category == "cat_0":
                return (False, "no good")
            return (True, "")

        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(2),
            critic_per_attempt=crit_per,
            final_replay=lambda *a, **kw: (True, ""),
            llm=fake_llm,
        )
        # Only ev1 should be in accepted_log
        assert len(result.accepted_log) == 1
        assert result.accepted_log[0].evidence_index == 1
        # 3 attempts on ev0 + 1 on ev1 = 4 attempts logged
        assert len(result.attempts_log) == 4

    def test_critic_failure_logs_reason_for_next_attempt(self, fake_llm):
        """The retry prompt should include why the previous attempt failed.
        We can't observe the prompt directly, but we can verify the
        AtomicAttempt's failure_reason carries the critic's rejection text
        so the next call sees it."""
        target = _target(1)
        for _ in range(2):
            fake_llm.push(_atomic_response("edit", body="x " * 100))

        critic_calls = {"n": 0}
        def crit_per(cand, cur, ev):
            critic_calls["n"] += 1
            if critic_calls["n"] == 1:
                return (False, "new still ignores vegan")
            return (True, "")

        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(1),
            critic_per_attempt=crit_per,
            final_replay=lambda *a, **kw: (True, ""),
            llm=fake_llm,
        )
        assert result.action == "edit"
        assert "vegan" in result.attempts_log[0].failure_reason


# ── LLM signals ─────────────────────────────────────────────────────────────

class TestLLMSignals:
    def test_done_action_short_circuits_loop(self, fake_llm):
        """An LLM-issued <action>done</action> stops processing remaining
        evidence."""
        target = _target(5)
        # First evidence: edit accepted; second evidence: LLM says done
        fake_llm.push(_atomic_response("edit", body="x " * 100))
        fake_llm.push(_atomic_response("done"))

        crit_per, rep_final = _validators_always_pass()
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(5),
            critic_per_attempt=crit_per,
            final_replay=rep_final,
            llm=fake_llm,
        )
        # Only 2 LLM calls — first edit + done; remaining 3 evidence skipped
        assert len(fake_llm.calls) == 2
        assert len(result.accepted_log) == 1   # only the first edit was accepted
        # The 'done' attempt is in the attempts_log
        assert any(a.action == "done" for a in result.attempts_log)

    def test_skip_action_moves_to_next_evidence(self, fake_llm):
        """An LLM-issued <action>skip</action> for one evidence doesn't halt
        the whole loop — just moves on."""
        target = _target(2)
        fake_llm.push(_atomic_response("skip"))                    # ev0
        fake_llm.push(_atomic_response("edit", body="y " * 100))   # ev1

        crit_per, rep_final = _validators_always_pass()
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(2),
            critic_per_attempt=crit_per,
            final_replay=rep_final,
            llm=fake_llm,
        )
        assert len(result.accepted_log) == 1
        assert result.accepted_log[0].evidence_index == 1


# ── Final replay observed ───────────────────────────────────────────────────

class TestFinalReplay:
    def test_final_replay_runs_once_on_cumulative_state(self, fake_llm):
        """After all evidence is processed, final_replay fires once on the
        cumulative state. No rollback regardless of its outcome."""
        target = _target(3)
        for _ in range(3):
            fake_llm.push(_atomic_response("edit", body="step " * 100))

        observed: list[str] = []
        def final_replay(cand, target, convs):
            observed.append(cand)
            return (False, "regression on baseline")   # outcome doesn't roll back

        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(3),
            critic_per_attempt=lambda *a, **kw: (True, ""),
            final_replay=final_replay,
            llm=fake_llm,
        )
        # Exactly one final_replay call regardless of pass/fail
        assert len(observed) == 1
        # All accepted changes stay; rollback is gone
        assert result.action == "edit"
        assert len(result.accepted_log) == 3


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_evidence_returns_skip(self, fake_llm):
        """A Target with no evidence has nothing to address."""
        target = Target(skill_name="x", evidence=[])
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations={},
            critic_per_attempt=lambda *a, **kw: (True, ""),
            final_replay=lambda *a, **kw: (True, ""),
            llm=fake_llm,
        )
        assert result.action == "skip"
        assert len(fake_llm.calls) == 0

    def test_empty_new_skill_md_treated_as_failure(self, fake_llm):
        """Edit with empty body should fail validation, not silently pass."""
        target = _target(1)
        # First attempt: empty body. Second: normal.
        fake_llm.push("<action>edit</action><reasoning>r</reasoning><new_skill_md>   </new_skill_md>")
        fake_llm.push(_atomic_response("edit", body="real " * 100))

        crit_per, rep_final = _validators_always_pass()
        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(1),
            critic_per_attempt=crit_per,
            final_replay=rep_final,
            llm=fake_llm,
        )
        assert result.action == "edit"
        assert len(result.accepted_log) == 1
        assert result.attempts_log[0].failure_reason == "Empty new_skill_md"

    def test_max_attempts_respected(self, fake_llm):
        """Constant gives 3 attempts per evidence — verify by making them
        all fail and counting calls."""
        target = _target(1)
        for _ in range(MAX_ATTEMPTS_PER_EVIDENCE):
            fake_llm.push(_atomic_response("edit", body="x " * 100))

        result = propose(
            target,
            current_skill_md="# original",
            program_md="# strat",
            conversations=_conversations(1),
            critic_per_attempt=lambda *a, **kw: (False, "nope"),
            final_replay=lambda *a, **kw: (True, ""),
            llm=fake_llm,
        )
        assert result.action == "skip"
        assert len(fake_llm.calls) == MAX_ATTEMPTS_PER_EVIDENCE
        assert all(not a.accepted for a in result.attempts_log)
