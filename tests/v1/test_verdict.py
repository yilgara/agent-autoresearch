"""Tests for `compute_verdict` — pure logic, easy to enumerate.

Covers all 4 labels (ACCEPT / HUMAN_REVIEW / REJECT / SKIP) plus
threshold edges. New semantics:

  - fix_target_score : strict `>` 0 (any improvement counts)
  - regression_score : >= 0.90 hard floor
"""

from __future__ import annotations

from agent_autoresearch.strategies.v1.critic import CriticResult
from agent_autoresearch.strategies.v1.propose import ProposeResult
from agent_autoresearch.strategies.v1.replay import ReplayResult, SessionReplay
from agent_autoresearch.strategies.v1.verdict import THRESHOLDS, compute_verdict


# ── Builders ────────────────────────────────────────────────────────────────

def _propose(action: str = "edit") -> ProposeResult:
    return ProposeResult(
        skill_name="x", action=action,
        new_skill_md="# new" if action == "edit" else None,
        reasoning="r", raw_response="",
        input_tokens=0, output_tokens=0,
    )


def _critic(approves: bool = True) -> CriticResult:
    return CriticResult(
        skill_name="x",
        verdict="APPROVE" if approves else "REQUEST_CHANGES",
        reasoning="r", concerns=[] if approves else ["c"],
        raw_response="", input_tokens=0, output_tokens=0,
    )


def _replay(*, fix_score: float, regr_score: float,
             n_fix: int = 3, n_baseline: int = 3) -> ReplayResult:
    """Build a ReplayResult whose pass-rate properties match the
    requested scores."""
    rr = ReplayResult(skill_name="x")
    n_fix_passes = round(fix_score * n_fix)
    for i in range(n_fix):
        rr.fix_target_replays.append(SessionReplay(
            session_id=f"s{i}", role="fix_target", focus_turn=1,
            user_message="", old_reply="", new_tool_plan="", new_reply="",
            new_passes=(i < n_fix_passes), judge_reasoning="",
            responder_tokens=(None, None), judge_tokens=(None, None),
        ))
    n_baseline_passes = round(regr_score * n_baseline)
    for i in range(n_baseline):
        rr.regression_replays.append(SessionReplay(
            session_id=f"b{i}", role="baseline", focus_turn=1,
            user_message="", old_reply="", new_tool_plan="", new_reply="",
            new_passes=(i < n_baseline_passes), judge_reasoning="",
            responder_tokens=(None, None), judge_tokens=(None, None),
        ))
    return rr


# ── SKIP ────────────────────────────────────────────────────────────────────

def test_skip_action_returns_skip():
    v = compute_verdict(
        skill_name="x", propose_result=_propose("skip"),
        critic_result=None, replay_result=None,
    )
    assert v.label == "SKIP"
    assert v.propose_action == "skip"


# ── Contract violation ──────────────────────────────────────────────────────

def test_replay_none_with_edit_action_raises():
    """Replay always runs for edits — passing replay_result=None for an
    edit action is a caller bug, not a verdict variant."""
    import pytest
    with pytest.raises(ValueError, match="replay_result is None"):
        compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(approves=True), replay_result=None,
        )


# ── REJECT ──────────────────────────────────────────────────────────────────

def test_critic_rejects_with_replay_returns_reject():
    v = compute_verdict(
        skill_name="x", propose_result=_propose(),
        critic_result=_critic(approves=False),
        replay_result=_replay(fix_score=1.0, regr_score=1.0),
    )
    assert v.label == "REJECT"
    assert "critic REQUEST_CHANGES" in v.reason


def test_low_regression_score_returns_reject():
    """regression_score below `regression_min` is the only hard floor."""
    threshold = THRESHOLDS["regression_min"]
    v = compute_verdict(
        skill_name="x", propose_result=_propose(),
        critic_result=_critic(approves=True),
        replay_result=_replay(fix_score=1.0, regr_score=threshold - 0.1),
    )
    assert v.label == "REJECT"


# ── ACCEPT ──────────────────────────────────────────────────────────────────

def test_both_gates_clearly_pass_returns_accept():
    v = compute_verdict(
        skill_name="x", propose_result=_propose(),
        critic_result=_critic(approves=True),
        replay_result=_replay(fix_score=1.0, regr_score=1.0),
    )
    assert v.label == "ACCEPT"
    assert v.fix_target_score == 1.0
    assert v.regression_score == 1.0


def test_single_fix_pass_is_enough_for_accept():
    """One fix session passing is enough — `> 0` gate, not `>= 70%`."""
    v = compute_verdict(
        skill_name="x", propose_result=_propose(),
        critic_result=_critic(approves=True),
        replay_result=_replay(fix_score=1/10, regr_score=1.0, n_fix=10),
    )
    assert v.label == "ACCEPT"
    assert v.fix_target_score == 0.1


# ── HUMAN_REVIEW ────────────────────────────────────────────────────────────

def test_zero_fix_passes_returns_human_review():
    """No improvement at all on fix sessions → HUMAN_REVIEW (not REJECT,
    since regression and critic are fine)."""
    v = compute_verdict(
        skill_name="x", propose_result=_propose(),
        critic_result=_critic(approves=True),
        replay_result=_replay(fix_score=0.0, regr_score=1.0),
    )
    assert v.label == "HUMAN_REVIEW"
    assert v.fix_target_score == 0.0


# ── Output detail ───────────────────────────────────────────────────────────

def test_verdict_carries_signals_for_markdown():
    v = compute_verdict(
        skill_name="my-skill", propose_result=_propose(),
        critic_result=_critic(approves=True),
        replay_result=_replay(fix_score=1.0, regr_score=1.0,
                                n_fix=3, n_baseline=3),
    )
    assert v.skill_name == "my-skill"
    assert v.propose_action == "edit"
    assert v.critic_verdict == "APPROVE"
    assert v.n_fix_replays == 3
    assert v.n_baseline_replays == 3
    md = v.to_markdown()
    assert "ACCEPT" in md
    assert "my-skill" in md
