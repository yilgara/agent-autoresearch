"""Tests for v3 `compute_verdict` — 4-gate logic with rubric + binary checks."""

from __future__ import annotations

import pytest

from agent_autoresearch.strategies.v3.critic import CriticResult
from agent_autoresearch.strategies.v3.judge import (
    CheckOutcome,
    RubricScore,
)
from agent_autoresearch.strategies.v3.propose import ProposeResult
from agent_autoresearch.strategies.v3.replay import ReplayResult, SessionReplay
from agent_autoresearch.strategies.v3.verdict import THRESHOLDS, compute_verdict


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


def _session(role, *, winner="new", rubric_new=3, rubric_old=2,
              checks_pass=True) -> SessionReplay:
    return SessionReplay(
        session_id=f"s_{role}_{winner}", role=role, focus_turn=1,
        user_message="", old_reply="", new_tool_plan="", new_reply="",
        winner=winner,
        rubric_scores=[
            RubricScore("a", rubric_new, rubric_old),
            RubricScore("b", rubric_new, rubric_old),
            RubricScore("c", rubric_new, rubric_old),
        ],
        check_results=[
            CheckOutcome(i, "pass" if checks_pass else "fail")
            for i in range(1, 6)
        ],
    )


def _replay(*, fix_sessions: list, baseline_sessions: list) -> ReplayResult:
    rr = ReplayResult(skill_name="x")
    rr.fix_target_replays.extend(fix_sessions)
    rr.regression_replays.extend(baseline_sessions)
    return rr


def _all_pass_replay(n_fix: int = 10, n_baseline: int = 10) -> ReplayResult:
    """A replay where every gate is at 100%."""
    return _replay(
        fix_sessions=[_session("fix_target") for _ in range(n_fix)],
        baseline_sessions=[_session("baseline", winner="tie") for _ in range(n_baseline)],
    )


# ── SKIP / contract violation ───────────────────────────────────────────────

def test_skip_action_returns_skip():
    v = compute_verdict(
        skill_name="x", propose_result=_propose("skip"),
        critic_result=None, replay_result=None,
    )
    assert v.label == "SKIP"


def test_replay_none_with_edit_action_raises():
    """Replay always runs for edits — passing None is a caller bug."""
    with pytest.raises(ValueError, match="replay_result is None"):
        compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(approves=True), replay_result=None,
        )


# ── Hard rejects ────────────────────────────────────────────────────────────

class TestHardRejects:
    def test_critic_rejects_with_replay_returns_reject(self):
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(approves=False),
            replay_result=_all_pass_replay(),
        )
        assert v.label == "REJECT"

    def test_low_fix_rate_triggers_floor(self):
        # 1/10 fix wins = 10%, well below 30% floor
        rr = _replay(
            fix_sessions=(
                [_session("fix_target", winner="new")]
                + [_session("fix_target", winner="old") for _ in range(9)]
            ),
            baseline_sessions=[_session("baseline", winner="tie") for _ in range(10)],
        )
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "REJECT"
        assert "fix_rate" in v.reason

    def test_low_binary_checks_triggers_floor(self):
        # All sessions fail every binary check
        rr = _replay(
            fix_sessions=[
                _session("fix_target", checks_pass=False) for _ in range(10)
            ],
            baseline_sessions=[
                _session("baseline", winner="tie", checks_pass=False)
                for _ in range(10)
            ],
        )
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "REJECT"
        assert "binary_checks_pass_rate" in v.reason


# ── ACCEPT ──────────────────────────────────────────────────────────────────

class TestAccept:
    def test_all_gates_pass(self):
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(),
            replay_result=_all_pass_replay(),
        )
        assert v.label == "ACCEPT"
        assert v.fix_rate == 1.0
        assert v.regression_rate == 1.0
        assert v.rubric_improvement_rate == 1.0
        assert v.binary_checks_pass_rate == 1.0

    def test_at_exact_thresholds(self):
        # Hit each threshold exactly: fix 50%, regr 90%, rubric 70%, checks 95%
        # 10+10 = 20 sessions, choose breakdown
        # Fix: 5/10 win = 50% ✓
        # Baselines: 9/10 not-loss → 9/10 = 90% ✓
        # Rubric: 14/20 sessions show improvement → 70% ✓
        # Checks: 19/20 all checks pass → 95% ✓

        # Build by hand
        fixes: list = []
        for i in range(10):
            fixes.append(_session(
                "fix_target",
                winner="new" if i < 5 else "old",
                rubric_new=3 if i < 7 else 1, rubric_old=2,
                checks_pass=(i < 9),
            ))
        baselines: list = []
        for i in range(10):
            baselines.append(_session(
                "baseline",
                winner="tie" if i < 9 else "old",
                rubric_new=2 if i < 7 else 1, rubric_old=2,
                checks_pass=(i < 10),
            ))
        rr = _replay(fix_sessions=fixes, baseline_sessions=baselines)
        # Sanity-check the rates
        assert rr.fix_rate == 0.50
        assert rr.regression_rate == 0.90
        # 7 fix improved + 7 base non-regressed = 14/20 = 70%
        assert rr.rubric_improvement_rate == 0.70
        # 9 fix + 10 base = 19/20 = 95%
        assert rr.binary_checks_pass_rate == 0.95

        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "ACCEPT"


# ── HUMAN_REVIEW ────────────────────────────────────────────────────────────

class TestHumanReview:
    def test_rubric_below_acceptance_above_floor(self):
        # All other gates pass; rubric at 50% (below 70% but no rubric floor exists)
        fixes = [
            _session("fix_target",
                      rubric_new=3 if i < 5 else 1,   # 5 improved, 5 regressed
                      rubric_old=2,
                      checks_pass=True)
            for i in range(10)
        ]
        baselines = [
            _session("baseline", winner="tie",
                      rubric_new=2, rubric_old=2,
                      checks_pass=True)
            for _ in range(10)
        ]
        rr = _replay(fix_sessions=fixes, baseline_sessions=baselines)
        # rubric_ok = 5 (fix improved) + 10 (base non-regressed) = 15/20 = 75% ?
        # Let's recompute: 5 fix improved (rubric_session_ok=True),
        # 5 fix regressed (rubric_session_ok=False),
        # 10 base non-regressed (True) → 15/20 = 75% which IS above 70%.
        # We need to make rubric_improvement < 70%.
        # Use 3 improved fixes + 7 regressed fixes + 10 baselines OK = 13/20 = 65%
        fixes = [
            _session("fix_target", winner="new",
                      rubric_new=3 if i < 3 else 1,
                      rubric_old=2,
                      checks_pass=True)
            for i in range(10)
        ]
        rr = _replay(fix_sessions=fixes, baseline_sessions=baselines)
        # All winners 'new' → fix_rate = 100% (passes)
        # rubric: 3 fix improved + 10 base non-regressed = 13/20 = 65% (below 70%)
        # checks: 100% (passes)
        # → HUMAN_REVIEW
        assert rr.fix_rate == 1.0
        assert rr.regression_rate == 1.0
        assert rr.binary_checks_pass_rate == 1.0
        assert 0.6 <= rr.rubric_improvement_rate < 0.7

        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "HUMAN_REVIEW"


# ── Verdict carries the right signals ──────────────────────────────────────

def test_verdict_preserves_all_4_rates():
    rr = _all_pass_replay(n_fix=5, n_baseline=5)
    v = compute_verdict(
        skill_name="my-skill", propose_result=_propose(),
        critic_result=_critic(), replay_result=rr,
    )
    assert v.skill_name == "my-skill"
    assert v.fix_rate == 1.0
    assert v.regression_rate == 1.0
    assert v.rubric_improvement_rate == 1.0
    assert v.binary_checks_pass_rate == 1.0
    assert v.n_fix_replays == 5
    assert v.n_baseline_replays == 5
    md = v.to_markdown()
    assert "ACCEPT" in md
    assert "fix_rate" in md
    assert "binary_checks_pass_rate" in md
