"""Tests for v3 `compute_verdict` — new metric semantics.

  - fix_rate                  : informational (no gate)
  - regression_rate           : pass rate over baselines ≥ 0.90
  - binary_checks_pass_rate   : over (baseline × check) pairs ≥ 0.90
  - rubric_score              : mean +1/0/-1 over (fix × axis) pairs ≥ 0

REJECT is reserved for the critic. Anything that fails a rate gate but
the critic approves → HUMAN_REVIEW.
"""

from __future__ import annotations

import pytest

from agent_autoresearch.strategies.v3.critic import CriticResult
from agent_autoresearch.strategies.v3.judge import CheckOutcome, RubricVote
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


def _fix_session(*, new_passes: bool = True,
                  rubric_winners: tuple[str, str, str] = ("new", "new", "new"),
                  checks_pass: bool = True) -> SessionReplay:
    """Fix session: rubric votes drive `rubric_score`; checks are
    aggregated only over baselines but we set them anyway to mirror
    real replays (and to verify they don't accidentally feed the
    aggregate)."""
    return SessionReplay(
        session_id="fix", role="fix_target", focus_turn=1,
        user_message="", old_reply="", new_tool_plan="", new_reply="",
        new_passes=new_passes,
        rubric_votes=[RubricVote(name, w) for name, w in
                      zip(("a", "b", "c"), rubric_winners)],
        check_results=[
            CheckOutcome(i, "pass" if checks_pass else "fail")
            for i in range(1, 6)
        ],
    )


def _baseline_session(*, new_passes: bool = True,
                      checks_pass: bool = True) -> SessionReplay:
    """Baseline session: new_passes drives `regression_rate`; check
    results drive `binary_checks_pass_rate`. Rubric votes here are
    ignored by the aggregate."""
    return SessionReplay(
        session_id="base", role="baseline", focus_turn=1,
        user_message="", old_reply="", new_tool_plan="", new_reply="",
        new_passes=new_passes,
        rubric_votes=[RubricVote("a", "tie"), RubricVote("b", "tie"),
                      RubricVote("c", "tie")],
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
    """Every gate at 100% / +1.00."""
    return _replay(
        fix_sessions=[_fix_session() for _ in range(n_fix)],
        baseline_sessions=[_baseline_session() for _ in range(n_baseline)],
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


# ── REJECT — critic only ────────────────────────────────────────────────────

class TestReject:
    def test_critic_rejects(self):
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(approves=False),
            replay_result=_all_pass_replay(),
        )
        assert v.label == "REJECT"

    def test_low_binary_checks_does_not_reject(self):
        """No hard-reject floors on rate metrics — only critic rejects.
        Low binary_checks_pass_rate lands in HUMAN_REVIEW, not REJECT."""
        rr = _replay(
            fix_sessions=[_fix_session() for _ in range(10)],
            baseline_sessions=[_baseline_session(checks_pass=False)
                                for _ in range(10)],
        )
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "HUMAN_REVIEW"
        assert v.binary_checks_pass_rate == 0.0


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
        assert v.rubric_score == 1.0
        assert v.binary_checks_pass_rate == 1.0

    def test_zero_fix_rate_can_still_accept(self):
        """fix_rate is informational — even 0% on fixes is fine if the
        other gates pass."""
        rr = _replay(
            fix_sessions=[_fix_session(new_passes=False) for _ in range(10)],
            baseline_sessions=[_baseline_session() for _ in range(10)],
        )
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "ACCEPT"
        assert v.fix_rate == 0.0

    def test_rubric_at_exact_threshold(self):
        """rubric_score = 0 (exactly the floor) → ACCEPT (>= 0)."""
        # 5 new, 5 old votes per session × 1 session → mean 0
        rr = _replay(
            fix_sessions=[
                _fix_session(rubric_winners=("new", "old", "tie")),
                _fix_session(rubric_winners=("new", "old", "tie")),
            ],
            baseline_sessions=[_baseline_session() for _ in range(10)],
        )
        # mean = (1 + -1 + 0 + 1 + -1 + 0) / 6 = 0.0
        assert rr.rubric_score == 0.0
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "ACCEPT"


# ── HUMAN_REVIEW ────────────────────────────────────────────────────────────

class TestHumanReview:
    def test_negative_rubric_score(self):
        """Rubric net negative → HUMAN_REVIEW (critic still approved)."""
        rr = _replay(
            fix_sessions=[
                _fix_session(rubric_winners=("old", "old", "tie"))
                for _ in range(5)
            ],
            baseline_sessions=[_baseline_session() for _ in range(10)],
        )
        assert rr.rubric_score < 0
        v = compute_verdict(
            skill_name="x", propose_result=_propose(),
            critic_result=_critic(), replay_result=rr,
        )
        assert v.label == "HUMAN_REVIEW"

    def test_regression_below_acceptance(self):
        """regression_rate at 80% (< 90%) → HUMAN_REVIEW."""
        bases = [_baseline_session(new_passes=i < 8) for i in range(10)]
        rr = _replay(
            fix_sessions=[_fix_session() for _ in range(5)],
            baseline_sessions=bases,
        )
        assert rr.regression_rate == 0.8
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
    assert v.rubric_score == 1.0
    assert v.binary_checks_pass_rate == 1.0
    assert v.n_fix_replays == 5
    assert v.n_baseline_replays == 5
    md = v.to_markdown()
    assert "ACCEPT" in md
    assert "fix_rate" in md
    assert "rubric_score" in md
    assert "binary_checks_pass_rate" in md
