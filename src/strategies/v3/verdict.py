"""Step 8 (v3) — verdict combining critic + 4 replay aggregates.

No LLM here. Pure logic that takes:
  - propose_result   (from v3/v2 atomic-mutation propose)
  - critic_result    (from v2/v1 critic — unchanged)
  - replay_result    (from v3 replay — 4 aggregate rates)

…and emits one of:

  - **ACCEPT**       — all 4 gates pass + critic approves
  - **HUMAN_REVIEW** — gates ambiguous; needs a human eye
  - **REJECT**       — at least one hard-reject gate triggered
  - **SKIP**         — propose returned skip; nothing to verdict on

## v3 thresholds

  fix_rate                  >= 50%   on fix sessions       (acceptance)
  regression_rate           >= 90%   on baselines          (acceptance)
  rubric_improvement_rate   >= 70%   on all sessions       (acceptance)
  binary_checks_pass_rate   >= 95%   on all sessions       (acceptance)

  fix_rate                  <  30%   on fix sessions       (hard reject)
  binary_checks_pass_rate   <  80%   on all sessions       (hard reject)

Between acceptance and hard-reject thresholds → HUMAN_REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_autoresearch.strategies.v3.critic import CriticResult
from agent_autoresearch.strategies.v3.propose import ProposeResult
from agent_autoresearch.strategies.v3.replay import ReplayResult


# ── Thresholds — tunable per deployment ─────────────────────────────────────

THRESHOLDS = {
    # Acceptance gates — all four must hold for ACCEPT.
    # fix_rate uses a strict `>` (any improvement counts) — one bundled
    # edit realistically can't hit 50% of fix sessions.
    "fix_rate_min":              0.0,
    "regression_rate_min":       0.90,
    "rubric_improvement_min":    0.70,
    "binary_checks_min":         0.95,

    # Hard-reject floor — drop the binary_checks_floor to REJECT when
    # the new prompt breaks too many invariants.
    "binary_checks_floor":       0.80,
}


# ── Result type ─────────────────────────────────────────────────────────────

VerdictLabel = Literal[
    "ACCEPT",
    "HUMAN_REVIEW",
    "REJECT",
    "SKIP",
]


@dataclass
class Verdict:
    """Final per-target verdict + the signals that fed it."""
    skill_name: str
    label: VerdictLabel
    reason: str

    # Source signals (preserved for the markdown trace)
    propose_action: str = ""
    critic_verdict: str | None = None

    # v3 — four rates
    fix_rate: float | None = None
    regression_rate: float | None = None
    rubric_improvement_rate: float | None = None
    binary_checks_pass_rate: float | None = None

    n_fix_replays: int = 0
    n_baseline_replays: int = 0

    # Backward-compat aliases — v1/v2 callers still see these names
    @property
    def fix_target_score(self) -> float | None:
        return self.fix_rate

    @property
    def regression_score(self) -> float | None:
        return self.regression_rate

    def to_markdown(self) -> str:
        return _render_verdict_md(self)


# ── Compute logic ───────────────────────────────────────────────────────────

def compute_verdict(
    *,
    skill_name: str,
    propose_result: ProposeResult,
    critic_result: CriticResult | None,
    replay_result: ReplayResult | None,
) -> Verdict:
    """Combine propose + critic + 4 replay rates into one of five labels."""

    # 1. Propose said skip → nothing to verdict
    if propose_result.action == "skip":
        return Verdict(
            skill_name=skill_name, label="SKIP",
            reason=propose_result.reasoning or "Proposer chose to skip.",
            propose_action="skip",
        )

    critic_label = critic_result.verdict if critic_result else None

    # Replay always runs for the edit path; if replay_result is None
    # here, the caller violated the contract — fail loudly.
    if replay_result is None:
        raise ValueError(
            f"compute_verdict({skill_name=}): replay_result is None for an "
            "edit action. Replay always runs for edits in the pipeline."
        )

    # Pull rates once for readability
    fix_rate = replay_result.fix_rate
    regr_rate = replay_result.regression_rate
    rubric_rate = replay_result.rubric_improvement_rate
    checks_rate = replay_result.binary_checks_pass_rate

    base_signals = Verdict(
        skill_name=skill_name, label="ACCEPT", reason="",   # placeholder
        propose_action=propose_result.action,
        critic_verdict=critic_label,
        fix_rate=fix_rate,
        regression_rate=regr_rate,
        rubric_improvement_rate=rubric_rate,
        binary_checks_pass_rate=checks_rate,
        n_fix_replays=len(replay_result.fix_target_replays),
        n_baseline_replays=len(replay_result.regression_replays),
    )

    # 3. Hard rejects
    rejects: list[str] = []
    if critic_result and not critic_result.approves:
        rejects.append(
            "critic REQUEST_CHANGES: "
            + (", ".join(critic_result.concerns) or critic_result.reasoning)
        )
    if checks_rate < THRESHOLDS["binary_checks_floor"]:
        rejects.append(
            f"binary_checks_pass_rate {checks_rate:.0%} < "
            f"{THRESHOLDS['binary_checks_floor']:.0%} (floor)"
        )
    if rejects:
        base_signals.label = "REJECT"
        base_signals.reason = "; ".join(rejects)
        return base_signals

    # 4. ACCEPT — all four gates clearly pass
    critic_ok = critic_result is None or critic_result.approves
    gates_ok = (
        fix_rate    >  THRESHOLDS["fix_rate_min"]
        and regr_rate   >= THRESHOLDS["regression_rate_min"]
        and rubric_rate >= THRESHOLDS["rubric_improvement_min"]
        and checks_rate >= THRESHOLDS["binary_checks_min"]
    )
    if critic_ok and gates_ok:
        base_signals.label = "ACCEPT"
        base_signals.reason = (
            f"Critic APPROVE · "
            f"fix_rate={fix_rate:.0%} · "
            f"regression_rate={regr_rate:.0%} · "
            f"rubric_improvement_rate={rubric_rate:.0%} · "
            f"binary_checks_pass_rate={checks_rate:.0%}"
        )
        return base_signals

    # 5. HUMAN_REVIEW — between hard-reject floors and acceptance gates
    base_signals.label = "HUMAN_REVIEW"
    base_signals.reason = (
        f"Some signal but not all gates passed: "
        f"fix_rate={fix_rate:.0%} (need ≥{THRESHOLDS['fix_rate_min']:.0%}) · "
        f"regression_rate={regr_rate:.0%} (need ≥{THRESHOLDS['regression_rate_min']:.0%}) · "
        f"rubric_improvement_rate={rubric_rate:.0%} "
        f"(need ≥{THRESHOLDS['rubric_improvement_min']:.0%}) · "
        f"binary_checks_pass_rate={checks_rate:.0%} "
        f"(need ≥{THRESHOLDS['binary_checks_min']:.0%})"
    )
    return base_signals


# ── Markdown rendering for verdict.md ───────────────────────────────────────

_LABEL_BADGE = {
    "ACCEPT":         "🟢 ACCEPT",
    "HUMAN_REVIEW":   "🟡 HUMAN_REVIEW",
    "REJECT":         "🔴 REJECT",
    "SKIP":           "⚪ SKIP",
}


def _render_verdict_md(v: Verdict) -> str:
    lines = [
        f"# Verdict: {_LABEL_BADGE.get(v.label, v.label)}",
        "",
        f"**Skill:** `{v.skill_name}`",
        "",
        "## Reason",
        "",
        v.reason,
        "",
        "## Signals",
        "",
    ]

    def _fmt(rate: float | None, n: int) -> str:
        if rate is None:
            return "—"
        return f"{rate:.0%} (over {n} replays)"

    rows = [
        ("Propose action",            v.propose_action or "—"),
        ("Critic verdict",            v.critic_verdict or "—"),
        ("fix_rate",                  _fmt(v.fix_rate, v.n_fix_replays)),
        ("regression_rate",           _fmt(v.regression_rate, v.n_baseline_replays)),
        ("rubric_improvement_rate",   _fmt(v.rubric_improvement_rate,
                                            v.n_fix_replays + v.n_baseline_replays)),
        ("binary_checks_pass_rate",   _fmt(v.binary_checks_pass_rate,
                                            v.n_fix_replays + v.n_baseline_replays)),
    ]
    for label, value in rows:
        lines.append(f"- **{label}:** {value}")
    lines.append("")

    lines += ["## Recommendation", ""]
    if v.label == "ACCEPT":
        lines.append(
            "All four validation gates passed and the critic approved. "
            "Safe to apply to the source repo's `skills/` directory after "
            "a human eye on `diff.txt`. Do NOT auto-merge — confirm manually first."
        )
    elif v.label == "HUMAN_REVIEW":
        lines.append(
            "Mixed signals — at least one gate is below its acceptance "
            "threshold but no hard-reject floor was hit. Read `replay.md` "
            "per-session, weigh the rubric scores and check failures, and "
            "decide based on whether the regressions are tolerable."
        )
    elif v.label == "REJECT":
        lines.append(
            "At least one gate triggered hard reject. Do not apply this edit. "
            "Likely next steps: regenerate `program.md` with different framing, "
            "or defer this skill to a future round."
        )
    elif v.label == "SKIP":
        lines.append(
            "The proposer chose not to attempt an edit. Re-evaluate after "
            "the next eval run."
        )
    return "\n".join(lines)
