"""Step 8 — deterministic verdict combining critic + replay.

No LLM here. Pure logic that takes the outputs of step 6 (critic)
and step 7 (replay) and emits one of:

  - **ACCEPT**       — both gates pass; safe to apply after human review
  - **HUMAN_REVIEW** — gates ambiguous; needs a human eye
  - **REJECT**       — at least one gate failed; don't ship
  - **SKIP**         — propose returned skip; nothing to verdict on
  - **NO_VALIDATION** — replay was disabled (`--no-validate`)

Thresholds in `THRESHOLDS` are conservative on purpose: burden of
proof is on the new skill. Tune after observing 5-10 real runs.

This is strategy v1's verdict logic. v2 plans to make the two-axis
structure (tests-must-pass + rubric-must-improve) explicit with
finer-grained scoring, which is why verdict lives inside the strategy
folder rather than at package level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_autoresearch.strategies.v2.critic import CriticResult
from agent_autoresearch.strategies.v2.propose import ProposeResult
from agent_autoresearch.strategies.v2.replay import ReplayResult


# ── Thresholds — tunable per deployment ─────────────────────────────────────

THRESHOLDS = {
    "fix_target_min":   0.7,    # at least 70% of fix-targets must improve
    "regression_min":   0.9,    # at least 90% of baselines must hold
    "fix_reject_below": 0.5,    # below this, it's a hard reject
}


# ── Result type ─────────────────────────────────────────────────────────────

VerdictLabel = Literal[
    "ACCEPT",
    "HUMAN_REVIEW",
    "REJECT",
    "SKIP",
    "NO_VALIDATION",
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
    fix_target_score: float | None = None
    regression_score: float | None = None
    n_fix_replays: int = 0
    n_baseline_replays: int = 0

    def to_markdown(self) -> str:
        """Render `verdict.md` — what gets written to the per-target output folder."""
        return _render_verdict_md(self)


# ── Compute logic ───────────────────────────────────────────────────────────

def compute_verdict(
    *,
    skill_name: str,
    propose_result: ProposeResult,
    critic_result: CriticResult | None,
    replay_result: ReplayResult | None,
) -> Verdict:
    """Combine the upstream signals into one of five labels."""

    # 1. Propose said skip → nothing was proposed
    if propose_result.action == "skip":
        return Verdict(
            skill_name=skill_name,
            label="SKIP",
            reason=propose_result.reasoning or "Proposer chose to skip.",
            propose_action="skip",
        )

    critic_label = critic_result.verdict if critic_result else None

    # 2. Replay disabled → human must judge alone (unless critic clearly rejects)
    if replay_result is None:
        if critic_result and not critic_result.approves:
            return Verdict(
                skill_name=skill_name, label="REJECT",
                reason=(
                    "Critic returned REQUEST_CHANGES: "
                    + (", ".join(critic_result.concerns) or critic_result.reasoning)
                ),
                propose_action=propose_result.action,
                critic_verdict=critic_label,
            )
        return Verdict(
            skill_name=skill_name, label="NO_VALIDATION",
            reason="Replay was disabled. Human must review without replay scores.",
            propose_action=propose_result.action,
            critic_verdict=critic_label,
        )

    # 3. Hard rejects
    rejects: list[str] = []
    if critic_result and not critic_result.approves:
        rejects.append(
            "critic REQUEST_CHANGES: "
            + (", ".join(critic_result.concerns) or critic_result.reasoning)
        )
    if replay_result.fix_target_score < THRESHOLDS["fix_reject_below"]:
        rejects.append(
            f"fix_target_score {replay_result.fix_target_score:.0%} < "
            f"{THRESHOLDS['fix_reject_below']:.0%} "
            f"(new skill failed to improve majority of fix-targets)"
        )
    if replay_result.regression_score < THRESHOLDS["regression_min"]:
        rejects.append(
            f"regression_score {replay_result.regression_score:.0%} < "
            f"{THRESHOLDS['regression_min']:.0%} "
            f"(new skill regressed on baseline sessions)"
        )
    if rejects:
        return Verdict(
            skill_name=skill_name, label="REJECT",
            reason="; ".join(rejects),
            propose_action=propose_result.action,
            critic_verdict=critic_label,
            fix_target_score=replay_result.fix_target_score,
            regression_score=replay_result.regression_score,
            n_fix_replays=len(replay_result.fix_target_replays),
            n_baseline_replays=len(replay_result.regression_replays),
        )

    # 4. ACCEPT — both gates clearly pass
    critic_ok = critic_result is None or critic_result.approves
    fix_ok = replay_result.fix_target_score >= THRESHOLDS["fix_target_min"]
    regr_ok = replay_result.regression_score >= THRESHOLDS["regression_min"]
    if critic_ok and fix_ok and regr_ok:
        return Verdict(
            skill_name=skill_name, label="ACCEPT",
            reason=(
                f"Critic APPROVE · "
                f"fix_target_score={replay_result.fix_target_score:.0%} · "
                f"regression_score={replay_result.regression_score:.0%}"
            ),
            propose_action=propose_result.action,
            critic_verdict=critic_label,
            fix_target_score=replay_result.fix_target_score,
            regression_score=replay_result.regression_score,
            n_fix_replays=len(replay_result.fix_target_replays),
            n_baseline_replays=len(replay_result.regression_replays),
        )

    # 5. HUMAN_REVIEW — between thresholds
    return Verdict(
        skill_name=skill_name, label="HUMAN_REVIEW",
        reason=(
            f"Critic APPROVE · "
            f"fix_target_score={replay_result.fix_target_score:.0%} "
            f"(threshold {THRESHOLDS['fix_target_min']:.0%}) · "
            f"regression_score={replay_result.regression_score:.0%} "
            "— some signal but below auto-ACCEPT threshold."
        ),
        propose_action=propose_result.action,
        critic_verdict=critic_label,
        fix_target_score=replay_result.fix_target_score,
        regression_score=replay_result.regression_score,
        n_fix_replays=len(replay_result.fix_target_replays),
        n_baseline_replays=len(replay_result.regression_replays),
    )


# ── Markdown rendering for verdict.md ───────────────────────────────────────

_LABEL_BADGE = {
    "ACCEPT":         "🟢 ACCEPT",
    "HUMAN_REVIEW":   "🟡 HUMAN_REVIEW",
    "REJECT":         "🔴 REJECT",
    "SKIP":           "⚪ SKIP",
    "NO_VALIDATION":  "⚪ NO_VALIDATION",
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
    rows = [
        ("Propose action",     v.propose_action or "—"),
        ("Critic verdict",     v.critic_verdict or "—"),
        ("Fix target score",
         f"{v.fix_target_score:.0%} ({v.n_fix_replays} replays)"
         if v.fix_target_score is not None else "—"),
        ("Regression score",
         f"{v.regression_score:.0%} ({v.n_baseline_replays} replays)"
         if v.regression_score is not None else "—"),
    ]
    for label, value in rows:
        lines.append(f"- **{label}:** {value}")
    lines.append("")

    lines += ["## Recommendation", ""]
    if v.label == "ACCEPT":
        lines.append(
            "Both validation gates passed. Safe to apply to the source "
            "repo's `skills/` directory after a human eye on `diff.txt`. "
            "Do NOT auto-merge — confirm manually first."
        )
    elif v.label == "HUMAN_REVIEW":
        lines.append(
            "Gates are mixed. Read `replay.md` per-session results and "
            "decide based on whether the new skill clearly addresses "
            "the failure pattern in `program.md`. The replay LLM is one "
            "imagining what another LLM would do — not ground truth."
        )
    elif v.label == "REJECT":
        lines.append(
            "At least one gate failed. Do not apply this edit. Likely "
            "next steps: (1) regenerate `program.md` with different "
            "framing, or (2) defer this skill to a future round when "
            "evidence is stronger."
        )
    elif v.label == "SKIP":
        lines.append(
            "The proposer chose not to attempt an edit. This is correct "
            "behavior when evidence is too weak or contradictory. "
            "Re-evaluate after the next eval run."
        )
    elif v.label == "NO_VALIDATION":
        lines.append(
            "Replay was disabled. A human reviewer must read "
            "`diff.txt`, `program.md`, and `critic.md` and decide. "
            "For higher-confidence verdicts, re-run with replay enabled."
        )
    return "\n".join(lines)
