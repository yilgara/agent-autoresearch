"""Step 8 (v3) — verdict combining critic + replay aggregates.

No LLM here. Pure logic that takes:
  - propose_result   (from v3 atomic-mutation propose)
  - critic_result    (from v3 critic — unchanged)
  - replay_result    (from v3 replay — 4 aggregates with new semantics)

…and emits one of:

  - **ACCEPT**       — all three gates pass + critic approves
  - **HUMAN_REVIEW** — gates ambiguous; needs a human eye
  - **REJECT**       — critic rejected the diff
  - **SKIP**         — propose returned skip; nothing to verdict on

## v3 thresholds (new semantics)

  fix_rate                  — informational only (gate `>= 0` is trivial).
                              One bundled edit realistically can't pass
                              a large fraction of fix sessions.
  regression_rate           >= 90%   on baseline sessions       (acceptance)
  binary_checks_pass_rate   >= 90%   over baseline session×check pairs
  rubric_score              >=  0    over fix session×axis pairs
                              (mean of +1/0/-1 votes — must be net positive)

There are no hard-reject floors on the rate metrics — REJECT is
reserved for the critic. Anything that fails an acceptance gate but
the critic approves goes to HUMAN_REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_autoresearch.strategies.v3.critic import CriticResult
from agent_autoresearch.strategies.v3.propose import ProposeResult
from agent_autoresearch.strategies.v3.replay import ReplayResult


# ── Thresholds — tunable per deployment ─────────────────────────────────────

THRESHOLDS = {
    # fix_rate has no gate; kept here as a documented constant so the
    # markdown render can show what the floor is (none).
    "fix_rate_min":           0.0,    # informational only — always passes
    "regression_rate_min":    0.90,   # over baseline sessions
    "binary_checks_min":      0.90,   # over baseline × check pairs
    "rubric_score_min":       0.0,    # mean of +1/0/-1 over fix × axis pairs
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

    # v3 rates (new semantics)
    fix_rate: float | None = None
    regression_rate: float | None = None
    binary_checks_pass_rate: float | None = None
    rubric_score: float | None = None

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
    """Combine propose + critic + replay rates into one of four labels."""

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

    fix_rate = replay_result.fix_rate
    regr_rate = replay_result.regression_rate
    checks_rate = replay_result.binary_checks_pass_rate
    rubric_score = replay_result.rubric_score

    base_signals = Verdict(
        skill_name=skill_name, label="ACCEPT", reason="",   # placeholder
        propose_action=propose_result.action,
        critic_verdict=critic_label,
        fix_rate=fix_rate,
        regression_rate=regr_rate,
        binary_checks_pass_rate=checks_rate,
        rubric_score=rubric_score,
        n_fix_replays=len(replay_result.fix_target_replays),
        n_baseline_replays=len(replay_result.regression_replays),
    )

    # 2. Critic rejection is the only hard reject
    if critic_result and not critic_result.approves:
        base_signals.label = "REJECT"
        base_signals.reason = (
            "critic REQUEST_CHANGES: "
            + (", ".join(critic_result.concerns) or critic_result.reasoning)
        )
        return base_signals

    # 3. ACCEPT — all rate gates clearly pass
    critic_ok = critic_result is None or critic_result.approves
    gates_ok = (
        regr_rate    >= THRESHOLDS["regression_rate_min"]
        and checks_rate  >= THRESHOLDS["binary_checks_min"]
        and rubric_score >= THRESHOLDS["rubric_score_min"]
    )
    if critic_ok and gates_ok:
        base_signals.label = "ACCEPT"
        base_signals.reason = (
            f"Critic APPROVE · "
            f"regression_rate={regr_rate:.0%} · "
            f"binary_checks_pass_rate={checks_rate:.0%} · "
            f"rubric_score={rubric_score:+.2f} · "
            f"fix_rate={fix_rate:.0%} (informational)"
        )
        return base_signals

    # 4. HUMAN_REVIEW — at least one acceptance gate failed
    base_signals.label = "HUMAN_REVIEW"
    base_signals.reason = (
        f"Some signal but not all gates passed: "
        f"regression_rate={regr_rate:.0%} (need ≥{THRESHOLDS['regression_rate_min']:.0%}) · "
        f"binary_checks_pass_rate={checks_rate:.0%} "
        f"(need ≥{THRESHOLDS['binary_checks_min']:.0%}) · "
        f"rubric_score={rubric_score:+.2f} (need ≥{THRESHOLDS['rubric_score_min']:.2f})"
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

    def _fmt_rate(rate: float | None, n: int, suffix: str = "") -> str:
        if rate is None:
            return "—"
        return f"{rate:.0%} ({n} sessions{suffix})"

    rows = [
        ("Propose action",          v.propose_action or "—"),
        ("Critic verdict",          v.critic_verdict or "—"),
        ("fix_rate (info)",         _fmt_rate(v.fix_rate, v.n_fix_replays)),
        ("regression_rate",         _fmt_rate(v.regression_rate, v.n_baseline_replays)),
        ("binary_checks_pass_rate",
            "—" if v.binary_checks_pass_rate is None
            else f"{v.binary_checks_pass_rate:.0%} (over baseline × check pairs)"),
        ("rubric_score",
            "—" if v.rubric_score is None
            else f"{v.rubric_score:+.2f} (over fix × axis pairs)"),
    ]
    for label, value in rows:
        lines.append(f"- **{label}:** {value}")
    lines.append("")

    lines += ["## Recommendation", ""]
    if v.label == "ACCEPT":
        lines.append(
            "All acceptance gates passed and the critic approved. "
            "Safe to apply to the source repo's `skills/` directory after "
            "a human eye on `diff.txt`. Do NOT auto-merge — confirm manually first."
        )
    elif v.label == "HUMAN_REVIEW":
        lines.append(
            "Mixed signals — at least one rate gate is below its acceptance "
            "threshold but the critic approved. Read `replay.md` per-session, "
            "weigh the rubric votes and check failures, and decide based on "
            "whether the regressions are tolerable."
        )
    elif v.label == "REJECT":
        lines.append(
            "Critic rejected the diff. Do not apply this edit. "
            "Likely next steps: regenerate `program.md` with different framing, "
            "or defer this skill to a future round."
        )
    elif v.label == "SKIP":
        lines.append(
            "The proposer chose not to attempt an edit. Re-evaluate after "
            "the next eval run."
        )
    return "\n".join(lines)
