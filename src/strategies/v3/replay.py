"""Step 7 (v3) — Soft replay with new_passes + per-axis rubric + checks aggregation.

Same shape as v1/v2: for each session in `target.fix_session_ids +
regression_baseline_ids`, run the responder (one hypothetical reply
under the new skill) and the judge (3-signal verdict at the focus
turn).

v3 metric semantics — each rate is computed over a specific population:

  * `fix_rate`                 — over FIX sessions only. Fraction where
                                  `new_passes == True`. No comparison vs
                                  old (old already failed). Informational
                                  for the verdict (no acceptance gate).

  * `regression_rate`          — over BASELINE sessions only. Fraction
                                  where `new_passes == True`. No
                                  comparison vs old (old already passed);
                                  we just need new to also pass.
                                  Acceptance threshold: ≥ 0.90.

  * `binary_checks_pass_rate`  — over BASELINE sessions × checks.
                                  Fraction of (session, check) pairs
                                  where check is `pass` or `na`. The
                                  new prompt must preserve invariants on
                                  sessions that already worked.
                                  Acceptance threshold: ≥ 0.90.

  * `rubric_score`             — over FIX sessions × axes. Mean of
                                  per-vote scores (+1 if new, 0 if tie,
                                  -1 if old). Range [-1, +1]. Positive
                                  means new beat old on average across
                                  axes on the broken sessions.
                                  Acceptance threshold: ≥ 0.

`SessionReplay` carries the per-session structured signals so
`replay.md` and `verdict.md` can show what actually happened on each
session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_autoresearch.core.data import Conversation, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v3._common import (
    evidence_for_session,
    focus_turn_old_reply,
    focus_turn_user,
    format_session_transcript,
    pick_focus_turn,
    truncate,
)
from agent_autoresearch.strategies.v3.judge import (
    CheckOutcome,
    JudgeResult,
    RubricVote,
    run_judge,
)
from agent_autoresearch.strategies.v3.program import BinaryCheck, RubricAxis
from agent_autoresearch.strategies.v3.responder import (
    ResponderResult,
    run_responder,
)


# Default sample sizes — replay is the most expensive step.
DEFAULT_FIX_SAMPLE      = 3
DEFAULT_BASELINE_SAMPLE = 3


SessionRole = Literal["fix_target", "baseline"]


# ── Per-session result ──────────────────────────────────────────────────────

@dataclass
class SessionReplay:
    """Per-session outcome: responder output + 3 judge signals."""
    session_id: str
    role: SessionRole
    focus_turn: int
    user_message: str
    old_reply: str
    new_tool_plan: str
    new_reply: str

    # Primary signal — does the new reply adequately handle this session?
    new_passes: bool = False

    # v3 structured signals
    rubric_votes: list[RubricVote] = field(default_factory=list)
    check_results: list[CheckOutcome] = field(default_factory=list)

    judge_reasoning: str = ""
    responder_tokens: tuple[int | None, int | None] = (None, None)
    judge_tokens: tuple[int | None, int | None] = (None, None)
    error: str | None = None


# ── Aggregate result ────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """v3 replay output — 4 aggregate rates + per-session detail.

    Rate populations differ per metric — see module docstring.
    """
    skill_name: str
    fix_target_replays: list[SessionReplay] = field(default_factory=list)
    regression_replays: list[SessionReplay] = field(default_factory=list)

    # ── 4 aggregate rates ──────────────────────────────────────────────────

    @property
    def fix_passes(self) -> int:
        return sum(1 for r in self.fix_target_replays if r.new_passes)

    @property
    def fix_rate(self) -> float:
        n = len(self.fix_target_replays)
        return (self.fix_passes / n) if n else 0.0

    @property
    def baseline_passes(self) -> int:
        return sum(1 for r in self.regression_replays if r.new_passes)

    @property
    def regression_rate(self) -> float:
        n = len(self.regression_replays)
        return (self.baseline_passes / n) if n else 1.0

    @property
    def binary_checks_pass_rate(self) -> float:
        """Over (baseline session × check) pairs, fraction passing.

        `na` counts as pass (the invariant didn't apply). Returns 1.0
        when there are no baseline replays or no checks (nothing to
        fail).
        """
        total = 0
        passed = 0
        for r in self.regression_replays:
            for c in r.check_results:
                total += 1
                if c.is_pass:
                    passed += 1
        return (passed / total) if total else 1.0

    @property
    def rubric_score(self) -> float:
        """Over (fix session × axis) pairs, mean of +1/0/-1 votes.

        Returns 0.0 when there are no fix replays or no axes (neutral).
        """
        total = 0
        score_sum = 0
        for r in self.fix_target_replays:
            for v in r.rubric_votes:
                total += 1
                score_sum += v.score
        return (score_sum / total) if total else 0.0

    # ── Backward-compat aliases (v1/v2 names) ──────────────────────────────

    @property
    def fix_target_score(self) -> float:
        return self.fix_rate

    @property
    def regression_score(self) -> float:
        return self.regression_rate

    # ── Token counters ─────────────────────────────────────────────────────

    @property
    def total_input_tokens(self) -> int:
        return _sum_tokens(self._all_replays(), idx=0)

    @property
    def total_output_tokens(self) -> int:
        return _sum_tokens(self._all_replays(), idx=1)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _all_replays(self) -> list[SessionReplay]:
        return self.fix_target_replays + self.regression_replays

    def to_markdown(self) -> str:
        return _render_replay_md(self)


def _sum_tokens(replays: list[SessionReplay], *, idx: int) -> int:
    total = 0
    for r in replays:
        for tokens in (r.responder_tokens, r.judge_tokens):
            v = tokens[idx]
            if v:
                total += v
    return total


# ── Per-session orchestrator ────────────────────────────────────────────────

def _replay_one_session(
    *,
    target: Target,
    conversation: Conversation,
    role: SessionRole,
    new_skill_md: str,
    program_md: str,
    rubric_axes: list[RubricAxis],
    binary_checks: list[BinaryCheck],
    llm: LLMProvider,
) -> SessionReplay:
    """Run responder + judge for one session.

    Errors are caught and surfaced via `SessionReplay.error`. On
    failure, `new_passes=False`, rubric votes default to `tie`, checks
    default to `fail` — safe defaults that don't optimistically count
    a broken session as a pass.
    """
    sid = conversation.session_id
    evidence = evidence_for_session(target.evidence, sid)
    focus_turn = pick_focus_turn(conversation, evidence)
    transcript = format_session_transcript(conversation, focus_turn=focus_turn)
    user_message = focus_turn_user(conversation, focus_turn)
    old_reply = focus_turn_old_reply(conversation, focus_turn)

    # Step 7a — responder
    try:
        resp = run_responder(
            sid,
            focus_turn=focus_turn,
            user_message=user_message,
            transcript=transcript,
            new_skill_md=new_skill_md,
            llm=llm,
        )
    except Exception as exc:
        return SessionReplay(
            session_id=sid, role=role, focus_turn=focus_turn,
            user_message=user_message, old_reply=old_reply,
            new_tool_plan="", new_reply="",
            new_passes=False,
            error=f"responder failed: {type(exc).__name__}: {exc}",
        )

    # Step 7b — judge (3 signals)
    try:
        judg: JudgeResult = run_judge(
            sid,
            focus_turn=focus_turn,
            user_message=user_message,
            transcript=transcript,
            old_reply=old_reply,
            new_reply=resp.reply,
            new_tool_plan=resp.tool_plan,
            program_md=program_md,
            rubric_axes=rubric_axes,
            binary_checks=binary_checks,
            llm=llm,
        )
    except Exception as exc:
        return SessionReplay(
            session_id=sid, role=role, focus_turn=focus_turn,
            user_message=user_message, old_reply=old_reply,
            new_tool_plan=resp.tool_plan, new_reply=resp.reply,
            new_passes=False,
            responder_tokens=(resp.input_tokens, resp.output_tokens),
            error=f"judge failed: {type(exc).__name__}: {exc}",
        )

    return SessionReplay(
        session_id=sid, role=role, focus_turn=focus_turn,
        user_message=user_message,
        old_reply=old_reply,
        new_tool_plan=resp.tool_plan,
        new_reply=resp.reply,
        new_passes=judg.new_passes,
        rubric_votes=judg.rubric_votes,
        check_results=judg.check_results,
        judge_reasoning=judg.reasoning,
        responder_tokens=(resp.input_tokens, resp.output_tokens),
        judge_tokens=(judg.input_tokens, judg.output_tokens),
    )


# ── Top-level entry point ───────────────────────────────────────────────────

def soft_replay(
    target: Target,
    *,
    new_skill_md: str,
    program_md: str,
    conversations: dict[str, Conversation],
    rubric_axes: list[RubricAxis],
    binary_checks: list[BinaryCheck],
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    llm: LLMProvider | None = None,
) -> ReplayResult:
    """Replay a sample of fix-targets + baselines with v3 judge signals.

    `rubric_axes` and `binary_checks` come from the v3 `ProgramResult`.
    They flow through to every judge call so each session is scored
    consistently against the same criteria.
    """
    llm = llm or default_llm_provider()
    result = ReplayResult(skill_name=target.skill_name)

    for sid in target.fix_session_ids[:fix_sample]:
        conv = conversations.get(sid)
        if conv is None:
            continue
        result.fix_target_replays.append(_replay_one_session(
            target=target, conversation=conv, role="fix_target",
            new_skill_md=new_skill_md, program_md=program_md,
            rubric_axes=rubric_axes, binary_checks=binary_checks, llm=llm,
        ))

    for sid in target.regression_baseline_ids[:baseline_sample]:
        conv = conversations.get(sid)
        if conv is None:
            continue
        result.regression_replays.append(_replay_one_session(
            target=target, conversation=conv, role="baseline",
            new_skill_md=new_skill_md, program_md=program_md,
            rubric_axes=rubric_axes, binary_checks=binary_checks, llm=llm,
        ))

    return result


# ── Markdown rendering for replay.md ────────────────────────────────────────

def _pass_badge(passed: bool) -> str:
    return "🟢 new_passes" if passed else "🔴 new fails"


def _rubric_badge(winner: str) -> str:
    return {"new": "🟢 new", "old": "🔴 old", "tie": "🟡 tie"}.get(winner, winner)


def _render_replay_md(r: ReplayResult) -> str:
    n_fix = len(r.fix_target_replays)
    n_base = len(r.regression_replays)
    lines = [
        f"# Soft-replay results — {r.skill_name}",
        "",
        f"**fix_rate:**         {r.fix_passes}/{n_fix}  ({r.fix_rate:.0%}) — "
        f"new passes on this fraction of fix sessions (informational)",
        f"**regression_rate:**  {r.baseline_passes}/{n_base}  ({r.regression_rate:.0%}) — "
        f"new passes on this fraction of baseline sessions",
        f"**binary_checks_pass_rate:** "
        f"({r.binary_checks_pass_rate:.0%}) — "
        f"fraction of (baseline session × check) pairs passing",
        f"**rubric_score:** {r.rubric_score:+.2f} — "
        f"mean of +1/0/-1 votes over (fix session × axis) pairs",
        "",
        f"**Replay tokens:** {r.total_input_tokens:,} in / "
        f"{r.total_output_tokens:,} out",
        "",
    ]

    def _section(title: str, replays: list[SessionReplay]) -> list[str]:
        out = [f"## {title} ({len(replays)} sessions)", ""]
        if not replays:
            out.append("_(none)_")
            out.append("")
            return out
        for s in replays:
            out += [
                f"### `{s.session_id}` (turn {s.focus_turn}) — {_pass_badge(s.new_passes)}",
                "",
                f"**User:** `{truncate(s.user_message, 300)}`",
                "",
                "**Old reply:**",
                "",
                "```",
                truncate(s.old_reply, 700),
                "```",
                "",
                "**New tool plan (under proposed skill):**",
                "```",
                truncate(s.new_tool_plan, 500),
                "```",
                "",
                "**New reply:**",
                "",
                "```",
                truncate(s.new_reply, 700),
                "```",
                "",
            ]
            if s.rubric_votes:
                out.append("**Rubric votes (new/tie/old):**")
                out.append("")
                for v in s.rubric_votes:
                    out.append(f"- `{v.name}`: {_rubric_badge(v.winner)} ({v.score:+d})")
                out.append("")
            if s.check_results:
                out.append("**Binary checks:**")
                out.append("")
                for c in s.check_results:
                    out.append(f"- check `{c.id}`: `{c.result}`")
                out.append("")
            out.append(f"**Judge reasoning:** {s.judge_reasoning}")
            out.append("")
            if s.error:
                out.append(f"⚠ Error during this replay: `{s.error}`")
                out.append("")
        return out

    lines += _section("Fix-target replays", r.fix_target_replays)
    lines += _section("Regression-baseline replays", r.regression_replays)
    return "\n".join(lines)
