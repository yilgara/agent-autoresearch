"""Step 7 (v3) — Soft replay with rubric + binary-check aggregation.

Same shape as v1/v2: for each session in `target.fix_session_ids +
regression_baseline_ids`, run the responder (one hypothetical reply
under the new skill) and the judge (compare old vs new at the focus
turn).

v3 differences:
  - The judge call now takes `rubric_axes` + `binary_checks` from
    program.md and produces three signals per session
    (winner / per-axis rubric scores / per-check results).
  - `ReplayResult` exposes 4 aggregate rates instead of 2:
      * `fix_rate`                 (new wins on fix sessions)
      * `regression_rate`          (new doesn't lose on baselines)
      * `rubric_improvement_rate`  (avg new ≥ avg old on a session,
                                    "improved" on fixes, "non-regressed"
                                    on baselines)
      * `binary_checks_pass_rate`  (every check passed on a session;
                                    `na` is treated as pass)

  - `SessionReplay` carries the per-session structured signals so
    `replay.md` and `verdict.md` can show what actually happened on
    each session, not just the win/loss.

`soft_replay()` writes `replay.md` per target via `to_markdown()`,
unchanged in shape; the new sections (rubric / checks) are appended
to the per-session blocks.
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
    JudgeWinner,
    RubricScore,
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

    # v1/v2 winner — kept for fix_rate / regression_rate
    winner: JudgeWinner

    # v3-additive — populated when the judge has rubric/checks
    rubric_scores: list[RubricScore] = field(default_factory=list)
    check_results: list[CheckOutcome] = field(default_factory=list)

    judge_reasoning: str = ""
    responder_tokens: tuple[int | None, int | None] = (None, None)
    judge_tokens: tuple[int | None, int | None] = (None, None)
    error: str | None = None

    # ── per-session derived ────────────────────────────────────────────────

    @property
    def avg_new_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return sum(s.new for s in self.rubric_scores) / len(self.rubric_scores)

    @property
    def avg_old_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return sum(s.old for s in self.rubric_scores) / len(self.rubric_scores)

    @property
    def rubric_session_ok(self) -> bool:
        """True when:
          - on fix sessions: avg new strictly > avg old (improved)
          - on baselines:    avg new >= avg old      (non-regressed)
        Sessions with no rubric scores default to True (nothing to fail).
        """
        if not self.rubric_scores:
            return True
        if self.role == "fix_target":
            return self.avg_new_score > self.avg_old_score
        return self.avg_new_score >= self.avg_old_score

    @property
    def all_checks_pass(self) -> bool:
        return all(c.is_pass for c in self.check_results)


# ── Aggregate result ────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """v3 replay output — 4 aggregate rates + per-session detail.

    Backward-compat: `fix_target_score` and `regression_score`
    properties keep their v1/v2 names (mapped to `fix_rate` and
    `regression_rate`) so v1/v2 verdict logic still reads them if
    a v3 ReplayResult is fed into v1/v2 verdict by mistake.
    """
    skill_name: str
    fix_target_replays: list[SessionReplay] = field(default_factory=list)
    regression_replays: list[SessionReplay] = field(default_factory=list)

    # ── 4 aggregate rates ──────────────────────────────────────────────────

    @property
    def fix_target_wins(self) -> int:
        return sum(1 for r in self.fix_target_replays if r.winner == "new")

    @property
    def fix_rate(self) -> float:
        n = len(self.fix_target_replays)
        return (self.fix_target_wins / n) if n else 0.0

    @property
    def regression_safe(self) -> int:
        return sum(1 for r in self.regression_replays
                   if r.winner in ("new", "tie"))

    @property
    def regression_rate(self) -> float:
        n = len(self.regression_replays)
        return (self.regression_safe / n) if n else 1.0

    @property
    def rubric_session_oks(self) -> int:
        return sum(1 for r in self._all_replays() if r.rubric_session_ok)

    @property
    def rubric_improvement_rate(self) -> float:
        n = len(self._all_replays())
        return (self.rubric_session_oks / n) if n else 1.0

    @property
    def all_checks_pass_count(self) -> int:
        return sum(1 for r in self._all_replays() if r.all_checks_pass)

    @property
    def binary_checks_pass_rate(self) -> float:
        n = len(self._all_replays())
        return (self.all_checks_pass_count / n) if n else 1.0

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

    Errors from either LLM call are caught and surfaced via
    `SessionReplay.error`. Default winner is `old` and rubric/checks
    are left empty, which downstream aggregations treat as "no
    improvement / no signal" — safe default.
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
            winner="old",
            error=f"responder failed: {type(exc).__name__}: {exc}",
        )

    # Step 7b — judge (now produces 3 signals)
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
            winner="old",
            responder_tokens=(resp.input_tokens, resp.output_tokens),
            error=f"judge failed: {type(exc).__name__}: {exc}",
        )

    return SessionReplay(
        session_id=sid, role=role, focus_turn=focus_turn,
        user_message=user_message,
        old_reply=old_reply,
        new_tool_plan=resp.tool_plan,
        new_reply=resp.reply,
        winner=judg.winner,
        rubric_scores=judg.rubric_scores,
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

_WINNER_BADGE = {"new": "🟢 new", "old": "🔴 old", "tie": "🟡 tie"}


def _render_replay_md(r: ReplayResult) -> str:
    lines = [
        f"# Soft-replay results — {r.skill_name}",
        "",
        f"**fix_rate:**         {r.fix_target_wins}/"
        f"{len(r.fix_target_replays)}  ({r.fix_rate:.0%}) — "
        f"new won on this fraction of fix sessions",
        f"**regression_rate:**  {r.regression_safe}/"
        f"{len(r.regression_replays)}  ({r.regression_rate:.0%}) — "
        f"new kept-or-improved on this fraction of baselines",
        f"**rubric_improvement_rate:** "
        f"{r.rubric_session_oks}/{len(r._all_replays())} "
        f"({r.rubric_improvement_rate:.0%}) — "
        f"per-session rubric average new > old (fixes) or new ≥ old (baselines)",
        f"**binary_checks_pass_rate:** "
        f"{r.all_checks_pass_count}/{len(r._all_replays())} "
        f"({r.binary_checks_pass_rate:.0%}) — "
        f"sessions where every binary check passed (or was n/a)",
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
            badge = _WINNER_BADGE.get(s.winner, s.winner)
            out += [
                f"### `{s.session_id}` (turn {s.focus_turn}) — winner: {badge}",
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
            if s.rubric_scores:
                out.append("**Rubric scores (1–3):**")
                out.append("")
                out.append("| axis | new | old |")
                out.append("|------|----:|----:|")
                for sc in s.rubric_scores:
                    out.append(f"| {sc.name} | {sc.new} | {sc.old} |")
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
