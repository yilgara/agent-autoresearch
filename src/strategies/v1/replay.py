"""Step 7 — Soft replay orchestrator (Validation Layer B).

For each session in `target.fix_session_ids + regression_baseline_ids`:

  1. Look up the conversation by session_id
  2. Pick the focus turn (from evidence if tagged, else last turn)
  3. Format the transcript with the focus turn marked
  4. Run the responder LLM → hypothetical reply under the new skill
  5. Run the judge LLM → pick `new`, `old`, or `tie`

Aggregate scores:
  - `fix_target_score`    — % of fix sessions where `new` won
  - `regression_score`    — % of baselines where `new` won OR tied
                            (i.e. didn't break what was working)

Returns a `ReplayResult` carrying per-session detail + the two scores.
The orchestrator writes `replay.md` per target via `to_markdown()`.

THIS IS NOT TRUE REPLAY. The responder is one LLM imagining what
another would do, judged by a third. It catches form problems and
clear substance regressions, but it cannot run real tools or see
what real APIs would return. For real ground truth, your eval
pipeline tomorrow tells you whether the accepted edit moved the
score in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_autoresearch.core.data import Conversation, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v1._common import (
    evidence_for_session,
    focus_turn_old_reply,
    focus_turn_user,
    format_session_transcript,
    pick_focus_turn,
    truncate,
)
from agent_autoresearch.strategies.v1.judge import (
    JudgeResult,
    run_judge,
)
from agent_autoresearch.strategies.v1.responder import (
    ResponderResult,
    run_responder,
)


# Default sample sizes — keep small, replay is the most expensive step.
DEFAULT_FIX_SAMPLE      = 3
DEFAULT_BASELINE_SAMPLE = 3


SessionRole = Literal["fix_target", "baseline"]


# ── Per-session result ──────────────────────────────────────────────────────

@dataclass
class SessionReplay:
    """Result of replaying one session — both LLM calls + verdict."""
    session_id: str
    role: SessionRole
    focus_turn: int
    user_message: str
    old_reply: str
    new_tool_plan: str
    new_reply: str
    new_passes: bool
    judge_reasoning: str
    responder_tokens: tuple[int | None, int | None]   # (input, output)
    judge_tokens: tuple[int | None, int | None]
    error: str | None = None


# ── Aggregate result ────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """Output of soft_replay() — per-session detail + aggregate scores.

    Each rate is computed over a specific population:
      * fix_target_score   — fraction of FIX sessions where new passes
                             (old already failed; no comparison needed)
      * regression_score   — fraction of BASELINE sessions where new
                             passes (old already passed; we just need
                             new to also pass)
    """
    skill_name: str
    fix_target_replays: list[SessionReplay] = field(default_factory=list)
    regression_replays: list[SessionReplay] = field(default_factory=list)

    @property
    def fix_target_passes(self) -> int:
        return sum(1 for r in self.fix_target_replays if r.new_passes)

    @property
    def fix_target_score(self) -> float:
        n = len(self.fix_target_replays)
        return (self.fix_target_passes / n) if n else 0.0

    @property
    def baseline_passes(self) -> int:
        return sum(1 for r in self.regression_replays if r.new_passes)

    @property
    def regression_score(self) -> float:
        n = len(self.regression_replays)
        return (self.baseline_passes / n) if n else 1.0  # no baselines = no risk

    @property
    def total_input_tokens(self) -> int:
        return _sum_tokens(self.fix_target_replays + self.regression_replays, idx=0)

    @property
    def total_output_tokens(self) -> int:
        return _sum_tokens(self.fix_target_replays + self.regression_replays, idx=1)

    def to_markdown(self) -> str:
        """Render `replay.md` — what gets written to the per-target output folder."""
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
    llm: LLMProvider,
) -> SessionReplay:
    """Run responder + judge for one session.

    Errors from either LLM call are caught and surfaced via
    `SessionReplay.error` so a single bad session doesn't abort the
    whole replay batch. On error the winner defaults to 'old' (burden
    of proof rule).
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
            new_passes=False, judge_reasoning="",
            responder_tokens=(None, None), judge_tokens=(None, None),
            error=f"responder failed: {type(exc).__name__}: {exc}",
        )

    # Step 7b — judge
    try:
        judg = run_judge(
            sid,
            focus_turn=focus_turn,
            user_message=user_message,
            transcript=transcript,
            old_reply=old_reply,
            new_reply=resp.reply,
            new_tool_plan=resp.tool_plan,
            program_md=program_md,
            llm=llm,
        )
    except Exception as exc:
        return SessionReplay(
            session_id=sid, role=role, focus_turn=focus_turn,
            user_message=user_message, old_reply=old_reply,
            new_tool_plan=resp.tool_plan, new_reply=resp.reply,
            new_passes=False, judge_reasoning="",
            responder_tokens=(resp.input_tokens, resp.output_tokens),
            judge_tokens=(None, None),
            error=f"judge failed: {type(exc).__name__}: {exc}",
        )

    return SessionReplay(
        session_id=sid, role=role, focus_turn=focus_turn,
        user_message=user_message,
        old_reply=old_reply,
        new_tool_plan=resp.tool_plan,
        new_reply=resp.reply,
        new_passes=judg.new_passes,
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
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    llm: LLMProvider | None = None,
) -> ReplayResult:
    """Step 7 — replay a sample of fix-targets + baselines.

    `conversations` is a dict `{session_id: Conversation}` — typically
    built by the orchestrator from `Adapter.load_conversations()`.
    Sessions whose IDs aren't in the dict are silently skipped (we
    can't replay something we don't have a transcript for).

    Sampling: take the first N from each list. The adapter is
    expected to surface them in priority order; if not, callers can
    sort `target.fix_session_ids` / `regression_baseline_ids` before
    invoking.
    """
    llm = llm or default_llm_provider()
    result = ReplayResult(skill_name=target.skill_name)

    # Fix targets — sessions where the skill broke
    fix_ids = target.fix_session_ids[:fix_sample]
    for sid in fix_ids:
        conv = conversations.get(sid)
        if conv is None:
            continue
        result.fix_target_replays.append(_replay_one_session(
            target=target, conversation=conv, role="fix_target",
            new_skill_md=new_skill_md, program_md=program_md, llm=llm,
        ))

    # Regression baselines — sessions where the skill worked
    baseline_ids = target.regression_baseline_ids[:baseline_sample]
    for sid in baseline_ids:
        conv = conversations.get(sid)
        if conv is None:
            continue
        result.regression_replays.append(_replay_one_session(
            target=target, conversation=conv, role="baseline",
            new_skill_md=new_skill_md, program_md=program_md, llm=llm,
        ))

    return result


# ── Markdown rendering for replay.md ────────────────────────────────────────

def _pass_badge(passed: bool) -> str:
    return "🟢 new_passes" if passed else "🔴 new fails"


def _render_replay_md(r: ReplayResult) -> str:
    lines = [
        f"# Soft-replay results — {r.skill_name}",
        "",
        f"**fix_target_score:** {r.fix_target_passes}/"
        f"{len(r.fix_target_replays)}  ({r.fix_target_score:.0%}) — "
        f"new passes on this fraction of fix sessions",
        "",
        f"**regression_score:** {r.baseline_passes}/"
        f"{len(r.regression_replays)}  ({r.regression_score:.0%}) — "
        f"new passes on this fraction of baseline sessions",
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
                f"**Judge reasoning:** {s.judge_reasoning}",
                "",
            ]
            if s.error:
                out.append(f"⚠ Error during this replay: `{s.error}`")
                out.append("")
        return out

    lines += _section("Fix-target replays", r.fix_target_replays)
    lines += _section("Regression-baseline replays", r.regression_replays)
    return "\n".join(lines)
