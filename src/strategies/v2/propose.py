"""Step 5 (v2) — atomic-mutation propose loop.

Where v1's propose makes ONE big edit per target, v2 builds the new
SKILL.md incrementally — one atomic change per evidence, validated
piece-by-piece.

## The flow

For each `Evidence` on the Target:

    for attempt in 1..MAX_ATTEMPTS_PER_EVIDENCE:
        change = propose_atomic(evidence, current_state, accepted_log,
                                 previous_attempts_for_this_evidence)
        candidate = apply(change)
        if critic(candidate).approves:
            accept and break to next evidence
        else:
            log the failure, retry

Once all evidence has been processed (or the LLM signaled `done`),
run **one final replay** on the cumulative state. No final critic —
each accepted change was already critic-validated per attempt — and
no rollback — rejected evidence simply doesn't get stacked.

## Why this is more expensive than v1

Per evidence: up to MAX_ATTEMPTS × (1 propose + 1 critic)
= up to ~6 LLM calls per evidence in the worst case.
Plus one final replay over the configured sample.

Cost ≈ 4-10× v1, in exchange for per-change attribution and a stronger
acceptance signal. See README cost section.

## Public contract

`propose()` still returns ONE `ProposeResult` with the final
`new_skill_md`. The atomic-change history is exposed via the
`accepted_log` field for the markdown trace; downstream stages
(critic/replay/verdict at orchestrator level) don't change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.data import Conversation, Evidence, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v2._common import (
    evidence_for_session,
    extract_tag,
    focus_turn_old_reply,
    focus_turn_user,
    format_session_transcript,
    pick_focus_turn,
)


_PROMPT_PATH = Path(__file__).parent / "prompts" / "propose.md"


# ── Tunables ───────────────────────────────────────────────────────────────

PROPOSE_MAX_TOKENS         = 8000
MAX_ATTEMPTS_PER_EVIDENCE  = 3        # rule A: 3 retries before moving on


# ── Action types ────────────────────────────────────────────────────────────

ProposeAction = Literal["edit", "skip"]   # public — same as v1
AtomicAction = Literal["edit", "skip", "done"]


# ── Result types ────────────────────────────────────────────────────────────

@dataclass
class AtomicAttempt:
    """One LLM call inside the per-evidence retry loop.

    Captures the full audit trail — useful for the markdown trace
    and for tests asserting that retries actually retried.
    """
    evidence_index: int                  # which evidence this targeted
    attempt_number: int                  # 1..MAX_ATTEMPTS_PER_EVIDENCE
    action: AtomicAction
    reasoning: str
    new_skill_md: str | None
    raw_response: str
    accepted: bool                       # passed critic + per-iteration replay
    failure_reason: str = ""             # why it didn't pass (if it didn't)


@dataclass
class ProposeResult:
    """Output of propose() — final SKILL.md state + atomic-change log.

    Public fields kept compatible with v1's ProposeResult so the
    orchestrator + verdict.py keep working unchanged. The `accepted_log`
    + `attempts_log` fields are v2-additive.
    """
    skill_name: str
    action: ProposeAction            # 'edit' or 'skip'
    new_skill_md: str | None         # populated only when action == 'edit'
    reasoning: str                   # human-readable summary of the run
    raw_response: str                # last LLM raw response (compat field)
    input_tokens: int | None
    output_tokens: int | None

    # v2-additive
    accepted_log: list[AtomicAttempt] = field(default_factory=list)
    attempts_log: list[AtomicAttempt] = field(default_factory=list)

    @property
    def is_edit(self) -> bool:
        return self.action == "edit" and bool(self.new_skill_md)


# ── Validator hooks (injected by the caller — keeps propose pure-ish) ───────

# Each validator returns (passed: bool, reason: str). Reason shows up
# in the failed-attempts log fed back into the next LLM call so it can
# course-correct.

CriticValidator = Any   # callable: (candidate_md, current_md, evidence) → (bool, str)
ReplayValidator = Any   # callable: (candidate_md, evidence, conversation) → (bool, str)


# ── Public entry point ──────────────────────────────────────────────────────

def propose(
    target: Target,
    *,
    current_skill_md: str,
    program_md: str,
    conversations: dict[str, Conversation],
    critic_per_attempt: CriticValidator,
    final_replay: ReplayValidator,
    llm: LLMProvider | None = None,
) -> ProposeResult:
    """v2 atomic-mutation propose. Iterates over `target.evidence` and
    builds the new SKILL.md one accepted change at a time.

    Per-attempt validation is critic-only (cheap gate). After the loop
    finishes, the full replay runs once on the cumulative state — the
    orchestrator reuses its result so no canonical re-run is needed.
    There is no final critic call (per-attempt critics already
    validated each accepted change) and no rollback (a rejected
    evidence simply isn't added; previously-accepted changes stand).
    """
    llm = llm or default_llm_provider()

    state = current_skill_md
    accepted_log: list[AtomicAttempt] = []
    attempts_log: list[AtomicAttempt] = []
    last_resp_text = ""
    total_in = 0
    total_out = 0
    early_done = False

    # ── Per-evidence loop ───────────────────────────────────────────────────

    for ev_idx, evidence in enumerate(target.evidence):
        if early_done:
            break

        previous_failed_attempts: list[AtomicAttempt] = []

        for attempt_num in range(1, MAX_ATTEMPTS_PER_EVIDENCE + 1):
            attempt = _propose_atomic(
                evidence_index=ev_idx,
                evidence=evidence,
                attempt_number=attempt_num,
                program_md=program_md,
                current_state=state,
                accepted_log=accepted_log,
                previous_failed_attempts=previous_failed_attempts,
                llm=llm,
            )
            attempts_log.append(attempt)
            last_resp_text = attempt.raw_response
            total_in += attempt._in_tokens or 0
            total_out += attempt._out_tokens or 0

            if attempt.action == "done":
                early_done = True
                break

            if attempt.action == "skip":
                # LLM says this evidence doesn't need addressing — move on
                # to the next evidence. (Distinguish from "skip the whole
                # propose stage" — that only happens if 0 evidence
                # produced an accepted change.)
                attempt.failure_reason = "LLM skipped this evidence"
                break

            # action == "edit" — validate
            candidate = attempt.new_skill_md or ""
            if not candidate.strip():
                attempt.failure_reason = "Empty new_skill_md"
                previous_failed_attempts.append(attempt)
                continue

            crit_ok, crit_reason = critic_per_attempt(candidate, state, evidence)
            if not crit_ok:
                attempt.failure_reason = f"Critic rejected: {crit_reason}"
                previous_failed_attempts.append(attempt)
                continue

            # Critic gate passed — accept and move to next evidence
            attempt.accepted = True
            accepted_log.append(attempt)
            state = candidate
            break

    # ── Final replay (no critic, no rollback) ───────────────────────────────

    if not accepted_log:
        # Nothing was accepted — propose result is a clean skip.
        return _build_skip_result(
            target.skill_name, last_resp_text, total_in, total_out,
            attempts_log,
            reason="No atomic change passed validation across all evidence.",
        )

    # One full-sample replay over the cumulative state. The validator
    # closure stores the ReplayResult in the orchestrator's captures
    # dict so verdict can read it without re-running.
    final_replay(state, target, conversations)

    summary = _summarize_run(accepted_log)
    return ProposeResult(
        skill_name=target.skill_name,
        action="edit",
        new_skill_md=state,
        reasoning=summary,
        raw_response=last_resp_text,
        input_tokens=total_in or None,
        output_tokens=total_out or None,
        accepted_log=accepted_log,
        attempts_log=attempts_log,
    )


# ── One LLM call (one attempt for one evidence) ─────────────────────────────

def _propose_atomic(
    *,
    evidence_index: int,
    evidence: Evidence,
    attempt_number: int,
    program_md: str,
    current_state: str,
    accepted_log: list[AtomicAttempt],
    previous_failed_attempts: list[AtomicAttempt],
    llm: LLMProvider,
) -> AtomicAttempt:
    """Make one LLM call for one (evidence, attempt) pair."""
    details = evidence.details or {}
    system, user = format_prompt(
        _PROMPT_PATH,
        program_md=program_md,
        current_skill_md=current_state,
        evidence_category=evidence.category,
        evidence_session_id=details.get("session_id", "(none)"),
        evidence_focus_turn=details.get("focus_turn", "(unknown)"),
        evidence_summary=details.get("summary", ""),
        accepted_log_block=_format_accepted_log(accepted_log),
        previous_attempts_block=_format_previous_attempts(previous_failed_attempts),
    )

    resp = llm.call(system=system, user=user, max_tokens=PROPOSE_MAX_TOKENS)
    action, reasoning, new_md = _parse_response(resp.text)

    attempt = AtomicAttempt(
        evidence_index=evidence_index,
        attempt_number=attempt_number,
        action=action,
        reasoning=reasoning,
        new_skill_md=new_md,
        raw_response=resp.text,
        accepted=False,
    )
    # Stash token counts for aggregation (private — not part of the dataclass schema)
    attempt._in_tokens = resp.input_tokens   # type: ignore[attr-defined]
    attempt._out_tokens = resp.output_tokens   # type: ignore[attr-defined]
    return attempt


# ── Prompt-block formatters ─────────────────────────────────────────────────

def _format_accepted_log(log: list[AtomicAttempt]) -> str:
    if not log:
        return "_(none yet — this is the first iteration.)_"
    lines = []
    for i, a in enumerate(log, 1):
        lines.append(f"{i}. Evidence #{a.evidence_index}: {a.reasoning}")
    return "\n".join(lines)


def _format_previous_attempts(failed: list[AtomicAttempt]) -> str:
    if not failed:
        return "_(none — this is your first attempt for this evidence.)_"
    lines = []
    for a in failed:
        lines.append(f"- Attempt {a.attempt_number}: {a.reasoning}")
        lines.append(f"  → failed: {a.failure_reason}")
    return "\n".join(lines)


# ── Skip-result helper ──────────────────────────────────────────────────────

def _build_skip_result(
    skill_name: str,
    last_resp: str,
    total_in: int,
    total_out: int,
    attempts_log: list[AtomicAttempt],
    *,
    accepted_log: list[AtomicAttempt] | None = None,
    reason: str,
) -> ProposeResult:
    return ProposeResult(
        skill_name=skill_name,
        action="skip",
        new_skill_md=None,
        reasoning=reason,
        raw_response=last_resp,
        input_tokens=total_in or None,
        output_tokens=total_out or None,
        accepted_log=accepted_log or [],
        attempts_log=attempts_log,
    )


# ── Summary text for ProposeResult.reasoning ────────────────────────────────

def _summarize_run(accepted_log: list[AtomicAttempt]) -> str:
    return (
        f"v2 atomic-mutation: {len(accepted_log)} change(s) accepted "
        "via per-attempt critic gating."
    )


# ── Response parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[AtomicAction, str, str | None]:
    """Pull <action>, <reasoning>, <new_skill_md> from the LLM text.

    Falls back to action='skip' on parse failures so a malformed
    response doesn't pretend to be a real edit.
    """
    raw_action = (extract_tag(raw, "action") or "").lower().strip()
    reasoning = extract_tag(raw, "reasoning") or ""
    new_md = extract_tag(raw, "new_skill_md")

    action: AtomicAction
    if raw_action in ("edit", "skip", "done"):
        action = raw_action  # type: ignore[assignment]
    elif new_md and len(new_md) > 200:
        # LLM forgot the tag but emitted a full body — treat as edit
        action = "edit"
    else:
        action = "skip"
        if not reasoning:
            reasoning = (
                "Parser could not extract <action>/<reasoning> tags; "
                f"raw response truncated to 500 chars: {raw[:500]}"
            )

    if action != "edit":
        new_md = None

    return action, reasoning, new_md
