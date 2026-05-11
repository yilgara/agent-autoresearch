"""Step 5 (v3) — atomic-mutation propose loop.

Where v1's propose makes ONE big edit per target, v3 builds the new
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
run a **final combined check** — full critic + full replay against
the cumulative diff (original → final state).

If the combined check fails, **recursively roll back** one accepted
change at a time and re-validate, until either:
- a rolled-back state passes → emit it
- the accepted log empties → emit `skip`

## Why this is more expensive than v1

Per evidence: up to MAX_ATTEMPTS × (1 propose + 1 critic)
= up to ~6 LLM calls per evidence in the worst case.
Plus final combined critic + replay.
Plus rollback re-validation if the final fails.

Cost ≈ 4-10× v1, in exchange for per-change attribution and a stronger
acceptance signal. See README cost section.

## Public contract

`propose()` still returns ONE `ProposeResult` with the final
`new_skill_md`. The atomic-change history is exposed via the
`accepted_log` field for the markdown trace; downstream stages
(critic/replay/verdict at orchestrator level) don't change.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.data import Conversation, Evidence, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v3._common import (
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
    rolled_back_steps: int = 0
    combined_check_passed: bool = False

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
    final_critic: CriticValidator,
    final_replay: ReplayValidator,
    llm: LLMProvider | None = None,
) -> ProposeResult:
    """v3 atomic-mutation propose. Iterates over `target.evidence` and
    builds the new SKILL.md one accepted change at a time.

    The three validator hooks let the orchestrator inject the right
    critic/replay implementations without this file having to import
    them directly. Per-attempt validation is critic-only (cheap gate);
    the full replay only runs at the final combined check.
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

    # ── Final combined check + recursive rollback ───────────────────────────

    rolled_back = 0
    combined_passed = False

    if not accepted_log:
        # Nothing was accepted — propose result is a clean skip.
        return _build_skip_result(
            target.skill_name, last_resp_text, total_in, total_out,
            attempts_log,
            reason="No atomic change passed validation across all evidence.",
        )

    final_state = state
    final_log = list(accepted_log)
    while final_log:
        crit_ok, crit_reason = final_critic(final_state, current_skill_md, None)
        rep_ok, rep_reason = final_replay(final_state, target, conversations)
        if crit_ok and rep_ok:
            combined_passed = True
            break
        # Roll back the most recent accepted change and re-validate
        final_log.pop()
        if not final_log:
            break
        final_state = final_log[-1].new_skill_md or current_skill_md
        rolled_back += 1

    if not final_log:
        # Recursive rollback emptied the log — emit skip
        return _build_skip_result(
            target.skill_name, last_resp_text, total_in, total_out,
            attempts_log,
            accepted_log=accepted_log,   # preserve for trace
            rolled_back_steps=rolled_back,
            reason=(
                "Final combined validation failed; recursive rollback "
                "emptied the accepted log without finding a passing state."
            ),
        )

    # Successful run — accepted_log may have been trimmed by rollback
    summary = _summarize_run(final_log, rolled_back, combined_passed)
    return ProposeResult(
        skill_name=target.skill_name,
        action="edit",
        new_skill_md=final_state,
        reasoning=summary,
        raw_response=last_resp_text,
        input_tokens=total_in or None,
        output_tokens=total_out or None,
        accepted_log=final_log,
        attempts_log=attempts_log,
        rolled_back_steps=rolled_back,
        combined_check_passed=combined_passed,
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
    rolled_back_steps: int = 0,
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
        rolled_back_steps=rolled_back_steps,
        combined_check_passed=False,
    )


# ── Summary text for ProposeResult.reasoning ────────────────────────────────

def _summarize_run(
    accepted_log: list[AtomicAttempt],
    rolled_back: int,
    combined_passed: bool,
) -> str:
    parts = [
        f"v2 atomic-mutation: {len(accepted_log)} change(s) accepted",
    ]
    if rolled_back:
        parts.append(f"{rolled_back} rolled back during final validation")
    if combined_passed:
        parts.append("combined check passed")
    return "; ".join(parts) + "."


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
