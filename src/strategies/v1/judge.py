"""Step 7b — judge. Pick winner between old and new agent replies.

Given the session transcript, the original agent reply at the focus
turn, and the responder's hypothetical reply under the new skill,
the judge LLM picks `new`, `old`, or `tie` with prose reasoning.

The judge defaults to `old` on ambiguity — burden of proof is on the
new skill.

This is strategy v1's implementation. See `prompts/judge.md` next to
this file for the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v1._common import extract_tag

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


# Token cap — short verdict + reasoning
JUDGE_MAX_TOKENS = 600

JudgeWinner = Literal["new", "old", "tie"]


@dataclass
class JudgeResult:
    """Output of run_judge() — which reply wins this comparison."""
    session_id: str
    focus_turn: int
    winner: JudgeWinner
    reasoning: str
    raw_response: str
    input_tokens: int | None
    output_tokens: int | None


# ── Stage entry point ───────────────────────────────────────────────────────

def run_judge(
    session_id: str,
    *,
    focus_turn: int,
    user_message: str,
    transcript: str,
    old_reply: str,
    new_reply: str,
    new_tool_plan: str,
    program_md: str,
    llm: LLMProvider | None = None,
) -> JudgeResult:
    """Step 7b — pick winner for the focus turn.

    `old_reply` is the original agent's reply at the focus turn (read
    from the conversation). `new_reply` and `new_tool_plan` are the
    responder's output. `program_md` provides context on what the
    proposed skill change was supposed to achieve.
    """
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        _PROMPT_PATH,
        transcript=transcript,
        focus_turn=focus_turn,
        user_message=user_message,
        old_reply=old_reply,
        new_reply=new_reply,
        new_tool_plan=new_tool_plan,
        program_md=program_md,
    )

    resp = llm.call(system=system, user=user, max_tokens=JUDGE_MAX_TOKENS)
    winner, reasoning = _parse_response(resp.text)

    return JudgeResult(
        session_id=session_id,
        focus_turn=focus_turn,
        winner=winner,
        reasoning=reasoning,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Response parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[JudgeWinner, str]:
    """Pull <winner>, <reasoning> from the LLM text.

    Defaults to `old` on parse failure — burden of proof is on the
    new skill so unclear judges shouldn't accidentally count as wins.
    """
    winner_raw = (extract_tag(raw, "winner") or "").strip().lower()
    reasoning = extract_tag(raw, "reasoning") or ""

    winner: JudgeWinner
    if winner_raw in ("new", "old", "tie"):
        winner = winner_raw  # type: ignore[assignment]
    else:
        winner = "old"
        if not reasoning:
            reasoning = (
                "Parser could not extract <winner> tag; defaulting to 'old'. "
                f"Raw response (500 chars): {raw[:500]}"
            )

    return winner, reasoning
