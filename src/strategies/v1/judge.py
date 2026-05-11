"""Step 7b — judge. Decide whether the NEW reply passes for this session.

Given the session transcript, the original agent reply at the focus
turn (for context only), and the responder's hypothetical reply under
the new skill, the judge LLM returns a single boolean: does the new
reply adequately handle the user's request at the focus turn?

No comparison vs old. On fix sessions old already failed; on baselines
old already passed — neither tells us anything about whether new
clears the bar.

The judge defaults to `False` on ambiguity — burden of proof is on
the new skill.

This is strategy v1's implementation. See `prompts/judge.md` next to
this file for the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v1._common import extract_tag

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


# Token cap — short verdict + reasoning
JUDGE_MAX_TOKENS = 600


@dataclass
class JudgeResult:
    """Output of run_judge() — does the new reply pass for this session."""
    session_id: str
    focus_turn: int
    new_passes: bool
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
    """Step 7b — decide whether the new reply passes at the focus turn.

    `old_reply` is shown to the judge for context (so it can see what
    actually happened in the session) but doesn't drive the verdict —
    judge `new_reply` on its own merit against the user's request.
    `program_md` describes what the proposed skill change was supposed
    to achieve.
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
    new_passes, reasoning = _parse_response(resp.text)

    return JudgeResult(
        session_id=session_id,
        focus_turn=focus_turn,
        new_passes=new_passes,
        reasoning=reasoning,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Response parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[bool, str]:
    """Pull <new_passes>, <reasoning> from the LLM text.

    Defaults to False on parse failure — burden of proof is on the
    new skill so unclear judges shouldn't accidentally count as a pass.
    """
    new_passes = _parse_bool(extract_tag(raw, "new_passes"))
    reasoning = extract_tag(raw, "reasoning") or ""

    if reasoning == "" and not new_passes:
        reasoning = (
            "Parser could not extract <new_passes> tag; defaulting to False. "
            f"Raw response (500 chars): {raw[:500]}"
        )

    return new_passes, reasoning


def _parse_bool(raw: str | None) -> bool:
    if not raw:
        return False
    return raw.strip().lower() in ("true", "yes", "1", "pass", "passes")
