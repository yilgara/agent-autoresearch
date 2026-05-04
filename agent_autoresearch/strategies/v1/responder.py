"""Step 7a — responder. Generate a hypothetical agent reply under the new skill.

For one session, given the full transcript with a focus turn marked
and the proposed new SKILL.md, the responder LLM produces:
- a `tool_plan` — the tools it would call (in order)
- a `reply` — the user-facing message it would send at the focus turn

This is "soft replay" — the responder doesn't actually call tools.
It narrates what it would do. The downstream judge compares this
hypothetical reply to the original agent's reply at the same turn.

This is strategy v1's implementation. See `prompts/responder.md`
next to this file for the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v1._common import extract_tag

_PROMPT_PATH = Path(__file__).parent / "prompts" / "responder.md"


# Token cap — responder emits one reply + tool plan, not a full document
RESPONDER_MAX_TOKENS = 1500


@dataclass
class ResponderResult:
    """Output of run_responder() — what the new skill would have produced."""
    session_id: str
    focus_turn: int
    tool_plan: str                   # narrated list of tool calls
    reply: str                       # user-facing text
    raw_response: str                # full LLM text in case of parsing issues
    input_tokens: int | None
    output_tokens: int | None


# ── Stage entry point ───────────────────────────────────────────────────────

def run_responder(
    session_id: str,
    *,
    focus_turn: int,
    user_message: str,
    transcript: str,
    new_skill_md: str,
    llm: LLMProvider | None = None,
) -> ResponderResult:
    """Step 7a — generate the hypothetical reply at `focus_turn` under the
    new skill.

    The transcript should be pre-formatted (with the focus turn marked)
    by the caller — `_common.format_session_transcript()` does that.
    """
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        _PROMPT_PATH,
        skill_md=new_skill_md,
        transcript=transcript,
        focus_turn=focus_turn,
        user_message=user_message,
    )

    resp = llm.call(system=system, user=user, max_tokens=RESPONDER_MAX_TOKENS)

    tool_plan = extract_tag(resp.text, "tool_plan") or "(none parsed)"
    # If the response wasn't tagged, fall back to the raw text — better
    # than nothing for the judge to look at.
    reply = extract_tag(resp.text, "reply") or resp.text

    return ResponderResult(
        session_id=session_id,
        focus_turn=focus_turn,
        tool_plan=tool_plan,
        reply=reply,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
