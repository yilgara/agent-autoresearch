"""Step 5 — propose. Apply the strategy to produce a new SKILL.md.

Reads the current SKILL.md + the program.md strategy from step 4,
calls the `propose` prompt, returns a `ProposeResult` carrying the
action ('edit' or 'skip') and the new SKILL.md content if editing.

The prompt asks for XML tags so the SKILL.md content stays clean
markdown (no JSON-escaping). Parser is forgiving: malformed output
defaults to 'skip' with a diagnostic in `reasoning` so the human
reviewer still gets visibility.

See `agent_autoresearch/prompts/propose.md` for the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.prompts._loader import format_prompt
from agent_autoresearch.stages._common import extract_tag


# Token cap — may emit a full revised SKILL.md (up to ~20 KB)
PROPOSE_MAX_TOKENS = 8000

ProposeAction = Literal["edit", "skip"]


@dataclass
class ProposeResult:
    """Output of propose() — applied edit or skip with reasoning."""
    skill_name: str
    action: ProposeAction            # 'edit' or 'skip'
    new_skill_md: str | None         # populated only when action == 'edit'
    reasoning: str
    raw_response: str
    input_tokens: int | None
    output_tokens: int | None

    @property
    def is_edit(self) -> bool:
        return self.action == "edit" and bool(self.new_skill_md)


# ── Stage entry point ───────────────────────────────────────────────────────

def propose(
    skill_name: str,
    *,
    current_skill_md: str,
    program_md: str,
    llm: LLMProvider | None = None,
) -> ProposeResult:
    """Step 5 — apply the strategy doc to produce a new SKILL.md.

    Returns a `ProposeResult`. If `action == "skip"`, `new_skill_md`
    is None and the run goes straight to verdict = SKIP. If
    `action == "edit"`, the orchestrator passes `new_skill_md` to
    step 6 (critic) and step 7 (replay) for validation.
    """
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        "propose",
        program_md=program_md,
        current_skill_md=current_skill_md,
    )

    resp = llm.call(system=system, user=user, max_tokens=PROPOSE_MAX_TOKENS)
    action, reasoning, new_md = _parse_response(resp.text)

    return ProposeResult(
        skill_name=skill_name,
        action=action,
        new_skill_md=new_md,
        reasoning=reasoning,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Response parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[ProposeAction, str, str | None]:
    """Pull <action>, <reasoning>, <new_skill_md> from the LLM text.

    Returns (action, reasoning, new_skill_md). On malformed output,
    falls back to action='skip' with the raw response in reasoning so
    the human reviewer still gets visibility.
    """
    raw_action = (extract_tag(raw, "action") or "").lower().strip()
    reasoning = extract_tag(raw, "reasoning") or ""
    new_md = extract_tag(raw, "new_skill_md")

    action: ProposeAction
    if raw_action == "edit":
        action = "edit"
    elif raw_action == "skip":
        action = "skip"
    else:
        # Sometimes the LLM forgets the action tag and dumps SKILL.md.
        # If we got a long markdown body but no action tag, infer 'edit'.
        if new_md and len(new_md) > 200:
            action = "edit"
        else:
            action = "skip"
            if not reasoning:
                reasoning = (
                    "Parser could not extract <action>/<reasoning> tags; "
                    f"raw response truncated to 500 chars: {raw[:500]}"
                )

    if action == "skip":
        new_md = None

    return action, reasoning, new_md
