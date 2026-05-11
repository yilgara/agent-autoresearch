"""Step 6 — critic. Independent audit of the proposer's diff (Validation Layer A).

Receives the strategy doc, the focused diff, AND both full files
so it can verify "what NOT to change" sections are byte-for-byte
intact and that protected structure (frontmatter, headings, code
fences, terminology) is preserved across the whole file — not just
around the change.

Returns a `CriticResult` with a verdict (APPROVE / REQUEST_CHANGES),
prose reasoning, and a list of specific concerns each anchored to a
diff line.

This is strategy v1's implementation. See `prompts/critic.md` next
to this file for the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v1._common import extract_tag

_PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.md"


# Token cap — short verdict + concerns list
CRITIC_MAX_TOKENS = 1500

CriticVerdict = Literal["APPROVE", "REQUEST_CHANGES"]


@dataclass
class CriticResult:
    """Output of critic() — Validation Layer A audit."""
    skill_name: str
    verdict: CriticVerdict
    reasoning: str
    concerns: list[str]              # parsed bullet points (each anchored to a diff line)
    raw_response: str
    input_tokens: int | None
    output_tokens: int | None

    @property
    def approves(self) -> bool:
        return self.verdict == "APPROVE"

    def to_markdown(self) -> str:
        """Render as `critic.md` — what gets written next to `diff.txt`."""
        lines = [
            "# Critic Review",
            "",
            f"**Verdict:** `{self.verdict}`",
            "",
            "## Reasoning",
            "",
            self.reasoning or "_(no reasoning provided)_",
            "",
            "## Concerns",
            "",
        ]
        if self.concerns:
            lines.extend(f"- {c}" for c in self.concerns)
        else:
            lines.append("_(none)_")
        return "\n".join(lines)


# ── Stage entry point ───────────────────────────────────────────────────────

def critic(
    skill_name: str,
    *,
    program_md: str,
    diff_text: str,
    v_old_md: str,
    v_new_md: str,
    llm: LLMProvider | None = None,
) -> CriticResult:
    """Step 6 — audit the proposer's diff against the editing rules.

    All four inputs are passed to the prompt:
      - `program_md` — what the proposer was told to do
      - `diff_text` — the focused view of what changed
      - `v_old_md` — full original SKILL.md (so critic can verify
        "what NOT to change" sections are byte-for-byte intact)
      - `v_new_md` — full proposed SKILL.md (so critic can compare
        in context — frontmatter, headings, terminology preserved)

    Returns a `CriticResult` with verdict + structured concerns.
    """
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        _PROMPT_PATH,
        program_md=program_md,
        diff_text=diff_text,
        v_old_md=v_old_md,
        v_new_md=v_new_md,
    )

    resp = llm.call(system=system, user=user, max_tokens=CRITIC_MAX_TOKENS)
    verdict, reasoning, concerns = _parse_response(resp.text)

    return CriticResult(
        skill_name=skill_name,
        verdict=verdict,
        reasoning=reasoning,
        concerns=concerns,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Response parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> tuple[CriticVerdict, str, list[str]]:
    """Pull <verdict>, <reasoning>, <concerns> from the LLM text.

    Defaults to REQUEST_CHANGES on ambiguity — burden of proof is on
    the new skill, so we err toward human review when in doubt.
    """
    verdict_raw = (extract_tag(raw, "verdict") or "").upper().strip()
    reasoning = extract_tag(raw, "reasoning") or ""
    concerns_raw = extract_tag(raw, "concerns") or ""

    verdict: CriticVerdict
    if "APPROV" in verdict_raw:                    # APPROVE / APPROVED / etc.
        verdict = "APPROVE"
    elif verdict_raw:
        verdict = "REQUEST_CHANGES"
    else:
        verdict = "REQUEST_CHANGES"
        if not reasoning:
            reasoning = (
                "Parser could not extract <verdict> tag; defaulting to "
                f"REQUEST_CHANGES. Raw response (500 chars): {raw[:500]}"
            )

    # Parse concerns: one bullet per line starting with "- "
    concerns: list[str] = []
    for line in concerns_raw.splitlines():
        line = line.strip()
        if line.startswith("- "):
            text = line[2:].strip()
            if text and text.lower() not in ("(none)", "none"):
                concerns.append(text)

    return verdict, reasoning, concerns
