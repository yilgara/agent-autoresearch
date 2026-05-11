"""Step 4 — build_program. Generate the per-target strategy doc.

Reads the target's evidence + the current SKILL.md, calls the
`program` prompt, returns a `ProgramResult` carrying the strategy
text. The orchestrator writes it to `outputs/<run>/<skill>/program.md`
and feeds it to step 5 (`propose`).

This is strategy v1's implementation. See `prompts/program.md` next
to this file for the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.data import Evidence, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v2._common import strip_chatter

_PROMPT_PATH = Path(__file__).parent / "prompts" / "program.md"


# Token cap — program.md is short (1-2 KB output)
PROGRAM_MAX_TOKENS = 2000

# Evidence trimming — keep diversity of failure modes by capping per
# `Evidence.category` rather than blindly truncating to the first N.
EVIDENCE_PER_CATEGORY = 8
EVIDENCE_MAX_TOTAL    = 8


@dataclass
class ProgramResult:
    """Output of build_program() — the strategy doc + token usage."""
    skill_name: str
    program_md: str                  # the parsed strategy doc
    raw_response: str                # full LLM text in case of parsing issues
    input_tokens: int | None
    output_tokens: int | None


# ── Evidence formatter ──────────────────────────────────────────────────────

def format_evidence_block(
    evidence: list[Evidence],
    *,
    per_category: int = EVIDENCE_PER_CATEGORY,
    max_total: int = EVIDENCE_MAX_TOTAL,
) -> str:
    """Render a representative sample of `Evidence` items for the
    strategy prompt's `{evidence_block}` placeholder.

    Trim strategy: keep up to `per_category` items per `category`,
    capped at `max_total` overall. This preserves DIVERSITY of failure
    modes — a skill with 20 items of `wrong_information` + 2 of
    `missing_step` becomes 8 + 2 = 10 examples covering both patterns,
    rather than 10 of a single pattern.

    A footer line announces the trim so the strategy LLM still knows
    the real total counts.
    """
    if not evidence:
        return "_(no evidence — unusual; flag this in your output)_"

    seen: dict[str, int] = {}
    selected: list[Evidence] = []
    for item in evidence:
        if seen.get(item.category, 0) >= per_category:
            continue
        seen[item.category] = seen.get(item.category, 0) + 1
        selected.append(item)
        if len(selected) >= max_total:
            break

    blocks: list[str] = []
    for i, item in enumerate(selected, start=1):
        head = f"### Evidence {i} · `{item.category}`"
        if item.confidence is not None:
            head += f"  ·  confidence={item.confidence:.2f}"
        body = [head]
        for k, v in (item.details or {}).items():
            body.append(f"  - **{k}:** {v}")
        blocks.append("\n".join(body))

    omitted = len(evidence) - len(selected)
    if omitted > 0:
        blocks.append(
            f"_(+ {omitted} additional evidence item(s) with similar patterns, "
            f"omitted to keep the prompt bounded)_"
        )
    return "\n\n".join(blocks)


# ── Stage entry point ───────────────────────────────────────────────────────

def build_program(
    target: Target,
    *,
    current_skill_md: str,
    llm: LLMProvider | None = None,
) -> ProgramResult:
    """Step 4 — generate the per-target program.md via one LLM call.

    The strategy LLM receives:
    - the skill name + rank + count of evidence items
    - the full current SKILL.md
    - the formatted evidence block (trimmed to top-N items per category)
    - replay coverage counts (so it can SKIP if there's no signal to validate)
    """
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        _PROMPT_PATH,
        skill_name=target.skill_name,
        rank=target.rank,
        n_evidence=target.n_evidence,
        current_skill_md=current_skill_md,
        evidence_block=format_evidence_block(target.evidence),
        n_fix_targets=len(target.fix_session_ids),
        n_baselines=len(target.regression_baseline_ids),
    )

    resp = llm.call(system=system, user=user, max_tokens=PROGRAM_MAX_TOKENS)
    program_md = strip_chatter(resp.text)

    return ProgramResult(
        skill_name=target.skill_name,
        program_md=program_md,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
