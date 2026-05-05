"""Step 4 (v3) — build_program with structured rubric + binary checks.

Same job as v1/v2: read the target's evidence + current SKILL.md, call
the program LLM, return a strategy doc.

v3 additions: the LLM produces structured `## Rubric` (3 graded axes)
and `## Binary checks` (≥ 5 invariants) sections. We parse those into
typed objects (`RubricAxis`, `BinaryCheck`) and forward them to the
judge stage. The judge uses them to score each replay session
per-axis and per-check, and verdict turns the aggregates into the
v3 acceptance gates (`fix_rate`, `regression_rate`,
`rubric_improvement_rate`, `binary_checks_pass_rate`).

The full program markdown is preserved in `ProgramResult.program_md`
exactly as v1/v2 — only the additional structured fields are new.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.data import Evidence, Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v3._common import strip_chatter

_PROMPT_PATH = Path(__file__).parent / "prompts" / "program.md"


# Token cap — larger than v1/v2 because v3 emits the rubric + check sections
PROGRAM_MAX_TOKENS = 3000

# Evidence trimming — same policy as v1/v2
EVIDENCE_PER_CATEGORY = 8
EVIDENCE_MAX_TOTAL    = 8

# Schema constants — match the prompt template exactly
RUBRIC_AXIS_COUNT     = 3
MIN_BINARY_CHECKS     = 5


# ── Structured outputs ──────────────────────────────────────────────────────

@dataclass
class RubricAxis:
    """One graded axis the judge will score per session.

    `name` is a snake_case identifier used as a key in judge output
    aggregation. `description` is one sentence of what "excellent"
    (score 3) looks like for THIS skill — the LLM judge gets it
    verbatim, so keep it specific.
    """
    name: str
    description: str


@dataclass
class BinaryCheck:
    """One yes/no invariant the judge will evaluate per session.

    `id` is just an integer for cross-referencing in judge output;
    the judge sees `text` (a single yes/no question).
    """
    id: int
    text: str


@dataclass
class ProgramResult:
    """v3 program output — strategy text + structured rubric/checks.

    `program_md` keeps the full markdown for v1/v2 backward
    compatibility (it's still written to `program.md` on disk, still
    fed verbatim into the propose stage). The new fields are what
    v3 judge / verdict consume.

    For SKIP outputs, `is_skip` is True and the rubric/check lists are
    empty — there's nothing to validate downstream.
    """
    skill_name: str
    program_md: str
    rubric_axes: list[RubricAxis] = field(default_factory=list)
    binary_checks: list[BinaryCheck] = field(default_factory=list)
    is_skip: bool = False
    raw_response: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def has_validation_schema(self) -> bool:
        """True iff we have enough rubric+checks to run v3 replay."""
        return (
            not self.is_skip
            and len(self.rubric_axes) == RUBRIC_AXIS_COUNT
            and len(self.binary_checks) >= MIN_BINARY_CHECKS
        )


# ── Evidence formatter — unchanged from v1/v2 ────────────────────────────────

def format_evidence_block(
    evidence: list[Evidence],
    *,
    per_category: int = EVIDENCE_PER_CATEGORY,
    max_total: int = EVIDENCE_MAX_TOTAL,
) -> str:
    """Render a representative sample of `Evidence` items for the prompt.

    Trim strategy: keep up to `per_category` items per `category`,
    capped at `max_total` overall. Preserves diversity of failure modes.
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
    """Step 4 (v3) — generate the strategy doc + rubric + binary checks.

    One LLM call. Output parsed into a `ProgramResult` with both the
    raw markdown (for the propose stage) and structured rubric/checks
    (for the judge stage).
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

    is_skip = _is_skip_program(program_md)
    rubric_axes: list[RubricAxis] = []
    binary_checks: list[BinaryCheck] = []
    if not is_skip:
        rubric_axes = _parse_rubric_axes(program_md)
        binary_checks = _parse_binary_checks(program_md)

    return ProgramResult(
        skill_name=target.skill_name,
        program_md=program_md,
        rubric_axes=rubric_axes,
        binary_checks=binary_checks,
        is_skip=is_skip,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Parsers ─────────────────────────────────────────────────────────────────

def _is_skip_program(md: str) -> bool:
    """Detect the SKIP variant — `## Recommendation: SKIP` heading."""
    return bool(re.search(r"^##\s+Recommendation\s*:\s*SKIP", md, re.MULTILINE | re.IGNORECASE))


def _extract_section(md: str, heading: str) -> str:
    """Return the body of a `## <heading>` section (until the next ## or EOF).

    Heading match is case-insensitive and tolerant of trailing punctuation.
    Returns empty string if not found.
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}.*?\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(md)
    return m.group(1).strip() if m else ""


# Rubric bullet shape: "- **axis_name**: description"
_RUBRIC_BULLET_RE = re.compile(
    r"^\s*[-*]\s*\*\*([A-Za-z][A-Za-z0-9_]*)\*\*\s*:\s*(.+?)\s*$",
    re.MULTILINE,
)


def _parse_rubric_axes(md: str) -> list[RubricAxis]:
    """Pull the 3 axes out of the `## Rubric` section.

    Tolerant: accepts more or fewer than 3, lets the consumer
    (`ProgramResult.has_validation_schema`) decide whether the count
    is acceptable. Empty list on no Rubric section.
    """
    body = _extract_section(md, "Rubric")
    if not body:
        return []
    out: list[RubricAxis] = []
    for m in _RUBRIC_BULLET_RE.finditer(body):
        name = m.group(1).strip()
        desc = m.group(2).strip()
        if name and desc:
            out.append(RubricAxis(name=name, description=desc))
    return out


# Binary-check bullet shape: "- [ ] question text" or "- [x] question text"
_CHECK_BULLET_RE = re.compile(
    r"^\s*[-*]\s*\[[ xX]\]\s*(.+?)\s*$",
    re.MULTILINE,
)


def _parse_binary_checks(md: str) -> list[BinaryCheck]:
    """Pull binary-check bullets out of the `## Binary checks` section.

    Strips any trailing question-mark normalization so check text is
    consistent across variations.
    """
    body = _extract_section(md, "Binary checks")
    if not body:
        return []
    out: list[BinaryCheck] = []
    for i, m in enumerate(_CHECK_BULLET_RE.finditer(body), start=1):
        text = m.group(1).strip().strip('"').strip("'")
        if text:
            out.append(BinaryCheck(id=i, text=text))
    return out
