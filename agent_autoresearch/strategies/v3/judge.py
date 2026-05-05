"""Step 7b (v3) — judge with three signals: winner + rubric + binary checks.

In one LLM call the v3 judge produces:
  1. `winner` — same `new`/`old`/`tie` as v1/v2 (used for fix_rate
     and regression_rate).
  2. `rubric_scores` — for each axis from program.md, an independent
     1–3 score for OLD and NEW (used for rubric_improvement_rate).
  3. `check_results` — pass/fail/na for each binary check from
     program.md (used for binary_checks_pass_rate).

Cost: longer prompt + longer response than v1/v2 (~1.3× per call),
but same call count per session — no extra LLM round-trips.

Defaults on parse failure are conservative:
  - `winner` → `old`
  - rubric scores → both 2 ("adequate") on missing axes
  - checks → `fail` on missing checks (regression-safe default)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v3._common import extract_tag
from agent_autoresearch.strategies.v3.program import BinaryCheck, RubricAxis

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


# Higher cap than v1/v2 — judge response now carries 3 signals
JUDGE_MAX_TOKENS = 1200

JudgeWinner = Literal["new", "old", "tie"]
CheckResult = Literal["pass", "fail", "na"]


# ── Per-signal data ─────────────────────────────────────────────────────────

@dataclass
class RubricScore:
    """One axis's old + new score (1–3 each)."""
    name: str
    new: int
    old: int

    @property
    def improved(self) -> bool:
        return self.new > self.old

    @property
    def regressed(self) -> bool:
        return self.new < self.old


@dataclass
class CheckOutcome:
    """One binary check's pass/fail/na verdict."""
    id: int
    result: CheckResult

    @property
    def is_pass(self) -> bool:
        # `na` is treated as pass — the invariant didn't apply at the
        # focus turn, so the new prompt didn't break it.
        return self.result in ("pass", "na")


@dataclass
class JudgeResult:
    """v3 judge output — three signals from one call."""
    session_id: str
    focus_turn: int
    winner: JudgeWinner
    rubric_scores: list[RubricScore]
    check_results: list[CheckOutcome]
    reasoning: str
    raw_response: str
    input_tokens: int | None
    output_tokens: int | None

    # ── per-session aggregations ────────────────────────────────────────────

    @property
    def avg_new_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return sum(s.new for s in self.rubric_scores) / len(self.rubric_scores)

    @property
    def avg_old_score(self) -> float:
        if not self.rubric_scores:
            return 0.0
        return sum(s.old for s in self.rubric_scores) / len(self.rubric_scores)

    @property
    def rubric_improved(self) -> bool:
        """Used for fix-session contribution to rubric_improvement_rate."""
        return self.avg_new_score > self.avg_old_score

    @property
    def rubric_non_regressed(self) -> bool:
        """Used for baseline-session contribution to rubric_improvement_rate."""
        return self.avg_new_score >= self.avg_old_score

    @property
    def all_checks_pass(self) -> bool:
        """Used for the binary_checks_pass_rate aggregation.

        Empty `check_results` returns True — no checks to fail.
        """
        return all(c.is_pass for c in self.check_results)


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
    rubric_axes: list[RubricAxis],
    binary_checks: list[BinaryCheck],
    llm: LLMProvider | None = None,
) -> JudgeResult:
    """Run one judge call. Returns winner + rubric scores + check outcomes."""
    llm = llm or default_llm_provider()

    system, user = format_prompt(
        _PROMPT_PATH,
        session_id=session_id,
        transcript=transcript,
        focus_turn=focus_turn,
        user_message=user_message,
        old_reply=old_reply,
        new_reply=new_reply,
        new_tool_plan=new_tool_plan,
        program_md=program_md,
        rubric_block=_format_rubric_block(rubric_axes),
        checks_block=_format_checks_block(binary_checks),
    )

    resp = llm.call(system=system, user=user, max_tokens=JUDGE_MAX_TOKENS)
    winner, rubric, checks, reasoning = _parse_response(
        resp.text,
        expected_axes=rubric_axes,
        expected_checks=binary_checks,
    )

    return JudgeResult(
        session_id=session_id,
        focus_turn=focus_turn,
        winner=winner,
        rubric_scores=rubric,
        check_results=checks,
        reasoning=reasoning,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Prompt block helpers ────────────────────────────────────────────────────

def _format_rubric_block(axes: list[RubricAxis]) -> str:
    if not axes:
        return "_(no rubric — judge will skip rubric scoring)_"
    return "\n".join(
        f"- **{a.name}**: {a.description}" for a in axes
    )


def _format_checks_block(checks: list[BinaryCheck]) -> str:
    if not checks:
        return "_(no binary checks)_"
    return "\n".join(f"- check `{c.id}`: {c.text}" for c in checks)


# ── Response parser ─────────────────────────────────────────────────────────

_AXIS_BLOCK_RE = re.compile(
    r"<axis\s*>(.*?)</axis\s*>", re.DOTALL | re.IGNORECASE,
)
_CHECK_BLOCK_RE = re.compile(
    r"<check\s*>(.*?)</check\s*>", re.DOTALL | re.IGNORECASE,
)


def _parse_response(
    raw: str,
    *,
    expected_axes: list[RubricAxis],
    expected_checks: list[BinaryCheck],
) -> tuple[JudgeWinner, list[RubricScore], list[CheckOutcome], str]:
    """Pull all three signals + reasoning. Defaults are conservative.

    Missing axes → score 2/2 (no improvement either way).
    Missing checks → result `fail` (regression-safe default).
    Missing winner → `old`.
    """
    # Winner
    winner_raw = (extract_tag(raw, "winner") or "").strip().lower()
    if winner_raw in ("new", "old", "tie"):
        winner: JudgeWinner = winner_raw   # type: ignore[assignment]
    else:
        winner = "old"

    reasoning = extract_tag(raw, "reasoning") or ""

    # Rubric — pull all <axis>...</axis> blocks, then align with expected axes
    rubric_section = extract_tag(raw, "rubric") or ""
    parsed_axes: dict[str, tuple[int, int]] = {}
    for block in _AXIS_BLOCK_RE.findall(rubric_section):
        name = (extract_tag(block, "name") or "").strip()
        if not name:
            continue
        new_s = _parse_score_clamped(extract_tag(block, "new"))
        old_s = _parse_score_clamped(extract_tag(block, "old"))
        parsed_axes[name.lower()] = (new_s, old_s)

    rubric: list[RubricScore] = []
    for ax in expected_axes:
        n, o = parsed_axes.get(ax.name.lower(), (2, 2))
        rubric.append(RubricScore(name=ax.name, new=n, old=o))

    # Checks — pull all <check>...</check> blocks, align with expected checks by id
    checks_section = extract_tag(raw, "checks") or ""
    parsed_checks: dict[int, CheckResult] = {}
    for block in _CHECK_BLOCK_RE.findall(checks_section):
        id_raw = (extract_tag(block, "id") or "").strip()
        result_raw = (extract_tag(block, "result") or "").strip().lower()
        try:
            cid = int(id_raw)
        except ValueError:
            continue
        if result_raw in ("pass", "fail", "na"):
            parsed_checks[cid] = result_raw   # type: ignore[assignment]

    check_outcomes: list[CheckOutcome] = []
    for ch in expected_checks:
        # Default to fail when the judge didn't return this check —
        # safer than silently passing.
        check_outcomes.append(CheckOutcome(
            id=ch.id,
            result=parsed_checks.get(ch.id, "fail"),
        ))

    if not reasoning and (not rubric or not check_outcomes):
        reasoning = (
            "Parser could only partially extract judge tags; conservative "
            f"defaults applied. Raw response (500 chars): {raw[:500]}"
        )

    return winner, rubric, check_outcomes, reasoning


def _parse_score_clamped(raw: str | None) -> int:
    """Parse a 1–3 score; clamp out-of-range or invalid to 2."""
    if not raw:
        return 2
    try:
        v = int(raw.strip())
    except ValueError:
        return 2
    if v < 1 or v > 3:
        return 2
    return v
