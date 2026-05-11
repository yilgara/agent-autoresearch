"""Step 7b (v3) — judge with three signals: new_passes + per-axis rubric + binary checks.

In one LLM call the v3 judge produces:
  1. `new_passes` — bool. Does the new reply (under the proposed skill)
     adequately handle this session on its own merit? No comparison
     against old — the old reply already failed (fix sessions) or
     succeeded (baselines); we just want to know whether new clears
     the bar for this session.
  2. `rubric_votes` — for each axis from program.md, judge picks a
     winner: `new`, `old`, or `tie`. Aggregated downstream over fix
     sessions into a per-axis +1/0/-1 score (see replay.py).
  3. `check_results` — pass/fail/na for each binary check from
     program.md. Aggregated downstream over baseline sessions.

Defaults on parse failure are conservative:
  - `new_passes` → False (don't optimistically count a parse miss as a pass)
  - rubric votes → `tie` (neutral, 0 score)
  - checks → `fail` (regression-safe default)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.strategies.v3._common import extract_tag
from agent_autoresearch.strategies.v3.program import BinaryCheck, RubricAxis

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


# Token cap — judge response carries 3 signals; shorter than the old
# 1-3 rubric since each axis is now a single winner tag.
JUDGE_MAX_TOKENS = 1000

RubricWinner = Literal["new", "old", "tie"]
CheckResult = Literal["pass", "fail", "na"]


# ── Per-signal data ─────────────────────────────────────────────────────────

@dataclass
class RubricVote:
    """One axis's verdict for this session.

    `score` maps the winner to +1/0/-1 so aggregation across
    (session, axis) pairs reduces to a simple mean.
    """
    name: str
    winner: RubricWinner

    @property
    def score(self) -> int:
        return {"new": 1, "tie": 0, "old": -1}[self.winner]


@dataclass
class CheckOutcome:
    """One binary check's pass/fail/na verdict on the new reply."""
    id: int
    result: CheckResult

    @property
    def is_pass(self) -> bool:
        # `na` counts as pass — the invariant didn't apply at the focus
        # turn, so the new prompt didn't violate it.
        return self.result in ("pass", "na")


@dataclass
class JudgeResult:
    """v3 judge output — three signals from one call."""
    session_id: str
    focus_turn: int
    new_passes: bool
    rubric_votes: list[RubricVote]
    check_results: list[CheckOutcome]
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
    rubric_axes: list[RubricAxis],
    binary_checks: list[BinaryCheck],
    llm: LLMProvider | None = None,
) -> JudgeResult:
    """Run one judge call. Returns new_passes + per-axis rubric votes +
    per-check outcomes."""
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
    new_passes, rubric, checks, reasoning = _parse_response(
        resp.text,
        expected_axes=rubric_axes,
        expected_checks=binary_checks,
    )

    return JudgeResult(
        session_id=session_id,
        focus_turn=focus_turn,
        new_passes=new_passes,
        rubric_votes=rubric,
        check_results=checks,
        reasoning=reasoning,
        raw_response=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


# ── Prompt block helpers ────────────────────────────────────────────────────

def _format_rubric_block(axes: list[RubricAxis]) -> str:
    if not axes:
        return "_(no rubric — judge will skip rubric voting)_"
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
) -> tuple[bool, list[RubricVote], list[CheckOutcome], str]:
    """Pull all three signals + reasoning. Defaults are conservative.

    Missing new_passes → False.
    Missing axes → vote `tie` (score 0).
    Missing checks → result `fail`.
    """
    new_passes = _parse_bool(extract_tag(raw, "new_passes"))
    reasoning = extract_tag(raw, "reasoning") or ""

    # Rubric — pull all <axis> blocks, align with expected axes by name
    rubric_section = extract_tag(raw, "rubric") or ""
    parsed_axes: dict[str, RubricWinner] = {}
    for block in _AXIS_BLOCK_RE.findall(rubric_section):
        name = (extract_tag(block, "name") or "").strip().lower()
        winner_raw = (extract_tag(block, "winner") or "").strip().lower()
        if name and winner_raw in ("new", "old", "tie"):
            parsed_axes[name] = winner_raw   # type: ignore[assignment]

    rubric: list[RubricVote] = []
    for ax in expected_axes:
        winner = parsed_axes.get(ax.name.lower(), "tie")
        rubric.append(RubricVote(name=ax.name, winner=winner))

    # Checks — pull all <check> blocks, align with expected checks by id
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
        check_outcomes.append(CheckOutcome(
            id=ch.id,
            result=parsed_checks.get(ch.id, "fail"),
        ))

    if not reasoning and not parsed_axes and not parsed_checks:
        reasoning = (
            "Parser could only partially extract judge tags; conservative "
            f"defaults applied. Raw response (500 chars): {raw[:500]}"
        )

    return new_passes, rubric, check_outcomes, reasoning


def _parse_bool(raw: str | None) -> bool:
    if not raw:
        return False
    return raw.strip().lower() in ("true", "yes", "1", "pass", "passes")
