"""Tests for v3's program parser — rubric + binary checks extraction."""

from __future__ import annotations

import pytest

from agent_autoresearch.strategies.v3.program import (
    MIN_BINARY_CHECKS,
    RUBRIC_AXIS_COUNT,
    BinaryCheck,
    ProgramResult,
    RubricAxis,
    _is_skip_program,
    _parse_binary_checks,
    _parse_rubric_axes,
)


_VALID_PROGRAM = """
# Improvement Strategy — find-restaurant

## Target
Vegan filter ignored across 3 sessions.

## Evidence from logs
- sess_001 turn 2: agent recommended a steakhouse for a vegan request

## Current skill
The skill defines search_restaurants but does not require dietary preferences
to be passed as the filter argument.

## Proposed change
Add an explicit rule that user-stated dietary preferences must be passed as
the filter argument to search_restaurants.

## What NOT to change
- Do not add new tools
- Do not change the agent's persona
- Do not modify the search query format

## Rubric — 3 axes, scored 1–3

- **dietary_constraint_handling**: Excellent means agent always passes user-stated dietary preferences as the filter argument.
- **query_specificity**: Excellent means search query includes all named constraints, not just location.
- **result_grounding**: Excellent means recommended restaurant exists in the tool output and respects all stated filters.

## Binary checks — invariants the new prompt must preserve

- [ ] Does the agent always pass user-stated dietary preferences as the filter argument?
- [ ] Does the agent never recommend a result contradicting a stated dietary preference?
- [ ] Does the agent never refuse a reasonable lookup with 'check Yelp'?
- [ ] Does the agent always call search_restaurants for restaurant lookups?
- [ ] Does the agent never recommend a restaurant not in the tool output?
"""


_SKIP_PROGRAM = """
# Improvement Strategy — find-restaurant

## Recommendation: SKIP

The 3 evidence items hit different and unrelated patterns with no shared
root cause. Wait for more evidence before editing.
"""


# ── _is_skip_program ────────────────────────────────────────────────────────

class TestIsSkip:
    def test_recognizes_skip(self):
        assert _is_skip_program(_SKIP_PROGRAM) is True

    def test_full_program_is_not_skip(self):
        assert _is_skip_program(_VALID_PROGRAM) is False

    def test_case_insensitive(self):
        assert _is_skip_program("## recommendation: skip\n") is True
        assert _is_skip_program("## RECOMMENDATION: SKIP") is True


# ── Rubric parser ───────────────────────────────────────────────────────────

class TestParseRubric:
    def test_extracts_three_axes(self):
        axes = _parse_rubric_axes(_VALID_PROGRAM)
        assert len(axes) == 3
        names = [a.name for a in axes]
        assert names == [
            "dietary_constraint_handling",
            "query_specificity",
            "result_grounding",
        ]

    def test_descriptions_captured(self):
        axes = _parse_rubric_axes(_VALID_PROGRAM)
        assert "dietary preferences" in axes[0].description
        assert "constraints" in axes[1].description

    def test_empty_when_no_rubric_section(self):
        md = "# Strategy\n\n## Target\n\nFoo"
        assert _parse_rubric_axes(md) == []

    def test_skip_programs_have_no_rubric(self):
        assert _parse_rubric_axes(_SKIP_PROGRAM) == []

    def test_alternative_bullet_styles_accepted(self):
        md = """
## Rubric

* **axis_a**: description a.
* **axis_b**: description b.
* **axis_c**: description c.
"""
        axes = _parse_rubric_axes(md)
        assert len(axes) == 3


# ── Binary checks parser ────────────────────────────────────────────────────

class TestParseBinaryChecks:
    def test_extracts_five_checks(self):
        checks = _parse_binary_checks(_VALID_PROGRAM)
        assert len(checks) == 5

    def test_check_ids_are_sequential(self):
        checks = _parse_binary_checks(_VALID_PROGRAM)
        assert [c.id for c in checks] == [1, 2, 3, 4, 5]

    def test_check_text_captured(self):
        checks = _parse_binary_checks(_VALID_PROGRAM)
        assert "dietary preferences" in checks[0].text
        assert "yelp" in checks[2].text.lower()

    def test_accepts_marked_boxes(self):
        md = """
## Binary checks

- [x] First check.
- [X] Second check.
- [ ] Third check.
"""
        checks = _parse_binary_checks(md)
        assert len(checks) == 3

    def test_empty_when_no_section(self):
        assert _parse_binary_checks("# strategy\n\n## Target\n\nFoo") == []


# ── ProgramResult invariants ────────────────────────────────────────────────

class TestProgramResultInvariants:
    def test_has_validation_schema_when_complete(self):
        r = ProgramResult(
            skill_name="x", program_md=_VALID_PROGRAM,
            rubric_axes=[RubricAxis(f"a{i}", "d") for i in range(RUBRIC_AXIS_COUNT)],
            binary_checks=[BinaryCheck(i, "c") for i in range(1, MIN_BINARY_CHECKS + 1)],
        )
        assert r.has_validation_schema is True

    def test_too_few_rubric_axes_fails_invariant(self):
        r = ProgramResult(
            skill_name="x", program_md="",
            rubric_axes=[RubricAxis("a", "d")],   # only 1
            binary_checks=[BinaryCheck(i, "c") for i in range(1, MIN_BINARY_CHECKS + 1)],
        )
        assert r.has_validation_schema is False

    def test_too_few_checks_fails_invariant(self):
        r = ProgramResult(
            skill_name="x", program_md="",
            rubric_axes=[RubricAxis(f"a{i}", "d") for i in range(RUBRIC_AXIS_COUNT)],
            binary_checks=[BinaryCheck(1, "c"), BinaryCheck(2, "c")],   # only 2
        )
        assert r.has_validation_schema is False

    def test_skip_result_never_has_schema(self):
        r = ProgramResult(skill_name="x", program_md=_SKIP_PROGRAM, is_skip=True)
        assert r.has_validation_schema is False
