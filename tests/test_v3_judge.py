"""Tests for v3's judge — winner + rubric + binary-check parser."""

from __future__ import annotations

import pytest

from agent_autoresearch.strategies.v3.judge import (
    CheckOutcome,
    JudgeResult,
    RubricScore,
    _parse_response,
    _parse_score_clamped,
)
from agent_autoresearch.strategies.v3.program import BinaryCheck, RubricAxis


_AXES = [
    RubricAxis(name="clarity", description="..."),
    RubricAxis(name="safety", description="..."),
    RubricAxis(name="completeness", description="..."),
]
_CHECKS = [BinaryCheck(id=i, text=f"check {i}") for i in range(1, 6)]


_HAPPY_RESPONSE = """
<winner>new</winner>
<rubric>
  <axis><name>clarity</name><new>3</new><old>1</old></axis>
  <axis><name>safety</name><new>3</new><old>2</old></axis>
  <axis><name>completeness</name><new>2</new><old>2</old></axis>
</rubric>
<checks>
  <check><id>1</id><result>pass</result></check>
  <check><id>2</id><result>pass</result></check>
  <check><id>3</id><result>na</result></check>
  <check><id>4</id><result>pass</result></check>
  <check><id>5</id><result>pass</result></check>
</checks>
<reasoning>New addresses dietary constraints; old ignored them.</reasoning>
"""


# ── Score clamping ──────────────────────────────────────────────────────────

class TestParseScoreClamped:
    @pytest.mark.parametrize("raw,expected", [("1", 1), ("2", 2), ("3", 3),
                                                 (" 3 ", 3)])
    def test_valid_in_range(self, raw, expected):
        assert _parse_score_clamped(raw) == expected

    @pytest.mark.parametrize("raw", ["0", "4", "10", "-1", "abc", "", None])
    def test_out_of_range_clamps_to_2(self, raw):
        assert _parse_score_clamped(raw) == 2


# ── Happy-path parsing ──────────────────────────────────────────────────────

class TestParseResponseHappy:
    def test_winner_extracted(self):
        winner, rubric, checks, reasoning = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert winner == "new"

    def test_rubric_scores_aligned_with_expected_axes(self):
        _, rubric, _, _ = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert len(rubric) == 3
        by_name = {s.name: s for s in rubric}
        assert by_name["clarity"].new == 3 and by_name["clarity"].old == 1
        assert by_name["safety"].new == 3 and by_name["safety"].old == 2
        assert by_name["completeness"].new == 2 and by_name["completeness"].old == 2

    def test_check_results_aligned_with_expected_ids(self):
        _, _, checks, _ = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert len(checks) == 5
        assert checks[2].result == "na"
        assert all(c.is_pass for c in checks)   # na counts as pass

    def test_reasoning_extracted(self):
        _, _, _, reasoning = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert "dietary" in reasoning.lower()


# ── Defensive parsing ───────────────────────────────────────────────────────

class TestDefensiveParsing:
    def test_missing_winner_defaults_to_old(self):
        winner, _, _, _ = _parse_response(
            "garbage with no tags",
            expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert winner == "old"

    def test_missing_axes_default_to_2_2(self):
        """When the LLM forgets some axes entirely, they default to 2/2."""
        partial = """
        <winner>new</winner>
        <rubric>
          <axis><name>clarity</name><new>3</new><old>1</old></axis>
        </rubric>
        <checks></checks>
        <reasoning>only clarity assessed</reasoning>
        """
        _, rubric, _, _ = _parse_response(
            partial, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        by_name = {s.name: s for s in rubric}
        assert by_name["clarity"].new == 3 and by_name["clarity"].old == 1
        # safety and completeness defaulted to 2/2
        assert by_name["safety"].new == 2 and by_name["safety"].old == 2
        assert by_name["completeness"].new == 2

    def test_missing_checks_default_to_fail(self):
        """Missing checks default to fail — regression-safe (don't silently pass)."""
        partial = """
        <winner>tie</winner>
        <rubric></rubric>
        <checks>
          <check><id>1</id><result>pass</result></check>
        </checks>
        <reasoning>only check 1 evaluated</reasoning>
        """
        _, _, checks, _ = _parse_response(
            partial, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert checks[0].result == "pass"
        # checks 2-5 defaulted to 'fail'
        assert all(c.result == "fail" for c in checks[1:])

    def test_out_of_range_score_normalized(self):
        partial = """
        <winner>new</winner>
        <rubric>
          <axis><name>clarity</name><new>5</new><old>0</old></axis>
        </rubric>
        <checks></checks>
        <reasoning>oh well</reasoning>
        """
        _, rubric, _, _ = _parse_response(
            partial, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        by_name = {s.name: s for s in rubric}
        assert by_name["clarity"].new == 2   # clamped from 5
        assert by_name["clarity"].old == 2   # clamped from 0


# ── JudgeResult derived properties ──────────────────────────────────────────

class TestJudgeResultDerived:
    def _build(self, rubric, checks, winner="new") -> JudgeResult:
        return JudgeResult(
            session_id="s", focus_turn=1, winner=winner,
            rubric_scores=rubric, check_results=checks,
            reasoning="", raw_response="",
            input_tokens=None, output_tokens=None,
        )

    def test_rubric_improved_when_avg_new_greater(self):
        r = self._build(
            rubric=[RubricScore("a", 3, 1), RubricScore("b", 2, 2)],
            checks=[],
        )
        # avg new = 2.5, avg old = 1.5
        assert r.rubric_improved is True
        assert r.rubric_non_regressed is True

    def test_rubric_non_regressed_includes_equal(self):
        r = self._build(
            rubric=[RubricScore("a", 2, 2), RubricScore("b", 2, 2)],
            checks=[],
        )
        assert r.rubric_improved is False         # not strictly better
        assert r.rubric_non_regressed is True     # not worse

    def test_rubric_regression_detected(self):
        r = self._build(
            rubric=[RubricScore("a", 1, 3)],
            checks=[],
        )
        assert r.rubric_improved is False
        assert r.rubric_non_regressed is False

    def test_all_checks_pass_handles_na(self):
        r = self._build(
            rubric=[],
            checks=[
                CheckOutcome(1, "pass"),
                CheckOutcome(2, "na"),
                CheckOutcome(3, "pass"),
            ],
        )
        assert r.all_checks_pass is True

    def test_one_failing_check_fails_all(self):
        r = self._build(
            rubric=[],
            checks=[CheckOutcome(1, "pass"), CheckOutcome(2, "fail")],
        )
        assert r.all_checks_pass is False

    def test_empty_checks_passes_trivially(self):
        r = self._build(rubric=[], checks=[])
        assert r.all_checks_pass is True
