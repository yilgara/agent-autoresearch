"""Tests for v3's judge — new_passes + rubric votes + binary-check parser."""

from __future__ import annotations

import pytest

from agent_autoresearch.strategies.v3.judge import (
    CheckOutcome,
    JudgeResult,
    RubricVote,
    _parse_bool,
    _parse_response,
)
from agent_autoresearch.strategies.v3.program import BinaryCheck, RubricAxis


_AXES = [
    RubricAxis(name="clarity", description="..."),
    RubricAxis(name="safety", description="..."),
    RubricAxis(name="completeness", description="..."),
]
_CHECKS = [BinaryCheck(id=i, text=f"check {i}") for i in range(1, 6)]


_HAPPY_RESPONSE = """
<new_passes>true</new_passes>
<rubric>
  <axis><name>clarity</name><winner>new</winner></axis>
  <axis><name>safety</name><winner>new</winner></axis>
  <axis><name>completeness</name><winner>tie</winner></axis>
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


# ── new_passes parsing ──────────────────────────────────────────────────────

class TestParseBool:
    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", "yes", "1", "pass", "passes"])
    def test_truthy(self, raw):
        assert _parse_bool(raw) is True

    @pytest.mark.parametrize("raw", ["false", "no", "0", "fail", "", None, "garbage"])
    def test_falsy(self, raw):
        assert _parse_bool(raw) is False


# ── Happy-path parsing ──────────────────────────────────────────────────────

class TestParseResponseHappy:
    def test_new_passes_extracted(self):
        new_passes, rubric, checks, reasoning = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert new_passes is True

    def test_rubric_votes_aligned_with_expected_axes(self):
        _, rubric, _, _ = _parse_response(
            _HAPPY_RESPONSE, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert len(rubric) == 3
        by_name = {v.name: v for v in rubric}
        assert by_name["clarity"].winner == "new"
        assert by_name["clarity"].score == 1
        assert by_name["safety"].winner == "new"
        assert by_name["completeness"].winner == "tie"
        assert by_name["completeness"].score == 0

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
    def test_missing_new_passes_defaults_to_false(self):
        new_passes, _, _, _ = _parse_response(
            "garbage with no tags",
            expected_axes=_AXES, expected_checks=_CHECKS,
        )
        assert new_passes is False

    def test_missing_axes_default_to_tie(self):
        """When the LLM forgets some axes entirely, they default to tie (score 0)."""
        partial = """
        <new_passes>true</new_passes>
        <rubric>
          <axis><name>clarity</name><winner>new</winner></axis>
        </rubric>
        <checks></checks>
        <reasoning>only clarity assessed</reasoning>
        """
        _, rubric, _, _ = _parse_response(
            partial, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        by_name = {v.name: v for v in rubric}
        assert by_name["clarity"].winner == "new"
        assert by_name["safety"].winner == "tie"
        assert by_name["completeness"].winner == "tie"
        assert by_name["safety"].score == 0

    def test_missing_checks_default_to_fail(self):
        """Missing checks default to fail — regression-safe (don't silently pass)."""
        partial = """
        <new_passes>false</new_passes>
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
        assert all(c.result == "fail" for c in checks[1:])

    def test_invalid_winner_defaults_to_tie(self):
        partial = """
        <new_passes>true</new_passes>
        <rubric>
          <axis><name>clarity</name><winner>maybe</winner></axis>
        </rubric>
        <checks></checks>
        <reasoning>weird value</reasoning>
        """
        _, rubric, _, _ = _parse_response(
            partial, expected_axes=_AXES, expected_checks=_CHECKS,
        )
        by_name = {v.name: v for v in rubric}
        assert by_name["clarity"].winner == "tie"


# ── RubricVote score mapping ───────────────────────────────────────────────

class TestRubricVoteScore:
    def test_new_scores_plus_one(self):
        assert RubricVote("x", "new").score == 1

    def test_tie_scores_zero(self):
        assert RubricVote("x", "tie").score == 0

    def test_old_scores_minus_one(self):
        assert RubricVote("x", "old").score == -1


# ── CheckOutcome.is_pass ────────────────────────────────────────────────────

class TestCheckOutcomeIsPass:
    @pytest.mark.parametrize("result,expected", [
        ("pass", True), ("na", True), ("fail", False),
    ])
    def test_is_pass(self, result, expected):
        assert CheckOutcome(1, result).is_pass is expected
