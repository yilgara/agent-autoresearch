"""Tests for the judging module + JSONLJudgeAdapter.

Covers:
  - Response parser edge cases (multi-block, hallucinated skills, dedup)
  - Caching via results.csv + .judged sidecar
  - JSONLJudgeAdapter delegation behavior
"""

from __future__ import annotations

import csv
import json

import pytest

from agent_autoresearch.adapters.jsonl_judge import JSONLJudgeAdapter
from agent_autoresearch.judging import (
    _parse_judge_response,
    judge_transcripts,
)


_ALLOWED = ["find-restaurant", "book-table"]


# ── Response parser ─────────────────────────────────────────────────────────

class TestParseJudgeResponse:
    def test_empty_response_returns_empty_list(self):
        assert _parse_judge_response("", allowed_skills=_ALLOWED) == []

    def test_single_pass_block(self):
        raw = """
        <judgement>
          <skill>find-restaurant</skill>
          <verdict>pass</verdict>
          <category></category>
          <turn></turn>
          <summary></summary>
        </judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert len(rows) == 1
        assert rows[0]["skill"] == "find-restaurant"
        assert rows[0]["score"] == "pass"
        assert rows[0]["category"] == ""

    def test_single_fail_block_with_metadata(self):
        raw = """
        <judgement>
          <skill>book-table</skill>
          <verdict>fail</verdict>
          <category>wrong_arguments</category>
          <turn>3</turn>
          <summary>Wrong date format.</summary>
        </judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert rows[0]["score"] == "fail"
        assert rows[0]["category"] == "wrong_arguments"
        assert rows[0]["turn"] == "3"
        assert "Wrong date format" in rows[0]["summary"]

    def test_multiple_blocks(self):
        raw = """
        <judgement><skill>find-restaurant</skill><verdict>pass</verdict>
          <category></category><turn></turn><summary></summary></judgement>
        <judgement><skill>book-table</skill><verdict>fail</verdict>
          <category>wrong_args</category><turn>2</turn>
          <summary>iso plz</summary></judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert len(rows) == 2
        skills = {r["skill"] for r in rows}
        assert skills == {"find-restaurant", "book-table"}

    def test_hallucinated_skill_rejected(self):
        raw = """
        <judgement><skill>not-a-real-skill</skill><verdict>fail</verdict>
          <category>x</category><turn>1</turn><summary>nope</summary></judgement>
        """
        assert _parse_judge_response(raw, allowed_skills=_ALLOWED) == []

    def test_skill_none_rejected(self):
        raw = """
        <judgement><skill>none</skill><verdict>pass</verdict>
          <category></category><turn></turn><summary></summary></judgement>
        """
        assert _parse_judge_response(raw, allowed_skills=_ALLOWED) == []

    def test_skill_matching_case_insensitive(self):
        raw = """
        <judgement><skill>FIND-RESTAURANT</skill><verdict>pass</verdict>
          <category></category><turn></turn><summary></summary></judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        # Should normalize back to the canonical form
        assert rows[0]["skill"] == "find-restaurant"

    def test_pass_block_scrubs_failure_metadata(self):
        """A pass block with leftover failure fields should be cleaned."""
        raw = """
        <judgement><skill>book-table</skill><verdict>pass</verdict>
          <category>oops</category><turn>5</turn>
          <summary>shouldnt be here</summary></judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert rows[0]["score"] == "pass"
        assert rows[0]["category"] == ""
        assert rows[0]["turn"] == ""
        assert rows[0]["summary"] == ""

    def test_dedup_fail_wins_over_pass(self):
        """If the LLM emits two blocks for the same skill, the fail wins."""
        raw = """
        <judgement><skill>book-table</skill><verdict>pass</verdict>
          <category></category><turn></turn><summary></summary></judgement>
        <judgement><skill>book-table</skill><verdict>fail</verdict>
          <category>wrong_args</category><turn>2</turn>
          <summary>real failure</summary></judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert len(rows) == 1
        assert rows[0]["score"] == "fail"
        assert rows[0]["category"] == "wrong_args"

    def test_invalid_turn_value_dropped(self):
        raw = """
        <judgement><skill>book-table</skill><verdict>fail</verdict>
          <category>x</category><turn>not-a-number</turn>
          <summary>bad turn</summary></judgement>
        """
        rows = _parse_judge_response(raw, allowed_skills=_ALLOWED)
        assert rows[0]["turn"] == ""

    def test_unparseable_blocks_dont_crash(self):
        raw = "garbage with no tags"
        assert _parse_judge_response(raw, allowed_skills=_ALLOWED) == []


# ── judge_transcripts integration ───────────────────────────────────────────

def _judgement(skill, verdict="pass", **fields):
    """Build a `<judgement>` block for fake LLM responses."""
    return (
        f"<judgement>"
        f"<skill>{skill}</skill>"
        f"<verdict>{verdict}</verdict>"
        f"<category>{fields.get('category', '')}</category>"
        f"<turn>{fields.get('turn', '')}</turn>"
        f"<summary>{fields.get('summary', '')}</summary>"
        f"</judgement>"
    )


def _make_transcripts(tmp_path, *session_ids):
    """Write tiny JSONL transcripts for each id; return the dir."""
    td = tmp_path / "t"
    td.mkdir()
    for sid in session_ids:
        (td / f"{sid}.jsonl").write_text(
            json.dumps({"turn": 1, "user": "hi", "agent": "yo"}) + "\n",
            encoding="utf-8",
        )
    return td


class TestJudgeTranscripts:
    def test_writes_csv_with_correct_schema(self, tmp_path, fake_llm):
        td = _make_transcripts(tmp_path, "s1", "s2")
        fake_llm.push(_judgement("find-restaurant", "fail",
                                  category="x", turn="1", summary="bad"))
        fake_llm.push(_judgement("book-table", "pass"))

        csv_out = tmp_path / "out.csv"
        report = judge_transcripts(
            transcripts_dir=td, skills=_ALLOWED,
            output_csv=csv_out, llm=fake_llm,
        )

        assert csv_out.exists()
        rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
        assert len(rows) == 2
        assert rows[0]["session_id"] == "s1"
        assert rows[0]["skill"] == "find-restaurant"
        assert rows[0]["score"] == "fail"
        assert report.n_rows_written == 2
        assert report.n_pass == 1
        assert report.n_fail == 1

    def test_empty_response_logged_in_judged_sidecar(self, tmp_path, fake_llm):
        """A session with no judgments still gets logged so re-runs skip it."""
        td = _make_transcripts(tmp_path, "s_unrelated")
        fake_llm.push("")   # empty judge response

        csv_out = tmp_path / "out.csv"
        report = judge_transcripts(
            transcripts_dir=td, skills=_ALLOWED,
            output_csv=csv_out, llm=fake_llm,
        )

        assert report.n_unrelated == 1
        assert report.n_rows_written == 0
        sidecar = csv_out.parent / (csv_out.name + ".judged")
        assert sidecar.exists()
        assert "s_unrelated" in sidecar.read_text(encoding="utf-8")

    def test_skip_existing_uses_csv_and_sidecar(self, tmp_path, fake_llm):
        """Re-run with same transcripts → no extra LLM calls."""
        td = _make_transcripts(tmp_path, "s1", "s_unrelated")

        # First run — judges both
        fake_llm.push(_judgement("find-restaurant", "fail",
                                  category="x", turn="1", summary="bad"))
        fake_llm.push("")   # unrelated

        csv_out = tmp_path / "out.csv"
        judge_transcripts(transcripts_dir=td, skills=_ALLOWED,
                          output_csv=csv_out, llm=fake_llm)
        first_call_count = len(fake_llm.calls)
        assert first_call_count == 2

        # Second run — should make zero new LLM calls
        report = judge_transcripts(transcripts_dir=td, skills=_ALLOWED,
                                    output_csv=csv_out, llm=fake_llm)
        assert len(fake_llm.calls) == first_call_count   # unchanged
        assert report.n_skipped_existing == 2
        assert report.n_judged == 0

    def test_one_session_multiple_skills_writes_multiple_rows(self, tmp_path, fake_llm):
        td = _make_transcripts(tmp_path, "s_multi")
        # One transcript → judge returns two blocks
        fake_llm.push(
            _judgement("find-restaurant", "pass")
            + _judgement("book-table", "fail",
                         category="wrong_args", turn="2", summary="iso plz")
        )

        csv_out = tmp_path / "out.csv"
        report = judge_transcripts(transcripts_dir=td, skills=_ALLOWED,
                                    output_csv=csv_out, llm=fake_llm)

        assert report.n_judged == 1            # one session
        assert report.n_rows_written == 2      # two judgments
        rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
        skills = {r["skill"] for r in rows}
        assert skills == {"find-restaurant", "book-table"}

    def test_empty_skills_list_raises(self, tmp_path, fake_llm):
        td = _make_transcripts(tmp_path, "s1")
        with pytest.raises(ValueError, match="`skills` list is empty"):
            judge_transcripts(transcripts_dir=td, skills=[],
                              output_csv=tmp_path / "x.csv", llm=fake_llm)

    def test_missing_transcripts_dir_raises(self, tmp_path, fake_llm):
        with pytest.raises(FileNotFoundError):
            judge_transcripts(
                transcripts_dir=tmp_path / "doesnt-exist",
                skills=_ALLOWED,
                output_csv=tmp_path / "x.csv",
                llm=fake_llm,
            )


# ── JSONLJudgeAdapter ───────────────────────────────────────────────────────

class TestJSONLJudgeAdapter:
    def _setup(self, tmp_path):
        # skills layout
        skills = tmp_path / "skills"
        for name in _ALLOWED:
            (skills / name).mkdir(parents=True)
            (skills / name / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
        # transcripts
        td = _make_transcripts(tmp_path, "s1")
        return skills, td

    def test_lazy_judge_runs_on_first_call(self, tmp_path, fake_llm):
        skills, td = self._setup(tmp_path)
        csv_out = tmp_path / "out.csv"

        adapter = JSONLJudgeAdapter(
            transcripts_dir=td, skills_dir=skills,
            results_csv=csv_out, llm=fake_llm,
        )
        fake_llm.push(_judgement("find-restaurant", "fail",
                                  category="x", turn="1", summary="bad"))

        # Construction is cheap — no LLM calls yet
        assert len(fake_llm.calls) == 0

        # First contract call triggers judge + delegate construction
        targets = adapter.load_targets()
        assert len(fake_llm.calls) == 1
        assert csv_out.exists()
        assert any(t.skill_name == "find-restaurant" for t in targets)

    def test_delegate_cached_across_methods(self, tmp_path, fake_llm):
        skills, td = self._setup(tmp_path)
        csv_out = tmp_path / "out.csv"
        adapter = JSONLJudgeAdapter(
            transcripts_dir=td, skills_dir=skills,
            results_csv=csv_out, llm=fake_llm,
        )
        fake_llm.push(_judgement("find-restaurant", "pass"))

        adapter.load_targets()
        n = len(fake_llm.calls)
        adapter.load_conversations()   # second contract method
        assert len(fake_llm.calls) == n   # no new LLM calls

    def test_force_rejudge_clears_cache(self, tmp_path, fake_llm):
        skills, td = self._setup(tmp_path)
        csv_out = tmp_path / "out.csv"

        # Pre-populate the cache as if a previous run happened
        csv_out.write_text(
            "session_id,skill,score,failure_category,failure_summary,failure_turn,transcript_path\n"
            "s1,find-restaurant,pass,,,,\n",
            encoding="utf-8",
        )
        (csv_out.parent / (csv_out.name + ".judged")).write_text(
            "s1\n", encoding="utf-8",
        )

        adapter = JSONLJudgeAdapter(
            transcripts_dir=td, skills_dir=skills,
            results_csv=csv_out, llm=fake_llm,
            force_rejudge=True,
        )
        # Push a different label so we can detect the re-judge took effect
        fake_llm.push(_judgement("find-restaurant", "fail",
                                  category="redone", turn="1", summary="x"))
        adapter.load_targets()

        rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["score"] == "fail"   # was 'pass' before; now 'fail'
        assert rows[0]["failure_category"] == "redone"

    def test_empty_skills_dir_raises(self, tmp_path, fake_llm):
        skills_dir = tmp_path / "empty"
        skills_dir.mkdir()
        td = _make_transcripts(tmp_path, "s1")
        adapter = JSONLJudgeAdapter(
            transcripts_dir=td, skills_dir=skills_dir,
            results_csv=tmp_path / "x.csv", llm=fake_llm,
        )
        with pytest.raises(FileNotFoundError, match="No skills found"):
            adapter.load_targets()

    def test_discovers_flat_md_skills(self, tmp_path, fake_llm):
        # Layout 2: skills/<name>.md (no folders)
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "find-restaurant.md").write_text("# x", encoding="utf-8")
        (skills / "book-table.md").write_text("# y", encoding="utf-8")

        td = _make_transcripts(tmp_path, "s1")
        adapter = JSONLJudgeAdapter(
            transcripts_dir=td, skills_dir=skills,
            results_csv=tmp_path / "out.csv", llm=fake_llm,
        )
        fake_llm.push(_judgement("find-restaurant", "pass"))
        adapter.load_targets()
        # If we got here without "No skills found", discovery worked.
        assert len(fake_llm.calls) == 1
