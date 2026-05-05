"""Tests for the CSV adapter.

The CSV adapter is the most external-shape-y bit of the library —
schema changes, regex tweaks, parsing edge cases would all break here
first. This file is intentionally exhaustive.
"""

from __future__ import annotations

import json

import pytest

from agent_autoresearch.adapters.csv import CSVAdapter, _parse_score


# ── Score parsing ───────────────────────────────────────────────────────────

class TestParseScore:
    @pytest.mark.parametrize("raw", ["1", "pass", "passed", "TRUE", "ok", "Success"])
    def test_pass_variants(self, raw):
        assert _parse_score(raw) is True

    @pytest.mark.parametrize("raw", ["0", "fail", "FAILED", "false", "ko", "  Failure  "])
    def test_fail_variants(self, raw):
        assert _parse_score(raw) is False

    @pytest.mark.parametrize("raw", ["", "maybe", "skip", None])
    def test_unknown_raises(self, raw):
        with pytest.raises(ValueError, match="Unrecognized score"):
            _parse_score(raw or "")


# ── Target construction ─────────────────────────────────────────────────────

class TestLoadTargets:
    def test_groups_by_skill(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        targets = adapter.load_targets()
        names = [t.skill_name for t in targets]
        assert set(names) == {"find-restaurant", "book-table"}

    def test_ranks_by_failure_count_desc(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        targets = adapter.load_targets()
        # find-restaurant has 2 fails vs book-table's 1
        assert targets[0].skill_name == "find-restaurant"
        assert targets[0].rank == 0
        assert targets[1].skill_name == "book-table"
        assert targets[1].rank == 1

    def test_fix_and_baseline_split(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_name = {t.skill_name: t for t in adapter.load_targets()}

        find = by_name["find-restaurant"]
        assert sorted(find.fix_session_ids) == ["sess_001", "sess_002"]
        assert sorted(find.regression_baseline_ids) == ["sess_003", "sess_004"]

        book = by_name["book-table"]
        assert book.fix_session_ids == ["sess_004"]
        assert book.regression_baseline_ids == []

    def test_multi_skill_session_in_both_roles(self, tiny_csv_data):
        """sess_004 passes find-restaurant and fails book-table — should
        appear as a baseline for one and a fix for the other."""
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_name = {t.skill_name: t for t in adapter.load_targets()}
        assert "sess_004" in by_name["find-restaurant"].regression_baseline_ids
        assert "sess_004" in by_name["book-table"].fix_session_ids

    def test_evidence_built_only_for_failures_with_summary(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_name = {t.skill_name: t for t in adapter.load_targets()}
        # find-restaurant: 2 failures, both have summaries → 2 evidence
        assert len(by_name["find-restaurant"].evidence) == 2
        # book-table: 1 failure with summary → 1 evidence
        assert len(by_name["book-table"].evidence) == 1

    def test_focus_turn_key_used_not_turn(self, tiny_csv_data):
        """Regression test: replay reads `focus_turn`, not `turn`. Adapter
        must write the canonical key or the failure_turn value is ignored."""
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_name = {t.skill_name: t for t in adapter.load_targets()}
        ev = by_name["book-table"].evidence[0]
        assert ev.details["focus_turn"] == 2
        assert "turn" not in ev.details   # specifically NOT this key

    def test_evidence_per_skill_per_session(self, tiny_csv_data):
        """sess_004 has rows for two skills — each Target should carry
        its own Evidence, not cross-pollinate."""
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_name = {t.skill_name: t for t in adapter.load_targets()}
        book_evidence = by_name["book-table"].evidence
        assert len(book_evidence) == 1
        assert book_evidence[0].category == "wrong_args"
        assert book_evidence[0].details["session_id"] == "sess_004"


# ── CSV validation ──────────────────────────────────────────────────────────

class TestCSVValidation:
    def test_missing_csv_raises(self, tmp_path):
        adapter = CSVAdapter(
            results_csv=tmp_path / "nope.csv",
            transcripts_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            adapter.load_targets()

    def test_missing_required_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("session_id,skill\nsess_x,find\n", encoding="utf-8")
        adapter = CSVAdapter(results_csv=bad, transcripts_dir=tmp_path)
        with pytest.raises(ValueError, match="missing required columns"):
            adapter.load_targets()

    def test_empty_csv_raises(self, tmp_path):
        bad = tmp_path / "empty.csv"
        bad.write_text("", encoding="utf-8")
        adapter = CSVAdapter(results_csv=bad, transcripts_dir=tmp_path)
        with pytest.raises(ValueError, match="empty"):
            adapter.load_targets()

    def test_unknown_score_raises_at_load_targets(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "session_id,skill,score\nsess_x,find,maybe\n",
            encoding="utf-8",
        )
        adapter = CSVAdapter(results_csv=bad, transcripts_dir=tmp_path)
        with pytest.raises(ValueError, match="Unrecognized score"):
            adapter.load_targets()


# ── Conversations ───────────────────────────────────────────────────────────

class TestLoadConversations:
    def test_returns_one_per_unique_session_id(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        convs = adapter.load_conversations()
        assert len(convs) == 4
        assert {c.session_id for c in convs} == {
            "sess_001", "sess_002", "sess_003", "sess_004",
        }

    def test_dedupes_multi_skill_session(self, tiny_csv_data, monkeypatch):
        """sess_004 has 2 CSV rows but the file should be read once."""
        from agent_autoresearch.adapters import csv as csv_mod
        calls = []
        original = csv_mod._load_transcript

        def counted(path, sid):
            calls.append(sid)
            return original(path, sid)

        monkeypatch.setattr(csv_mod, "_load_transcript", counted)

        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        adapter.load_conversations()
        assert calls.count("sess_004") == 1

    def test_jsonl_loads_with_tool_calls(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_id = {c.session_id: c for c in adapter.load_conversations()}
        sess_001 = by_id["sess_001"]
        assert sess_001.n_turns == 2
        assert sess_001.turns[1].tool_calls[0].name == "search"

    def test_json_format_supported(self, tiny_csv_data):
        adapter = CSVAdapter(
            results_csv=tiny_csv_data["csv"],
            transcripts_dir=tiny_csv_data["transcripts_dir"],
        )
        by_id = {c.session_id: c for c in adapter.load_conversations()}
        sess_003 = by_id["sess_003"]   # written as .json in conftest
        assert sess_003.n_turns == 1
        assert sess_003.turns[0].agent == "Green Leaf"


# ── Transcript path resolution ──────────────────────────────────────────────

class TestTranscriptResolution:
    def test_explicit_path_overrides_default(self, tmp_path):
        # Default location: tmp_path/transcripts/sess_x.jsonl (won't exist)
        # Explicit path elsewhere
        custom = tmp_path / "custom_logs"
        custom.mkdir()
        (custom / "explicit.jsonl").write_text(
            json.dumps({"turn": 1, "user": "hi", "agent": "yo"}) + "\n",
            encoding="utf-8",
        )

        csv_path = tmp_path / "r.csv"
        csv_path.write_text(
            "session_id,skill,score,transcript_path\n"
            "sess_x,find,fail,custom_logs/explicit.jsonl\n",
            encoding="utf-8",
        )
        adapter = CSVAdapter(
            results_csv=csv_path,
            transcripts_dir=tmp_path / "transcripts",  # doesn't exist
        )
        convs = adapter.load_conversations()
        assert len(convs) == 1
        assert convs[0].turns[0].agent == "yo"

    def test_jsonl_preferred_over_json_when_both_exist(self, tmp_path):
        td = tmp_path / "t"
        td.mkdir()
        (td / "sess_x.jsonl").write_text(
            json.dumps({"turn": 1, "user": "j", "agent": "from-jsonl"}) + "\n",
            encoding="utf-8",
        )
        (td / "sess_x.json").write_text(
            json.dumps({"turns": [{"turn": 1, "user": "j", "agent": "from-json"}]}),
            encoding="utf-8",
        )
        csv_path = tmp_path / "r.csv"
        csv_path.write_text(
            "session_id,skill,score\nsess_x,find,fail\n",
            encoding="utf-8",
        )
        adapter = CSVAdapter(results_csv=csv_path, transcripts_dir=td)
        convs = adapter.load_conversations()
        assert convs[0].turns[0].agent == "from-jsonl"

    def test_missing_transcript_raises(self, tmp_path):
        td = tmp_path / "t"
        td.mkdir()
        csv_path = tmp_path / "r.csv"
        csv_path.write_text(
            "session_id,skill,score\nsess_x,find,fail\n",
            encoding="utf-8",
        )
        adapter = CSVAdapter(results_csv=csv_path, transcripts_dir=td)
        with pytest.raises(FileNotFoundError):
            adapter.load_conversations()
