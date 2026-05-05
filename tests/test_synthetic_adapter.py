"""Tests for the synthetic adapter — verifies the fixtures are well-formed.

The synthetic adapter is pure data (no parsing, no I/O). The point of
these tests isn't logic coverage — it's catching accidental drift in
the hardcoded fixtures (a typo'd session_id breaking the multi-skill
demo, evidence pointing at a session that no longer exists, etc.).
"""

from __future__ import annotations

from agent_autoresearch.adapters.synthetic import SyntheticAdapter
from agent_autoresearch.core.skill_io import UNATTRIBUTED


class TestSyntheticAdapterFixtures:
    def setup_method(self):
        self.adapter = SyntheticAdapter()
        self.targets = self.adapter.load_targets()
        self.convs = self.adapter.load_conversations()
        self.conv_ids = {c.session_id for c in self.convs}

    def test_adapter_name(self):
        assert SyntheticAdapter.name == "synthetic"

    def test_two_targets(self):
        names = {t.skill_name for t in self.targets}
        assert names == {"find-restaurant", "book-table"}

    def test_no_unattributed_targets(self):
        assert all(t.skill_name != UNATTRIBUTED for t in self.targets)

    def test_every_referenced_session_has_a_conversation(self):
        """Adapter authors must keep target session_ids consistent with
        what `load_conversations` returns. Catches drift in the fixtures."""
        referenced: set[str] = set()
        for t in self.targets:
            referenced.update(t.fix_session_ids)
            referenced.update(t.regression_baseline_ids)
        missing = referenced - self.conv_ids
        assert not missing, f"Targets reference sessions with no transcript: {missing}"

    def test_evidence_session_ids_match_fix_targets(self):
        """Each Evidence's session_id should correspond to a fix-target session."""
        for t in self.targets:
            for ev in t.evidence:
                sid = ev.details.get("session_id")
                assert sid in t.fix_session_ids, (
                    f"Evidence for {t.skill_name} points at sess {sid!r} "
                    "which isn't in fix_session_ids"
                )

    def test_evidence_uses_focus_turn_key(self):
        """Replay reads `focus_turn`, not `turn`. Regression test."""
        for t in self.targets:
            for ev in t.evidence:
                assert "focus_turn" in ev.details, (
                    f"Evidence for {t.skill_name} missing 'focus_turn' key — "
                    f"replay would default to last-turn"
                )
                assert "turn" not in ev.details, (
                    f"Evidence for {t.skill_name} uses obsolete 'turn' key"
                )

    def test_multi_skill_session_present(self):
        """Documented invariant: sess_004 is a baseline for find-restaurant
        AND a fix-target for book-table. The fixture exists specifically
        to exercise the multi-skill path; if this breaks the demo no
        longer demonstrates that case."""
        by_name = {t.skill_name: t for t in self.targets}
        assert "sess_004" in by_name["find-restaurant"].regression_baseline_ids
        assert "sess_004" in by_name["book-table"].fix_session_ids

    def test_returns_defensive_copies(self):
        """Mutating the returned list shouldn't affect subsequent calls."""
        first = self.adapter.load_targets()
        first.clear()
        second = self.adapter.load_targets()
        assert len(second) > 0

    def test_focus_turn_within_session_length(self):
        """Each evidence's focus_turn must point at an actual turn in
        the session — otherwise replay's pick_focus_turn clamps it,
        which probably isn't what the fixture author intended."""
        convs_by_id = {c.session_id: c for c in self.convs}
        for t in self.targets:
            for ev in t.evidence:
                sid = ev.details["session_id"]
                conv = convs_by_id[sid]
                ft = ev.details["focus_turn"]
                assert 1 <= ft <= conv.n_turns, (
                    f"{t.skill_name}: evidence focus_turn={ft} but session "
                    f"{sid} only has {conv.n_turns} turns"
                )
