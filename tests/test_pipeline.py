"""Tests for the pipeline orchestrator.

Focus on the `run_target` skip paths and `run_pipeline` failure
isolation — the bookkeeping policies that the strategy stages don't
own. Strategy stages have their own (LLM-mocked) tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import Conversation, Target
from agent_autoresearch.core.skill_io import SkillIO, UNATTRIBUTED
from agent_autoresearch.pipeline import run_pipeline, run_target


# ── Helpers ─────────────────────────────────────────────────────────────────

class _MemorySkillIO(SkillIO):
    """In-memory SkillIO for tests — no disk I/O for skill load/write."""

    def __init__(self, skills: dict[str, str] | None = None):
        self.skills = dict(skills or {})
        self.written: list[tuple[str, str]] = []

    def load(self, name: str) -> str:
        if name not in self.skills:
            raise FileNotFoundError(name)
        return self.skills[name]

    def write_version(self, name, new_content, *, run_id, outputs_root):
        from agent_autoresearch.core.skill_io import _write_version_to_filesystem
        # Reuse the standard writer for output layout consistency
        old = self.skills.get(name, "")
        self.written.append((name, new_content))
        return _write_version_to_filesystem(
            name, old_content=old, new_content=new_content,
            run_id=run_id, outputs_root=outputs_root,
        )


class _StaticAdapter(Adapter):
    """Adapter that returns whatever you hand it."""
    name = "test_static"

    def __init__(self, targets: list[Target], conversations: list[Conversation]):
        self._targets = targets
        self._conversations = conversations

    def load_targets(self):
        return list(self._targets)

    def load_conversations(self):
        return list(self._conversations)


# ── run_target skip paths ───────────────────────────────────────────────────

class TestRunTargetSkipPaths:
    def test_unattributed_target_short_circuits_to_skip(self, tmp_path, fake_llm):
        target = Target(skill_name=UNATTRIBUTED)
        skill_io = _MemorySkillIO()
        result = run_target(
            target, conversations={},
            skill_io=skill_io, run_id="test_run",
            outputs_root=tmp_path, llm=fake_llm,
        )
        assert result.verdict.label == "SKIP"
        assert "UNATTRIBUTED" in result.verdict.reason
        assert len(fake_llm.calls) == 0
        # Verdict file written
        assert (tmp_path / "test_run" / UNATTRIBUTED / "verdict.md").exists()

    def test_skill_not_found_returns_reject(self, tmp_path, fake_llm):
        target = Target(skill_name="missing-skill")
        skill_io = _MemorySkillIO()   # empty — no skills
        result = run_target(
            target, conversations={},
            skill_io=skill_io, run_id="test_run",
            outputs_root=tmp_path, llm=fake_llm,
        )
        assert result.verdict.label == "REJECT"
        assert result.error is not None
        assert "FileNotFoundError" in result.error
        assert len(fake_llm.calls) == 0

    def test_propose_skip_short_circuits(self, tmp_path, fake_llm):
        """Propose returns 'skip' → no critic, no replay, just SKIP verdict."""
        target = Target(
            skill_name="my-skill",
            evidence=[],
            fix_session_ids=[],
            regression_baseline_ids=[],
        )
        skill_io = _MemorySkillIO({"my-skill": "# Original SKILL.md"})

        # Two LLM calls expected: program (always runs first) + propose (returns skip)
        fake_llm.push("# Strategy doc\n\nSomething.")   # program
        fake_llm.push(
            "<action>skip</action>\n"
            "<reasoning>not enough evidence</reasoning>"
        )   # propose

        result = run_target(
            target, conversations={},
            skill_io=skill_io, run_id="test_run",
            outputs_root=tmp_path, llm=fake_llm,
        )
        assert result.verdict.label == "SKIP"
        # Exactly two LLM calls — no critic, no replay
        assert len(fake_llm.calls) == 2
        # Output folder has program.md + skip.md + verdict.md (no critic/replay)
        target_dir = tmp_path / "test_run" / "my-skill"
        assert (target_dir / "program.md").exists()
        assert (target_dir / "skip.md").exists()
        assert (target_dir / "verdict.md").exists()
        assert not (target_dir / "critic.md").exists()
        assert not (target_dir / "replay.md").exists()


# ── run_pipeline failure isolation ──────────────────────────────────────────

class TestRunPipelineIsolation:
    def test_dry_run_makes_no_llm_calls(self, tmp_path):
        """Dry-run path skips LLM construction entirely — no API key needed."""
        target = Target(skill_name="my-skill")
        adapter = _StaticAdapter(targets=[target], conversations=[])
        result = run_pipeline(
            adapter,
            skill_io=_MemorySkillIO(),
            outputs_root=tmp_path,
            top_n=1,
            dry_run=True,
            # Note: no llm= passed; default_llm_provider() would raise
            # without an API key, so this verifies the dry-run path stays lazy
        )
        assert result.dry_run is True
        assert len(result.targets) == 1
        # summary.md gets written even in dry-run mode
        assert (tmp_path / result.run_id / "summary.md").exists()

    def test_unattributed_targets_excluded_from_top_n(self, tmp_path, fake_llm):
        good = Target(skill_name="real-skill")
        sentinel = Target(skill_name=UNATTRIBUTED)
        adapter = _StaticAdapter(
            targets=[sentinel, good, sentinel],   # sentinel ranked first; should skip
            conversations=[],
        )
        result = run_pipeline(
            adapter, skill_io=_MemorySkillIO(),
            outputs_root=tmp_path, top_n=1, dry_run=True,
        )
        # Only the real one survives the filter
        assert len(result.targets) == 1
        assert result.targets[0].skill_name == "real-skill"

    def test_per_target_exception_isolated(self, tmp_path, fake_llm):
        """One bad target shouldn't kill the whole run."""
        adapter = _StaticAdapter(
            targets=[
                Target(skill_name="skill-A"),   # will fail (skill not in SkillIO)
                Target(skill_name="skill-B"),   # will fail (skill not in SkillIO)
            ],
            conversations=[],
        )
        result = run_pipeline(
            adapter, skill_io=_MemorySkillIO(),   # empty
            llm=fake_llm,
            outputs_root=tmp_path, top_n=2,
            raise_on_error=False,
        )
        assert len(result.target_results) == 2
        # Both should be REJECTs — first one is the FileNotFoundError path,
        # not an exception bubbling up
        assert all(r.verdict.label == "REJECT" for r in result.target_results)

    def test_raise_on_error_propagates(self, tmp_path, fake_llm):
        """When `raise_on_error=True`, the first exception bubbles up."""

        class _BadAdapter(_StaticAdapter):
            def load_conversations(self):
                raise RuntimeError("boom")

        adapter = _BadAdapter(
            targets=[Target(skill_name="x")], conversations=[],
        )
        with pytest.raises(RuntimeError, match="boom"):
            run_pipeline(
                adapter, skill_io=_MemorySkillIO({"x": "# x"}),
                llm=fake_llm, outputs_root=tmp_path,
                raise_on_error=True,
            )


# ── Aggregate counts ────────────────────────────────────────────────────────

class TestPipelineRunResult:
    def test_verdict_counts_aggregated(self, tmp_path, fake_llm):
        """Two reject targets (skill not found) → n_reject == 2."""
        adapter = _StaticAdapter(
            targets=[
                Target(skill_name="missing-A"),
                Target(skill_name="missing-B"),
            ],
            conversations=[],
        )
        result = run_pipeline(
            adapter, skill_io=_MemorySkillIO(),   # empty — both skills missing
            llm=fake_llm,
            outputs_root=tmp_path, top_n=99, dry_run=False,
        )
        assert result.n_reject == 2
        assert result.n_accept == 0
        assert result.n_skip == 0
        # Sanity: counters dict matches per-target result labels
        assert result.verdicts_by_label == {"REJECT": 2}
