"""End-to-end dispatch test: pipeline.run_target(strategy='v2').

Exercises the full v2 path through the orchestrator:
  - registry resolves "v2" → strategies.v2 module
  - run_target builds atomic-mutation validators + calls v2.propose
  - LLM-driven `done` action short-circuits the propose loop
  - skip-path artifacts written; no critic/replay called

This is the cheapest e2e path through v2 (2 LLM calls). The full
"all gates pass" path lives in tests/v3/test_dispatch.py — same
structure works for v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._synthetic_fixture import SyntheticAdapter
from agent_autoresearch.core.skill_io import FilesystemSkillIO
from agent_autoresearch.pipeline import run_target


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SKILLS = REPO_ROOT / "examples" / "synthetic-skills"


@pytest.mark.skipif(
    not EXAMPLE_SKILLS.exists(),
    reason="examples/synthetic-skills/ not found",
)
class TestV2Dispatch:
    def test_done_signal_short_circuits_v2_loop(self, tmp_path, fake_llm):
        """LLM says <action>done</action> on first iteration → no accepted
        changes → propose returns skip → orchestrator emits SKIP verdict
        without calling critic or replay. Total LLM calls: 2 (program + propose).
        """
        # Canned responses (in order):
        # 1. program.md content (free-form markdown)
        # 2. v2 propose: <action>done</action>
        fake_llm.push("# Strategy\n\nNo useful change apparent.\n")
        fake_llm.push(
            "<action>done</action>\n"
            "<reasoning>nothing more to add</reasoning>"
        )

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)
        target = adapter.load_targets()[0]   # find-restaurant
        conversations = {c.session_id: c for c in adapter.load_conversations()}

        result = run_target(
            target, conversations,
            skill_io=skill_io, llm=fake_llm,
            run_id="test_v2_dispatch",
            outputs_root=tmp_path,
            strategy="v2",
        )

        assert result.verdict.label == "SKIP"
        assert len(fake_llm.calls) == 2

        target_dir = tmp_path / "test_v2_dispatch" / "find-restaurant"
        assert (target_dir / "program.md").exists()
        assert (target_dir / "skip.md").exists()
        assert (target_dir / "verdict.md").exists()
        assert not (target_dir / "critic.md").exists()
        assert not (target_dir / "replay.md").exists()

    def test_module_dispatched_via_registry(self, tmp_path, fake_llm):
        """run_target(strategy='v2') actually uses v2's modules — verified
        by checking that v2's STRATEGY_VERSION constant is what gets read."""
        from agent_autoresearch.strategies.registry import get_strategy
        s = get_strategy("v2")
        assert s.STRATEGY_VERSION == "v2"
        # And the propose function comes from v2, not v1
        assert "strategies.v2" in s.propose.__module__
