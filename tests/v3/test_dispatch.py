"""End-to-end dispatch test: pipeline.run_target(strategy='v3').

Exercises the full v3 path through the orchestrator:
  - registry resolves "v3" → strategies.v3 module
  - run_target builds atomic-mutation validators (with rubric/checks
    plumbed for v3) + calls v3.propose
  - LLM-driven `done` action short-circuits the propose loop

The skip-path is the cheapest end-to-end exercise (2 LLM calls). A
fuller "all gates pass" e2e would need many more canned responses
because v3's per-iteration validators each call critic + soft_replay
which themselves run responder + judge. The skip path proves the
dispatch wiring without that cost.
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
class TestV3Dispatch:
    def test_done_signal_short_circuits_v3_loop(self, tmp_path, fake_llm):
        """LLM signals done on first iteration → no accepted changes →
        propose returns skip → SKIP verdict, no critic/replay calls."""
        # Canned responses:
        # 1. program (free-form; downstream not parsed when LLM says done)
        # 2. v3 propose: <action>done</action>
        fake_llm.push("# Strategy\n\nUnclear evidence.\n")
        fake_llm.push(
            "<action>done</action>\n"
            "<reasoning>insufficient signal to commit to a change</reasoning>"
        )

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)
        target = adapter.load_targets()[0]
        conversations = {c.session_id: c for c in adapter.load_conversations()}

        result = run_target(
            target, conversations,
            skill_io=skill_io, llm=fake_llm,
            run_id="test_v3_dispatch",
            outputs_root=tmp_path,
            strategy="v3",
        )

        assert result.verdict.label == "SKIP"
        assert len(fake_llm.calls) == 2

        target_dir = tmp_path / "test_v3_dispatch" / "find-restaurant"
        assert (target_dir / "program.md").exists()
        assert (target_dir / "skip.md").exists()
        assert (target_dir / "verdict.md").exists()

    def test_module_dispatched_via_registry(self, tmp_path, fake_llm):
        """Confirm v3 module is actually loaded for strategy='v3'."""
        from agent_autoresearch.strategies.registry import get_strategy
        s = get_strategy("v3")
        assert s.STRATEGY_VERSION == "v3"
        # v3 has rubric + binary check types that v1/v2 don't
        assert hasattr(s, "RubricAxis")
        assert hasattr(s, "BinaryCheck")
        assert hasattr(s, "RUBRIC_AXIS_COUNT")
        assert hasattr(s, "MIN_BINARY_CHECKS")
