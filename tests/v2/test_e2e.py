"""End-to-end "all gates pass" test for v2.

Drives `pipeline.run_target(strategy='v2')` against the synthetic
adapter + bundled example skills, with FakeLLM canned responses for
every LLM call along the path. Asserts:

  - verdict label is ACCEPT
  - the atomic-mutation log captures both accepted changes
  - all expected output files are written
  - the canned-response queue is consumed exactly (no leftovers, no
    extras requested) — proves the LLM call count is what we expect

## Call sequence (with top_n=1, fix_sample=1, baseline_sample=1)

For find-restaurant which has 2 evidence items:

```
 1. build_program
 2-3.  evidence #1 attempt 1:
       2. v2 propose         (<action>edit</action> + new SKILL.md)
       3. critic_per_attempt (<verdict>APPROVE</verdict>)
 4-5.  evidence #2 attempt 1: same 2 calls
 6.    final_critic                  (<verdict>APPROVE</verdict>)
 7-10. final_replay (1 fix + 1 baseline)
       7.  fix responder
       8.  fix judge          (new_passes=true)
       9.  baseline responder
       10. baseline judge     (new_passes=true)
```

Total: 10 LLM calls. The orchestrator reuses propose()'s
final_critic + final_replay results — no canonical re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._synthetic_fixture import SyntheticAdapter
from agent_autoresearch.core.skill_io import FilesystemSkillIO
from agent_autoresearch.pipeline import run_target


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SKILLS = REPO_ROOT / "examples" / "synthetic-skills"


# ── Canned response strings ─────────────────────────────────────────────────

_PROGRAM_MD = """\
# Improvement Strategy — find-restaurant

## Target
Agent ignores user-stated dietary preferences across 2 sessions.

## Evidence from logs
- sess_001 turn 2: recommended steakhouse for a vegan request
- sess_002 turn 1: refused with "check Yelp"

## Current skill
The skill defines search_restaurants but lacks explicit guidance about
dietary filters.

## Proposed change
Add an explicit rule that user-stated dietary preferences must be
passed as the filter argument to search_restaurants.

## What NOT to change
- Persona
- Tool list
- Heading structure
"""


def _propose_edit(label: str) -> str:
    """Atomic-edit response — body just needs to be > 200 chars to satisfy
    the parser's "looks like a real SKILL.md" heuristic."""
    body = (
        f"# find-restaurant ({label})\n\n"
        "You help users find restaurants.\n\n"
        "## Tools\n"
        "- `search_restaurants(q, filter)` — returns matches.\n\n"
        "## Guidelines\n"
        "1. Always call `search_restaurants` for restaurant lookups.\n"
        "2. **Always pass user-stated dietary preferences as the filter "
        "argument.** This is non-negotiable.\n"
        "3. Return the top match by name.\n"
    )
    return (
        "<action>edit</action>\n"
        f"<reasoning>Atomic step '{label}' addressing the dietary constraint.</reasoning>\n"
        f"<new_skill_md>\n{body}\n</new_skill_md>"
    )


_CRITIC_APPROVE = """\
<verdict>APPROVE</verdict>
<reasoning>Targeted edit, traces to evidence.</reasoning>
<concerns></concerns>
"""

_RESPONDER = """\
<tool_plan>Call search_restaurants(q="downtown", filter="vegan").</tool_plan>
<reply>Found Green Leaf — vegan-friendly downtown.</reply>
"""

_JUDGE_PASS = """\
<new_passes>true</new_passes>
<reasoning>The new reply respects the vegan filter and clears the bar.</reasoning>
"""


# ── Test ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not EXAMPLE_SKILLS.exists(),
    reason="examples/synthetic-skills/ not found",
)
class TestV2EndToEndAccept:
    def test_full_pipeline_produces_accept(self, tmp_path, fake_llm):
        # ── Queue all 10 canned responses in pipeline order ─────────────
        responses = [
            _PROGRAM_MD,                       # 1. build_program
            # evidence #1 attempt
            _propose_edit("ev1"),              # 2. propose
            _CRITIC_APPROVE,                   # 3. critic_per_attempt
            # evidence #2 attempt
            _propose_edit("ev2"),              # 4.
            _CRITIC_APPROVE,                   # 5.
            # final pass inside propose (orchestrator reuses these)
            _CRITIC_APPROVE,                   # 6. final_critic
            _RESPONDER,                        # 7. final_replay fix responder
            _JUDGE_PASS,                       # 8. final_replay fix judge
            _RESPONDER,                        # 9. final_replay baseline responder
            _JUDGE_PASS,                       # 10. final_replay baseline judge
        ]
        for r in responses:
            fake_llm.push(r)

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)
        target = adapter.load_targets()[0]   # find-restaurant (rank 0)
        conversations = {c.session_id: c for c in adapter.load_conversations()}

        result = run_target(
            target, conversations,
            skill_io=skill_io, llm=fake_llm,
            run_id="test_v2_e2e",
            outputs_root=tmp_path,
            fix_sample=1, baseline_sample=1,
            strategy="v2",
        )

        # 1. Verdict label
        assert result.verdict.label == "ACCEPT", (
            f"Expected ACCEPT, got {result.verdict.label}: {result.verdict.reason}"
        )

        # 2. v2 atomic-mutation bookkeeping
        prop = result.propose_result
        assert prop is not None
        assert prop.action == "edit"
        assert len(prop.accepted_log) == 2     # both evidence got accepted
        assert prop.combined_check_passed is True
        assert prop.rolled_back_steps == 0

        # 3. Exactly the expected number of LLM calls
        assert len(fake_llm.calls) == 10, (
            f"Expected 10 LLM calls, got {len(fake_llm.calls)}"
        )

        # 4. All expected artifacts on disk
        target_dir = tmp_path / "test_v2_e2e" / "find-restaurant"
        for name in (
            "program.md",
            "v_old.md",
            "v_new.md",
            "diff.txt",
            "propose_reasoning.md",
            "critic.md",
            "replay.md",
            "verdict.md",
        ):
            assert (target_dir / name).exists(), f"missing {name}"

        # 5. v_new contains the atomic-edit body
        v_new = (target_dir / "v_new.md").read_text(encoding="utf-8")
        assert "non-negotiable" in v_new

        # 6. v1/v2 replay rates land where expected
        assert result.verdict.fix_target_score == 1.0   # 1/1 fix new_passes
        assert result.verdict.regression_score == 1.0   # 1/1 baseline new_passes
