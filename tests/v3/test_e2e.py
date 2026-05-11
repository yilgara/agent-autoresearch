"""End-to-end "all gates pass" test for v3.

Same shape as v2's e2e, but the program response now contains a
3-axis rubric and ≥5 binary checks, and every judge response carries
the full v3 schema (`<winner>`, `<rubric>`, `<checks>`, `<reasoning>`).

Asserts:
  - verdict label is ACCEPT
  - all 4 v3 rates land at expected values (fix_rate, regression_rate,
    rubric_improvement_rate, binary_checks_pass_rate)
  - per-evidence atomic accepts logged
  - LLM call queue is consumed exactly (proves call count)
  - every artifact written to disk

## Call sequence

Same as v2 e2e (9 calls — no final critic, orchestrator reuses
propose's final replay result):

```
 1. build_program       (v3 — emits rubric + binary checks)
 2-3.  ev #1 atomic loop  (propose + critic)
 4-5.  ev #2 atomic loop
 6-9.  final_replay     (1 fix + 1 baseline; responder + judge each)
```

The judge calls (rounds 7 and 9) use v3's 3-signal output schema;
everything else is identical to v2.
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

_PROGRAM_MD_V3 = """\
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

## Rubric — 3 axes, scored 1–3

- **dietary_constraint_handling**: Excellent means agent always passes user-stated dietary preferences as the filter argument.
- **query_specificity**: Excellent means search query includes all named user constraints.
- **result_grounding**: Excellent means recommended restaurant exists in the tool output and respects all stated filters.

## Binary checks — invariants the new prompt must preserve

- [ ] Does the agent always pass user-stated dietary preferences as the filter argument?
- [ ] Does the agent never recommend a result contradicting a stated dietary preference?
- [ ] Does the agent never refuse a reasonable lookup with 'check Yelp'?
- [ ] Does the agent always call search_restaurants for restaurant lookups?
- [ ] Does the agent never recommend a restaurant not in the tool output?
"""


def _propose_edit(label: str) -> str:
    body = (
        f"# find-restaurant ({label})\n\n"
        "You help users find restaurants.\n\n"
        "## Tools\n"
        "- `search_restaurants(q, filter)` — returns matches.\n\n"
        "## Guidelines\n"
        "1. Always call `search_restaurants` for restaurant lookups.\n"
        "2. **Always pass user-stated dietary preferences as the filter "
        "argument.** This is non-negotiable.\n"
        "3. Return the top match by name from the tool output.\n"
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


def _judge_v3(*, new_passes: bool, rubric_winners: tuple,
              all_pass: bool = True) -> str:
    """Build a v3 judge response with all three signals.

    `rubric_winners` is a 3-tuple of "new"/"old"/"tie" (one per axis).
    `all_pass` controls whether all 5 binary checks pass.
    """
    axis_names = ["dietary_constraint_handling", "query_specificity", "result_grounding"]
    rubric_xml = "\n".join(
        f"  <axis><name>{name}</name><winner>{w}</winner></axis>"
        for name, w in zip(axis_names, rubric_winners)
    )
    check_result = "pass" if all_pass else "fail"
    checks_xml = "\n".join(
        f"  <check><id>{i}</id><result>{check_result}</result></check>"
        for i in range(1, 6)
    )
    np_str = "true" if new_passes else "false"
    return (
        f"<new_passes>{np_str}</new_passes>\n"
        f"<rubric>\n{rubric_xml}\n</rubric>\n"
        f"<checks>\n{checks_xml}\n</checks>\n"
        "<reasoning>Synthesized v3 judge output for tests.</reasoning>"
    )


# Fix-session judge: NEW passes, rubric all-new (positive score), checks pass
_JUDGE_FIX = _judge_v3(
    new_passes=True,
    rubric_winners=("new", "new", "tie"),
    all_pass=True,
)

# Baseline-session judge: NEW passes, rubric ties (score 0 — but baselines
# don't feed rubric), checks pass
_JUDGE_BASELINE = _judge_v3(
    new_passes=True,
    rubric_winners=("tie", "tie", "tie"),
    all_pass=True,
)


# ── Test ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not EXAMPLE_SKILLS.exists(),
    reason="examples/synthetic-skills/ not found",
)
class TestV3EndToEndAccept:
    def test_full_pipeline_produces_accept(self, tmp_path, fake_llm):
        responses = [
            _PROGRAM_MD_V3,                    # 1. build_program (with rubric+checks)
            # evidence #1 atomic loop
            _propose_edit("ev1"),              # 2. propose
            _CRITIC_APPROVE,                   # 3. critic_per_attempt
            # evidence #2 atomic loop
            _propose_edit("ev2"),              # 4.
            _CRITIC_APPROVE,                   # 5.
            # final replay (orchestrator reuses these)
            _RESPONDER,                        # 6. final_replay fix responder
            _JUDGE_FIX,                        # 7. final_replay fix judge
            _RESPONDER,                        # 8. final_replay baseline responder
            _JUDGE_BASELINE,                   # 9. final_replay baseline judge
        ]
        for r in responses:
            fake_llm.push(r)

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)
        target = adapter.load_targets()[0]
        conversations = {c.session_id: c for c in adapter.load_conversations()}

        result = run_target(
            target, conversations,
            skill_io=skill_io, llm=fake_llm,
            run_id="test_v3_e2e",
            outputs_root=tmp_path,
            fix_sample=1, baseline_sample=1,
            strategy="v3",
        )

        # 1. Verdict label
        assert result.verdict.label == "ACCEPT", (
            f"Expected ACCEPT, got {result.verdict.label}: {result.verdict.reason}"
        )

        # 2. All 4 v3 rates landed where expected
        v = result.verdict
        assert v.fix_rate == 1.0                    # 1/1 fix new_passes
        assert v.regression_rate == 1.0             # 1/1 baseline new_passes
        # rubric: 2 new + 1 tie over 1 fix session = (1+1+0)/3 = +2/3 ≈ 0.667
        assert v.rubric_score == pytest.approx(2 / 3)
        assert v.binary_checks_pass_rate == 1.0     # all baseline×check pairs pass

        # 3. Atomic-mutation bookkeeping
        prop = result.propose_result
        assert prop is not None
        assert prop.action == "edit"
        assert len(prop.accepted_log) == 2

        # 4. Program parsed rubric + binary checks correctly
        prog = result.program_result
        assert prog is not None
        assert len(prog.rubric_axes) == 3
        assert prog.rubric_axes[0].name == "dietary_constraint_handling"
        assert len(prog.binary_checks) == 5
        assert prog.has_validation_schema is True

        # 5. LLM call count
        assert len(fake_llm.calls) == 9

        # 6. All artifacts on disk (no critic.md — v3 doesn't run a final critic)
        target_dir = tmp_path / "test_v3_e2e" / "find-restaurant"
        for name in (
            "program.md", "v_old.md", "v_new.md", "diff.txt",
            "propose_reasoning.md", "replay.md", "verdict.md",
        ):
            assert (target_dir / name).exists(), f"missing {name}"

        # 7. replay.md mentions all 4 v3 rates (catches accidental regressions
        # in the markdown renderer)
        replay_md = (target_dir / "replay.md").read_text(encoding="utf-8")
        for rate in ("fix_rate", "regression_rate",
                     "rubric_score", "binary_checks_pass_rate"):
            assert rate in replay_md, f"replay.md missing {rate}"
