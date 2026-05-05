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

Same as v2 e2e (19 calls):

```
 1. build_program       (v3 — emits rubric + binary checks)
 2-5.  ev #1 atomic loop
 6-9.  ev #2 atomic loop
10.    final_critic
11-14. final_replay  (1 fix + 1 baseline)
15.    canonical critic
16-19. canonical soft_replay
```

The judge calls (rounds 5/9/12/14/17/19) all use v3's 3-signal output
schema; everything else is identical to v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_autoresearch.adapters.synthetic import SyntheticAdapter
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


def _judge_v3(*, winner: str, new_scores: tuple, old_scores: tuple,
              all_pass: bool = True) -> str:
    """Build a v3 judge response with all three signals.

    `new_scores` and `old_scores` are 3-tuples of int (one per axis).
    `all_pass` controls whether all 5 binary checks pass.
    """
    axis_names = ["dietary_constraint_handling", "query_specificity", "result_grounding"]
    rubric_xml = "\n".join(
        f"  <axis><name>{name}</name><new>{n}</new><old>{o}</old></axis>"
        for name, n, o in zip(axis_names, new_scores, old_scores)
    )
    check_result = "pass" if all_pass else "fail"
    checks_xml = "\n".join(
        f"  <check><id>{i}</id><result>{check_result}</result></check>"
        for i in range(1, 6)
    )
    return (
        f"<winner>{winner}</winner>\n"
        f"<rubric>\n{rubric_xml}\n</rubric>\n"
        f"<checks>\n{checks_xml}\n</checks>\n"
        "<reasoning>Synthesized v3 judge output for tests.</reasoning>"
    )


# Fix-session judge: NEW wins, rubric improves, all checks pass
_JUDGE_FIX = _judge_v3(
    winner="new",
    new_scores=(3, 3, 2),
    old_scores=(1, 2, 2),
    all_pass=True,
)

# Baseline-session judge: tie, rubric unchanged, all checks pass
_JUDGE_BASELINE = _judge_v3(
    winner="tie",
    new_scores=(2, 2, 2),
    old_scores=(2, 2, 2),
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
            _RESPONDER,                        # 4. replay_per_attempt responder
            _JUDGE_FIX,                        # 5. replay_per_attempt judge
            # evidence #2 atomic loop
            _propose_edit("ev2"),              # 6.
            _CRITIC_APPROVE,                   # 7.
            _RESPONDER,                        # 8.
            _JUDGE_FIX,                        # 9.
            # final pass inside propose
            _CRITIC_APPROVE,                   # 10. final_critic
            _RESPONDER,                        # 11. final_replay fix responder
            _JUDGE_FIX,                        # 12. final_replay fix judge
            _RESPONDER,                        # 13. final_replay baseline responder
            _JUDGE_BASELINE,                   # 14. final_replay baseline judge
            # canonical pass after propose returns
            _CRITIC_APPROVE,                   # 15. canonical critic
            _RESPONDER,                        # 16. canonical replay fix responder
            _JUDGE_FIX,                        # 17. canonical replay fix judge
            _RESPONDER,                        # 18. canonical replay baseline responder
            _JUDGE_BASELINE,                   # 19. canonical replay baseline judge
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
        assert v.fix_rate == 1.0                    # 1/1 fix new wins
        assert v.regression_rate == 1.0             # 1/1 baseline tie
        # rubric: fix improved (new>old) + baseline non-regressed (new>=old) = 2/2
        assert v.rubric_improvement_rate == 1.0
        assert v.binary_checks_pass_rate == 1.0     # all checks pass on both

        # 3. Atomic-mutation bookkeeping
        prop = result.propose_result
        assert prop is not None
        assert prop.action == "edit"
        assert len(prop.accepted_log) == 2
        assert prop.combined_check_passed is True
        assert prop.rolled_back_steps == 0

        # 4. Program parsed rubric + binary checks correctly
        prog = result.program_result
        assert prog is not None
        assert len(prog.rubric_axes) == 3
        assert prog.rubric_axes[0].name == "dietary_constraint_handling"
        assert len(prog.binary_checks) == 5
        assert prog.has_validation_schema is True

        # 5. LLM call count
        assert len(fake_llm.calls) == 19

        # 6. All artifacts on disk
        target_dir = tmp_path / "test_v3_e2e" / "find-restaurant"
        for name in (
            "program.md", "v_old.md", "v_new.md", "diff.txt",
            "propose_reasoning.md", "critic.md", "replay.md", "verdict.md",
        ):
            assert (target_dir / name).exists(), f"missing {name}"

        # 7. replay.md mentions all 4 v3 rates (catches accidental regressions
        # in the markdown renderer)
        replay_md = (target_dir / "replay.md").read_text(encoding="utf-8")
        for rate in ("fix_rate", "regression_rate",
                     "rubric_improvement_rate", "binary_checks_pass_rate"):
            assert rate in replay_md, f"replay.md missing {rate}"
