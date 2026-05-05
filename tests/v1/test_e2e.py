"""End-to-end pipeline test: synthetic adapter + example skills + FakeLLM.

This is the only test in the suite that exercises the **full** 8-step
pipeline (program → propose → critic → responder → judge → replay → verdict)
in one shot. Every other test stubs out part of the chain.

Setup:
  - Adapter:  `SyntheticAdapter` (hardcoded fixtures)
  - Skills:   `examples/synthetic-skills/<name>/SKILL.md`
  - LLM:      `FakeLLM` with canned responses for all 7 calls

Asserts the full output folder structure on disk + the propagated
verdict label, which is what users see at the end of a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_autoresearch.adapters.synthetic import SyntheticAdapter
from agent_autoresearch.core.skill_io import FilesystemSkillIO
from agent_autoresearch.pipeline import run_pipeline


# ── Path to the example skills bundled in the repo ──────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]   # tests/v1/ → repo root
EXAMPLE_SKILLS = REPO_ROOT / "examples" / "synthetic-skills"


# ── Canned LLM responses ────────────────────────────────────────────────────
#
# Order matters — each call pops the next response. With top_n=1,
# fix_sample=1, baseline_sample=1, the pipeline issues 7 LLM calls per
# target:
#   1. program           → strategy markdown
#   2. propose           → <action>edit</action> + <new_skill_md>...
#   3. critic            → <verdict>APPROVE</verdict> + <reasoning>
#   4. responder (fix)   → <tool_plan> + <reply>
#   5. judge (fix)       → <winner>new</winner>
#   6. responder (base)  → <tool_plan> + <reply>
#   7. judge (base)      → <winner>tie</winner>

_PROGRAM_MD = """\
# Strategy: improve find-restaurant

The skill currently ignores explicit dietary constraints and sometimes
refuses lookups outright. Two evidence categories:

- ignored_constraint (1 example)
- refused_reasonable_request (1 example)

## Proposed direction

Add explicit guidance: never refuse a restaurant lookup, always pass
user-stated dietary preferences to `search_restaurants` as the filter
argument.
"""

_PROPOSE_RESPONSE = """\
<action>edit</action>
<reasoning>Tighten the dietary-filter rule and explicitly forbid refusing lookups.</reasoning>
<new_skill_md>
# find-restaurant

You are helping the user find a restaurant.

## Tools

- `search_restaurants(q: str, filter?: str)` — returns matching restaurants
  by free-text query, with optional filter tags (`vegan`, `vegetarian`,
  `gluten-free`, etc.)

## Guidelines

1. **Always call `search_restaurants`** for restaurant lookups. Never
   refuse a reasonable request by suggesting external sites like Yelp.
2. When the user states a dietary preference, **always** pass it as the
   `filter` argument — this is non-negotiable.
3. Return the top match by name. Don't fabricate restaurants you didn't
   see in the tool output.
</new_skill_md>
"""

_CRITIC_RESPONSE = """\
<verdict>APPROVE</verdict>
<reasoning>The diff is targeted, anchored to the failure modes in program.md, and doesn't over-edit.</reasoning>
<concerns></concerns>
"""

_RESPONDER_FIX = """\
<tool_plan>Call search_restaurants(q="downtown", filter="vegan").</tool_plan>
<reply>Found Green Leaf — a vegan-friendly spot downtown. Want me to book it?</reply>
"""

_JUDGE_FIX_NEW_WINS = """\
<winner>new</winner>
<reasoning>The new reply respects the vegan filter; the old one ignored it.</reasoning>
"""

_RESPONDER_BASELINE = """\
<tool_plan>Same approach as the original — search with the vegan filter.</tool_plan>
<reply>Found Green Leaf — vegan-friendly, downtown.</reply>
"""

_JUDGE_BASELINE_TIE = """\
<winner>tie</winner>
<reasoning>Both replies handle the user's request equivalently well.</reasoning>
"""


# ── End-to-end test ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not EXAMPLE_SKILLS.exists(),
    reason="examples/synthetic-skills/ not found — option A not shipped",
)
class TestEndToEndSynthetic:
    def test_full_pipeline_produces_accept(self, tmp_path, fake_llm):
        """One full happy-path run: program → edit → APPROVE → 1 win + 1 tie → ACCEPT."""
        # Queue all 7 canned responses in order
        for resp in [
            _PROGRAM_MD,
            _PROPOSE_RESPONSE,
            _CRITIC_RESPONSE,
            _RESPONDER_FIX,
            _JUDGE_FIX_NEW_WINS,
            _RESPONDER_BASELINE,
            _JUDGE_BASELINE_TIE,
        ]:
            fake_llm.push(resp)

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)

        result = run_pipeline(
            adapter,
            skill_io=skill_io,
            llm=fake_llm,
            outputs_root=tmp_path,
            top_n=1,
            fix_sample=1,
            baseline_sample=1,
        )

        # 1. Verdict propagated correctly
        assert len(result.target_results) == 1
        verdict = result.target_results[0].verdict
        assert verdict.label == "ACCEPT", (
            f"Expected ACCEPT, got {verdict.label}: {verdict.reason}"
        )
        assert verdict.skill_name == "find-restaurant"
        assert verdict.fix_target_score == 1.0      # 1/1 win
        assert verdict.regression_score == 1.0      # tie counts as safe

        # 2. Aggregated counts
        assert result.n_accept == 1
        assert result.n_reject == 0

        # 3. All 7 LLM calls were made (no extras, no skips)
        assert len(fake_llm.calls) == 7

        # 4. Output folder contains every expected artifact
        target_dir = tmp_path / result.run_id / "find-restaurant"
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

        # 5. v_new.md actually contains the proposed edit
        v_new = (target_dir / "v_new.md").read_text(encoding="utf-8")
        assert "non-negotiable" in v_new

        # 6. summary.md was written and references the target
        summary = (tmp_path / result.run_id / "summary.md").read_text(encoding="utf-8")
        assert "find-restaurant" in summary
        assert "ACCEPT" in summary

    def test_propose_skip_short_circuits_full_pipeline(self, tmp_path, fake_llm):
        """If propose returns skip, the run should write program.md + skip.md +
        verdict.md and stop — no critic.md, no replay.md."""
        fake_llm.push(_PROGRAM_MD)
        fake_llm.push(
            "<action>skip</action>\n"
            "<reasoning>Evidence too thin to commit to an edit yet.</reasoning>"
        )

        adapter = SyntheticAdapter()
        skill_io = FilesystemSkillIO(root=EXAMPLE_SKILLS)

        result = run_pipeline(
            adapter,
            skill_io=skill_io,
            llm=fake_llm,
            outputs_root=tmp_path,
            top_n=1,
            fix_sample=1,
            baseline_sample=1,
        )

        assert result.target_results[0].verdict.label == "SKIP"
        assert len(fake_llm.calls) == 2   # only program + propose

        target_dir = tmp_path / result.run_id / "find-restaurant"
        assert (target_dir / "program.md").exists()
        assert (target_dir / "skip.md").exists()
        assert (target_dir / "verdict.md").exists()
        assert not (target_dir / "critic.md").exists()
        assert not (target_dir / "replay.md").exists()
