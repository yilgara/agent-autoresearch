"""Pipeline orchestrator — wires the stages of one strategy version together.

Two public entry points:

  - `run_target(...)`    — run all stages (4-8) for ONE target, write its
                           output folder, return a `TargetRunResult`
  - `run_pipeline(...)`  — load top-N targets from an adapter, run each,
                           write a top-level summary.md, return a
                           `PipelineRunResult`

This module is silent (no print). The CLI in `agent_autoresearch.cli`
wraps these for interactive output. Library callers can use these
directly for programmatic runs.

Failure isolation: an exception in one target is caught, recorded in
the result, and the next target continues. Set
`raise_on_error=True` to disable this if you're debugging.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import Target
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider
from agent_autoresearch.core.skill_io import (
    FilesystemSkillIO,
    SkillIO,
    UNATTRIBUTED,
    make_run_id,
    write_artifact,
)
from agent_autoresearch.strategies.v1 import (
    CriticResult,
    DEFAULT_BASELINE_SAMPLE,
    DEFAULT_FIX_SAMPLE,
    ProgramResult,
    ProposeResult,
    ReplayResult,
    Verdict,
    build_program,
    compute_verdict,
    critic,
    propose,
    soft_replay,
)


# ── Result types ────────────────────────────────────────────────────────────

@dataclass
class TargetRunResult:
    """Per-target outcome of one run.

    Carries the verdict + every intermediate result. `error` is set
    when an exception was caught — the verdict in that case is REJECT
    with the traceback in `reason`.
    """
    target: Target
    verdict: Verdict
    program_result: ProgramResult | None = None
    propose_result: ProposeResult | None = None
    critic_result: CriticResult | None = None
    replay_result: ReplayResult | None = None
    error: str | None = None


@dataclass
class PipelineRunResult:
    """Top-level outcome of one `autoresearch run` invocation."""
    run_id: str
    repo_label: str                  # display label, e.g. adapter name
    targets: list[Target]
    target_results: list[TargetRunResult]
    elapsed_seconds: float
    outputs_root: Path
    dry_run: bool = False

    @property
    def verdicts_by_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.target_results:
            counts[r.verdict.label] = counts.get(r.verdict.label, 0) + 1
        return counts

    @property
    def n_accept(self) -> int: return self.verdicts_by_label.get("ACCEPT", 0)

    @property
    def n_review(self) -> int: return self.verdicts_by_label.get("HUMAN_REVIEW", 0)

    @property
    def n_reject(self) -> int: return self.verdicts_by_label.get("REJECT", 0)

    @property
    def n_skip(self) -> int: return self.verdicts_by_label.get("SKIP", 0)


# Optional progress callback signature: (target, stage_name) → None
StageHook = Callable[[Target, str], None]


# ── Per-target run ──────────────────────────────────────────────────────────

def run_target(
    target: Target,
    conversations: dict[str, "Conversation"],  # noqa: F821 — forward ref ok
    *,
    skill_io: SkillIO,
    run_id: str,
    outputs_root: Path,
    llm: LLMProvider | None = None,
    do_validate: bool = True,
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    on_stage: StageHook | None = None,
) -> TargetRunResult:
    """Run all stages 4-8 for one target. Writes the target's output folder
    via `skill_io.write_version()` + `write_artifact()`. Returns the result
    object.

    Skips early when:
      - target.skill_name is `UNATTRIBUTED` (sentinel from adapters that
        couldn't tag a session to any specific skill)
      - propose returns 'skip' (jumps straight to verdict = SKIP)
    """
    llm = llm or default_llm_provider()

    def _hook(stage: str) -> None:
        if on_stage:
            on_stage(target, stage)

    # 0. Sanity: unattributed sentinel — nothing to load, nothing to edit
    if target.skill_name == UNATTRIBUTED:
        v = Verdict(
            skill_name=target.skill_name, label="SKIP",
            reason="Target has no attributable skill (UNATTRIBUTED sentinel).",
        )
        write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                       run_id=run_id, outputs_root=outputs_root)
        return TargetRunResult(target=target, verdict=v)

    # Load the current SKILL.md
    try:
        current_skill_md = skill_io.load(target.skill_name)
    except FileNotFoundError as exc:
        v = Verdict(
            skill_name=target.skill_name, label="REJECT",
            reason=f"Skill not found by SkillIO: {exc}",
        )
        write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                       run_id=run_id, outputs_root=outputs_root)
        return TargetRunResult(target=target, verdict=v,
                               error=f"FileNotFoundError: {exc}")

    # Step 4 — strategy doc
    _hook("program")
    prog = build_program(target, current_skill_md=current_skill_md, llm=llm)
    write_artifact(target.skill_name, "program.md", prog.program_md,
                   run_id=run_id, outputs_root=outputs_root)

    # Step 5 — propose
    _hook("propose")
    prop = propose(target.skill_name,
                   current_skill_md=current_skill_md,
                   program_md=prog.program_md, llm=llm)

    # Skip path — short-circuit
    if prop.action != "edit" or not prop.new_skill_md:
        write_artifact(
            target.skill_name, "skip.md",
            "# Skipped — no edit proposed\n\n"
            f"## Reasoning\n\n{prop.reasoning}\n\n"
            f"## Strategy that was generated\n\n{prog.program_md}",
            run_id=run_id, outputs_root=outputs_root,
        )
        v = compute_verdict(skill_name=target.skill_name,
                            propose_result=prop,
                            critic_result=None, replay_result=None)
        write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                       run_id=run_id, outputs_root=outputs_root)
        return TargetRunResult(target=target, verdict=v,
                               program_result=prog, propose_result=prop)

    # Edit path — write the proposed version + reasoning
    new_path = skill_io.write_version(
        target.skill_name, prop.new_skill_md,
        run_id=run_id, outputs_root=outputs_root,
    )
    write_artifact(target.skill_name, "propose_reasoning.md", prop.reasoning,
                   run_id=run_id, outputs_root=outputs_root)
    diff_text = (new_path.parent / "diff.txt").read_text(encoding="utf-8")

    # Step 6 — critic
    _hook("critic")
    crit = critic(
        target.skill_name,
        program_md=prog.program_md,
        diff_text=diff_text,
        v_old_md=current_skill_md,
        v_new_md=prop.new_skill_md,
        llm=llm,
    )
    write_artifact(target.skill_name, "critic.md", crit.to_markdown(),
                   run_id=run_id, outputs_root=outputs_root)

    # Step 7 — replay (optional)
    rep: ReplayResult | None = None
    if do_validate:
        _hook("replay")
        rep = soft_replay(
            target,
            new_skill_md=prop.new_skill_md,
            program_md=prog.program_md,
            conversations=conversations,
            fix_sample=fix_sample,
            baseline_sample=baseline_sample,
            llm=llm,
        )
        write_artifact(target.skill_name, "replay.md", rep.to_markdown(),
                       run_id=run_id, outputs_root=outputs_root)

    # Step 8 — verdict
    _hook("verdict")
    v = compute_verdict(skill_name=target.skill_name,
                        propose_result=prop,
                        critic_result=crit, replay_result=rep)
    write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                   run_id=run_id, outputs_root=outputs_root)

    return TargetRunResult(
        target=target, verdict=v,
        program_result=prog, propose_result=prop,
        critic_result=crit, replay_result=rep,
    )


# ── Pipeline-level run ──────────────────────────────────────────────────────

def run_pipeline(
    adapter: Adapter,
    *,
    skill_io: SkillIO | None = None,
    llm: LLMProvider | None = None,
    top_n: int = 3,
    do_validate: bool = True,
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    outputs_root: Path | None = None,
    dry_run: bool = False,
    raise_on_error: bool = False,
    on_stage: StageHook | None = None,
    on_target_done: Callable[[TargetRunResult], None] | None = None,
) -> PipelineRunResult:
    """Run the full pipeline against one adapter's targets.

    Loads targets + conversations via the adapter, picks the top-N
    (skipping UNATTRIBUTED), runs each through `run_target`, writes
    a top-level `summary.md`, and returns a `PipelineRunResult`.

    `dry_run=True` skips all LLM calls — just loads + builds targets,
    writes a parse-only summary. Useful for sanity-checking adapters
    without burning API tokens.
    """
    t0 = time.time()
    skill_io = skill_io or FilesystemSkillIO()
    outputs_root = outputs_root or Path("outputs")
    # LLM provider is lazy — dry-run never needs one, and constructing
    # the default provider raises if ANTHROPIC_API_KEY is unset.

    run_id = make_run_id()
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load targets — pick top-N excluding the unattributed sentinel
    all_targets = adapter.load_targets()
    targets: list[Target] = []
    for t in all_targets:
        if t.skill_name == UNATTRIBUTED:
            continue
        targets.append(t)
        if len(targets) >= top_n:
            break

    # Dry-run path — write a stub summary and return
    if dry_run:
        _write_summary(
            run_id=run_id, repo_label=adapter.name,
            target_results=[], targets=targets, dry_run=True,
            elapsed=time.time() - t0, outputs_root=outputs_root,
        )
        return PipelineRunResult(
            run_id=run_id, repo_label=adapter.name,
            targets=targets, target_results=[],
            elapsed_seconds=time.time() - t0,
            outputs_root=outputs_root, dry_run=True,
        )

    # Live path — instantiate the LLM provider now, then load
    # conversations once and reuse them across targets.
    llm = llm or default_llm_provider()
    conv_list = adapter.load_conversations()
    conversations = {c.session_id: c for c in conv_list}

    # Run each target
    target_results: list[TargetRunResult] = []
    for t in targets:
        try:
            r = run_target(
                t, conversations,
                skill_io=skill_io, llm=llm,
                run_id=run_id, outputs_root=outputs_root,
                do_validate=do_validate,
                fix_sample=fix_sample, baseline_sample=baseline_sample,
                on_stage=on_stage,
            )
        except Exception as exc:
            if raise_on_error:
                raise
            r = TargetRunResult(
                target=t,
                verdict=Verdict(
                    skill_name=t.skill_name, label="REJECT",
                    reason=f"Pipeline error: {type(exc).__name__}: {exc}",
                ),
                error=traceback.format_exc(),
            )
        target_results.append(r)
        if on_target_done:
            on_target_done(r)

    elapsed = time.time() - t0

    _write_summary(
        run_id=run_id, repo_label=adapter.name,
        target_results=target_results, targets=targets, dry_run=False,
        elapsed=elapsed, outputs_root=outputs_root,
    )

    return PipelineRunResult(
        run_id=run_id, repo_label=adapter.name,
        targets=targets, target_results=target_results,
        elapsed_seconds=elapsed, outputs_root=outputs_root,
        dry_run=False,
    )


# ── summary.md rendering ────────────────────────────────────────────────────

_LABEL_BADGE = {
    "ACCEPT":         "🟢 ACCEPT",
    "HUMAN_REVIEW":   "🟡 HUMAN_REVIEW",
    "REJECT":         "🔴 REJECT",
    "SKIP":           "⚪ SKIP",
    "NO_VALIDATION":  "⚪ NO_VALIDATION",
}


def _write_summary(
    *,
    run_id: str,
    repo_label: str,
    targets: list[Target],
    target_results: list[TargetRunResult],
    dry_run: bool,
    elapsed: float,
    outputs_root: Path,
) -> Path:
    """Write a top-level `summary.md` across all targets in a run."""
    lines: list[str] = [
        f"# Autoresearch run · `{run_id}`",
        "",
        f"**Adapter:** `{repo_label}`  ·  **Targets:** {len(targets)}  ·  "
        f"**Elapsed:** {elapsed:.1f}s  ·  "
        f"**Mode:** {'dry-run' if dry_run else 'live'}",
        "",
    ]

    if dry_run:
        lines += [
            "## Targets (no LLM calls made)",
            "",
            "| # | Skill | Evidence | Fix sessions | Baselines |",
            "|---|---|---:|---:|---:|",
        ]
        for i, t in enumerate(targets, 1):
            lines.append(
                f"| {i} | `{t.skill_name}` | {len(t.evidence)} | "
                f"{len(t.fix_session_ids)} | {len(t.regression_baseline_ids)} |"
            )
        lines.append("")
        path = outputs_root / run_id / "summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # Live mode — full per-target table + reasons
    by_label: dict[str, int] = {}
    for r in target_results:
        by_label[r.verdict.label] = by_label.get(r.verdict.label, 0) + 1

    lines += [
        "## Outcomes",
        "",
        "  ·  ".join(
            f"**{_LABEL_BADGE.get(k, k)}**: {n}"
            for k, n in sorted(by_label.items())
        ) or "_(no targets ran)_",
        "",
        "## Per-target verdicts",
        "",
        "| # | Skill | Action | Critic | Fix score | Regr score | Verdict |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for i, r in enumerate(target_results, 1):
        v = r.verdict
        fix = (f"{v.fix_target_score:.0%}"
               if v.fix_target_score is not None else "—")
        regr = (f"{v.regression_score:.0%}"
                if v.regression_score is not None else "—")
        lines.append(
            f"| {i} | `{v.skill_name}` | {v.propose_action or '—'} | "
            f"{v.critic_verdict or '—'} | {fix} | {regr} | "
            f"{_LABEL_BADGE.get(v.label, v.label)} |"
        )
    lines.append("")

    lines += ["## Reasons", ""]
    for r in target_results:
        lines.append(f"### `{r.verdict.skill_name}` — "
                     f"{_LABEL_BADGE.get(r.verdict.label, r.verdict.label)}")
        lines.append("")
        lines.append(r.verdict.reason)
        lines.append("")

    lines += [
        "## Output folders",
        "",
        *[f"- `{outputs_root}/{run_id}/{r.verdict.skill_name}/`"
          for r in target_results],
        "",
    ]

    path = outputs_root / run_id / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
