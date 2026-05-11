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
from agent_autoresearch.strategies.registry import get_strategy
# Type-only imports from v1 — used as the canonical type hints in
# `TargetRunResult`. v2 and v3's result types are subclasses /
# structurally-compatible drop-ins, so this is fine.
from agent_autoresearch.strategies.v1 import (
    CriticResult,
    DEFAULT_BASELINE_SAMPLE,
    DEFAULT_FIX_SAMPLE,
    ProgramResult,
    ProposeResult,
    ReplayResult,
    Verdict,
)


# Default strategy when none is specified — preserves v1 backward compat.
DEFAULT_STRATEGY = "v1"


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
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    on_stage: StageHook | None = None,
    strategy: str = DEFAULT_STRATEGY,
) -> TargetRunResult:
    """Run all stages 4-8 for one target using the named strategy version.

    `strategy` picks which `agent_autoresearch.strategies.vN` module to
    use for build_program / propose / critic / replay / verdict.
    Default is v1 for backward compat.

    Skips early when:
      - target.skill_name is `UNATTRIBUTED` (sentinel from adapters)
      - propose returns 'skip' (jumps straight to verdict = SKIP)
    """
    llm = llm or default_llm_provider()
    strategy_mod = get_strategy(strategy)
    version = getattr(strategy_mod, "STRATEGY_VERSION", "v1")

    def _hook(stage: str) -> None:
        if on_stage:
            on_stage(target, stage)

    # 0. Sanity: unattributed sentinel
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
    prog = strategy_mod.build_program(target, current_skill_md=current_skill_md, llm=llm)
    write_artifact(target.skill_name, "program.md", prog.program_md,
                   run_id=run_id, outputs_root=outputs_root)

    # Step 5 — propose. v1 has a single-shot propose; v2/v3 use the
    # atomic-mutation loop with injected validators that wrap the
    # version's own critic/replay.
    _hook("propose")
    if version == "v1":
        prop = strategy_mod.propose(
            target.skill_name,
            current_skill_md=current_skill_md,
            program_md=prog.program_md,
            llm=llm,
        )
    else:
        validators, _ = _build_atomic_validators(
            strategy_mod=strategy_mod,
            target=target,
            current_skill_md=current_skill_md,
            program_result=prog,
            conversations=conversations,
            fix_sample=fix_sample,
            baseline_sample=baseline_sample,
            llm=llm,
        )
        # v2 dropped the per-attempt mini-replay gate; only critic gates
        # each atomic attempt. v3 still uses replay_per_attempt.
        if version == "v2":
            validators = {k: v for k, v in validators.items()
                          if k != "replay_per_attempt"}
        prop = strategy_mod.propose(
            target,
            current_skill_md=current_skill_md,
            program_md=prog.program_md,
            conversations=conversations,
            **validators,
            llm=llm,
        )

    # Skip path — short-circuit (same regardless of strategy)
    if prop.action != "edit" or not prop.new_skill_md:
        write_artifact(
            target.skill_name, "skip.md",
            "# Skipped — no edit proposed\n\n"
            f"## Reasoning\n\n{prop.reasoning}\n\n"
            f"## Strategy that was generated\n\n{prog.program_md}",
            run_id=run_id, outputs_root=outputs_root,
        )
        v = strategy_mod.compute_verdict(
            skill_name=target.skill_name, propose_result=prop,
            critic_result=None, replay_result=None,
        )
        write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                       run_id=run_id, outputs_root=outputs_root)
        return TargetRunResult(target=target, verdict=v,
                               program_result=prog, propose_result=prop)

    # Edit path — write v_new + diff
    new_path = skill_io.write_version(
        target.skill_name, prop.new_skill_md,
        run_id=run_id, outputs_root=outputs_root,
    )
    write_artifact(target.skill_name, "propose_reasoning.md", prop.reasoning,
                   run_id=run_id, outputs_root=outputs_root)
    diff_text = (new_path.parent / "diff.txt").read_text(encoding="utf-8")

    # Step 6 — critic (canonical pass for verdict)
    _hook("critic")
    crit = strategy_mod.critic(
        target.skill_name,
        program_md=prog.program_md,
        diff_text=diff_text,
        v_old_md=current_skill_md,
        v_new_md=prop.new_skill_md,
        llm=llm,
    )
    write_artifact(target.skill_name, "critic.md", crit.to_markdown(),
                   run_id=run_id, outputs_root=outputs_root)

    # Step 7 — replay (always runs for the edit path; the only way to
    # skip this is to skip the whole target via propose action='skip',
    # which short-circuited above)
    _hook("replay")
    replay_kwargs: dict = dict(
        new_skill_md=prop.new_skill_md,
        program_md=prog.program_md,
        conversations=conversations,
        fix_sample=fix_sample,
        baseline_sample=baseline_sample,
        llm=llm,
    )
    # v3 needs the rubric + checks for structured judging
    if version == "v3":
        replay_kwargs["rubric_axes"] = prog.rubric_axes
        replay_kwargs["binary_checks"] = prog.binary_checks
    rep: ReplayResult = strategy_mod.soft_replay(target, **replay_kwargs)
    write_artifact(target.skill_name, "replay.md", rep.to_markdown(),
                   run_id=run_id, outputs_root=outputs_root)

    # Step 8 — verdict
    _hook("verdict")
    v = strategy_mod.compute_verdict(
        skill_name=target.skill_name, propose_result=prop,
        critic_result=crit, replay_result=rep,
    )
    write_artifact(target.skill_name, "verdict.md", v.to_markdown(),
                   run_id=run_id, outputs_root=outputs_root)

    return TargetRunResult(
        target=target, verdict=v,
        program_result=prog, propose_result=prop,
        critic_result=crit, replay_result=rep,
    )


# ── Atomic-mutation validators (used by v2/v3 propose) ──────────────────────

def _build_atomic_validators(
    *,
    strategy_mod,
    target: Target,
    current_skill_md: str,
    program_result,
    conversations: dict,
    fix_sample: int,
    baseline_sample: int,
    llm: LLMProvider,
):
    """Build the four validator callables that v2/v3 propose() expects.

    Validators wrap the strategy's own `critic()` and `soft_replay()`
    so the atomic-mutation loop's gating decisions use the canonical
    LLM stages — same prompts, same parsing, same thresholds.

    Returns `(validators_dict, captures_dict)` where `captures_dict`
    holds mutable boxes the validators write into. The orchestrator
    doesn't currently read them (the post-propose canonical critic
    + replay calls do the verdict-level work), but they're available
    for future optimisation.
    """
    import difflib

    captures: dict = {"final_critic_result": None, "final_replay_result": None}
    version = getattr(strategy_mod, "STRATEGY_VERSION", "v1")

    def _diff(old: str, new: str) -> str:
        return "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="v_old", tofile="v_new", lineterm="",
        ))

    def _full_replay_kwargs(candidate_md: str) -> dict:
        kw: dict = dict(
            new_skill_md=candidate_md,
            program_md=program_result.program_md,
            conversations=conversations,
            fix_sample=fix_sample,
            baseline_sample=baseline_sample,
            llm=llm,
        )
        if version == "v3":
            kw["rubric_axes"] = program_result.rubric_axes
            kw["binary_checks"] = program_result.binary_checks
        return kw

    def _check_replay_thresholds(rep) -> tuple[bool, str]:
        """Compare the strategy's full replay result against its acceptance
        thresholds. Used by the per-attempt gate and the final pass.
        """
        thresholds = strategy_mod.THRESHOLDS
        if version == "v3":
            ok = (
                rep.fix_rate >= thresholds["fix_rate_min"]
                and rep.regression_rate >= thresholds["regression_rate_min"]
                and rep.rubric_improvement_rate >= thresholds["rubric_improvement_min"]
                and rep.binary_checks_pass_rate >= thresholds["binary_checks_min"]
            )
            reason = (
                f"fix={rep.fix_rate:.0%} regr={rep.regression_rate:.0%} "
                f"rubric={rep.rubric_improvement_rate:.0%} "
                f"checks={rep.binary_checks_pass_rate:.0%}"
            )
        else:
            # v2: fix_target_min is a strict-> floor (any improvement counts)
            ok = (
                rep.fix_target_score > thresholds["fix_target_min"]
                and rep.regression_score >= thresholds["regression_min"]
            )
            reason = (
                f"fix={rep.fix_target_score:.0%} "
                f"regr={rep.regression_score:.0%}"
            )
        return ok, reason

    def critic_per_attempt(candidate_md: str, current_md: str, _ev) -> tuple[bool, str]:
        """Real critic call against the small per-iteration diff."""
        diff = _diff(current_md, candidate_md)
        result = strategy_mod.critic(
            target.skill_name,
            program_md=program_result.program_md,
            diff_text=diff,
            v_old_md=current_md,
            v_new_md=candidate_md,
            llm=llm,
        )
        reason = ", ".join(result.concerns) or result.reasoning
        return result.approves, reason

    def replay_per_attempt(candidate_md: str, evidence, conv) -> tuple[bool, str]:
        """Lightweight per-iteration replay — single fix-target session.

        Builds a one-session mini-target, runs the strategy's replay
        with sample=1, and accepts only if the new wins outright.
        """
        if conv is None:
            return True, "(no transcript for evidence session — can't replay)"
        sid = conv.session_id
        from dataclasses import replace as dc_replace
        mini_target = dc_replace(
            target, fix_session_ids=[sid], regression_baseline_ids=[],
        )
        kw: dict = dict(
            new_skill_md=candidate_md,
            program_md=program_result.program_md,
            conversations={sid: conv},
            fix_sample=1,
            baseline_sample=0,
            llm=llm,
        )
        if version == "v3":
            kw["rubric_axes"] = program_result.rubric_axes
            kw["binary_checks"] = program_result.binary_checks
        rep = strategy_mod.soft_replay(mini_target, **kw)
        # New must win outright on this single session
        ok = rep.fix_target_score >= 1.0
        return ok, f"fix_target_score={rep.fix_target_score:.0%}"

    def final_critic(candidate_md: str, current_md: str, _ev) -> tuple[bool, str]:
        """Critic against the full cumulative diff (original → final)."""
        diff = _diff(current_skill_md, candidate_md)
        result = strategy_mod.critic(
            target.skill_name,
            program_md=program_result.program_md,
            diff_text=diff,
            v_old_md=current_skill_md,
            v_new_md=candidate_md,
            llm=llm,
        )
        captures["final_critic_result"] = result
        reason = ", ".join(result.concerns) or result.reasoning
        return result.approves, reason

    def final_replay(candidate_md: str, _target, _convs) -> tuple[bool, str]:
        """Full replay over the configured fix + baseline sample."""
        rep = strategy_mod.soft_replay(target, **_full_replay_kwargs(candidate_md))
        captures["final_replay_result"] = rep
        return _check_replay_thresholds(rep)

    return {
        "critic_per_attempt": critic_per_attempt,
        "replay_per_attempt": replay_per_attempt,
        "final_critic":       final_critic,
        "final_replay":       final_replay,
    }, captures


# ── Pipeline-level run ──────────────────────────────────────────────────────

def run_pipeline(
    adapter: Adapter,
    *,
    skill_io: SkillIO | None = None,
    llm: LLMProvider | None = None,
    top_n: int = 3,
    fix_sample: int = DEFAULT_FIX_SAMPLE,
    baseline_sample: int = DEFAULT_BASELINE_SAMPLE,
    outputs_root: Path | None = None,
    dry_run: bool = False,
    raise_on_error: bool = False,
    on_stage: StageHook | None = None,
    on_target_done: Callable[[TargetRunResult], None] | None = None,
    strategy: str = DEFAULT_STRATEGY,
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
                fix_sample=fix_sample, baseline_sample=baseline_sample,
                on_stage=on_stage,
                strategy=strategy,
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
