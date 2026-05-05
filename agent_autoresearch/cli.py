"""`autoresearch` CLI — thin wrapper around `pipeline.run_pipeline()`.

Run with:

    autoresearch run --adapter <name> [--top-n N] [--strategy v1|v2|v3] ...

The CLI does three things and tries to do nothing else:

  1. Discover adapter implementations via entry-points
     (`agent_autoresearch.adapters` group). Third-party packages
     register their `Adapter` subclass there and become callable
     by name on this CLI.
  2. Wire up a `SkillIO` (default `FilesystemSkillIO`) and an
     `LLMProvider` (default Anthropic Sonnet) and pass them to
     `pipeline.run_pipeline`.
  3. Render live progress + a summary table via Rich.

Anything more interesting (sampling logic, verdict thresholds,
prompt templates) lives in the strategy code — not here. If you
find yourself adding strategy logic to the CLI, push it down.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_autoresearch import __version__
from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.skill_io import FilesystemSkillIO
from agent_autoresearch.pipeline import (
    PipelineRunResult,
    TargetRunResult,
    run_pipeline,
)
from agent_autoresearch.strategies.v1 import (
    DEFAULT_BASELINE_SAMPLE,
    DEFAULT_FIX_SAMPLE,
)


_ADAPTER_GROUP = "agent_autoresearch.adapters"

_LABEL_STYLE = {
    "ACCEPT":        "bold green",
    "HUMAN_REVIEW":  "bold yellow",
    "REJECT":        "bold red",
    "SKIP":          "dim",
}


# ── Adapter discovery ───────────────────────────────────────────────────────

def _discover_adapters() -> dict[str, type[Adapter]]:
    """Map adapter name → class via the `agent_autoresearch.adapters` group.

    Failures to load any one entry-point don't break discovery — we
    swallow the error and skip that adapter so a broken third-party
    package can't make `autoresearch run` unusable.
    """
    out: dict[str, type[Adapter]] = {}
    eps = entry_points()
    # Python 3.10+: `select(group=...)`; older fallback omitted (we require 3.11+).
    for ep in eps.select(group=_ADAPTER_GROUP):
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001 — we deliberately want to ignore here
            continue
        out[ep.name] = cls
    return out


def _resolve_adapter(name: str) -> type[Adapter]:
    """Look up an adapter by name, or raise UsageError listing options."""
    available = _discover_adapters()
    if name not in available:
        raise click.UsageError(
            f"Unknown adapter {name!r}. "
            f"Available: {', '.join(sorted(available)) or '(none registered)'}.\n"
            "Register one by adding it to the "
            f"`[project.entry-points.\"{_ADAPTER_GROUP}\"]` table in your "
            "package's pyproject.toml."
        )
    return available[name]


# ── CLI group ───────────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="autoresearch")
def cli() -> None:
    """Auto-improve agent skill prompts from your eval pipeline output."""


# ── `autoresearch adapters` ─────────────────────────────────────────────────

@cli.command("adapters")
def cmd_adapters() -> None:
    """List adapters registered via entry-points."""
    console = Console()
    found = _discover_adapters()
    if not found:
        console.print(
            "[yellow]No adapters registered.[/yellow]\n"
            "Adapters are discovered via the "
            f"`{_ADAPTER_GROUP}` entry-point group. See "
            "`docs/writing_an_adapter.md` for how to register one."
        )
        return

    table = Table(title="Registered adapters", show_header=True, header_style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("Class")
    table.add_column("Module", style="dim")
    for name, cls in sorted(found.items()):
        table.add_row(name, cls.__name__, cls.__module__)
    console.print(table)


# ── `autoresearch run` ──────────────────────────────────────────────────────

@cli.command("run")
@click.option(
    "--adapter", "adapter_name", required=True, metavar="<name>",
    help="Name of a registered adapter (see `autoresearch adapters`).",
)
@click.option(
    "--top-n", "top_n", default=3, show_default=True, type=int,
    help="How many top-ranked targets to run from the adapter.",
)
@click.option(
    "--fix-sample", "fix_sample", default=DEFAULT_FIX_SAMPLE, show_default=True,
    type=int,
    help="How many fix-target sessions to replay per target.",
)
@click.option(
    "--baseline-sample", "baseline_sample", default=DEFAULT_BASELINE_SAMPLE,
    show_default=True, type=int,
    help="How many regression baselines to replay per target.",
)
@click.option(
    "--outputs-root", "outputs_root", default="outputs", show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where each run's output folder is created.",
)
@click.option(
    "--skills-root", "skills_root", default="skills", show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where the FilesystemSkillIO looks for current SKILL.md files.",
)
@click.option(
    "--skill-path-template", "skill_path_template", default=None,
    metavar="<template>",
    help=(
        "Override the default `{root}/{name}/SKILL.md` layout. "
        "Use `{root}` and `{name}` placeholders."
    ),
)
@click.option(
    "--strategy", "strategy",
    type=click.Choice(["v1", "v2", "v3"], case_sensitive=False),
    default="v1", show_default=True,
    help="Strategy version to run. v1 is the original loop, v2 adds "
         "atomic-mutation propose, v3 adds rubric + binary checks.",
)
@click.option(
    "--llm-provider", "llm_provider",
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    default="anthropic", show_default=True,
    help=(
        "Which LLM provider to use. Each provider needs its own API key "
        "(ANTHROPIC_API_KEY or OPENAI_API_KEY)."
    ),
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Load adapter targets but make no LLM calls. Sanity-check your adapter.",
)
def cmd_run(
    adapter_name: str,
    top_n: int,
    fix_sample: int,
    baseline_sample: int,
    outputs_root: Path,
    skills_root: Path,
    skill_path_template: str | None,
    strategy: str,
    llm_provider: str,
    dry_run: bool,
) -> None:
    """Run the autoresearch pipeline against the named adapter."""
    console = Console()

    # 1. Resolve adapter class + instantiate
    adapter_cls = _resolve_adapter(adapter_name)
    try:
        adapter = adapter_cls()
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Failed to construct adapter[/red] [bold]{adapter_name}[/bold]: "
            f"{type(exc).__name__}: {exc}\n"
            "If your adapter requires init arguments, instantiate it in your own "
            "script and call `agent_autoresearch.pipeline.run_pipeline()` directly."
        )
        sys.exit(2)

    # 2. SkillIO (filesystem default; users wanting custom IO call run_pipeline directly)
    skill_io = FilesystemSkillIO(
        root=skills_root,
        path_template=skill_path_template,
    )

    # 3. Pre-flight the chosen provider's API key
    resolved_provider = llm_provider.lower()
    required_env_key = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }.get(resolved_provider)

    if not dry_run and required_env_key and not os.environ.get(required_env_key, "").strip():
        console.print(
            f"[red]{required_env_key} is not set.[/red] "
            f"The {resolved_provider!r} provider requires it. "
            "Either export it, switch providers with [bold]--llm-provider[/bold], "
            "or re-run with [bold]--dry-run[/bold] to skip live calls."
        )
        sys.exit(2)

    # Construct the provider explicitly so we can pass it down (rather
    # than letting `default_llm_provider()` pick from env at every call site).
    llm_instance = None
    if not dry_run:
        try:
            from agent_autoresearch.core.llm import default_llm_provider
            llm_instance = default_llm_provider(resolved_provider)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[red]Failed to initialize LLM provider {resolved_provider!r}:[/red] "
                f"{type(exc).__name__}: {exc}"
            )
            sys.exit(2)

    # 4. Header panel
    mode = "[yellow]dry-run[/yellow]" if dry_run else "[green]live[/green]"
    console.print(Panel.fit(
        f"adapter: [bold]{adapter_name}[/bold]  ·  "
        f"strategy: [bold]{strategy}[/bold]  ·  "
        f"llm: [bold]{resolved_provider}[/bold]  ·  "
        f"top-n: [bold]{top_n}[/bold]  ·  "
        f"mode: {mode}",
        title="autoresearch run",
        border_style="cyan",
    ))

    # 5. Set up live progress callbacks
    def _on_stage(target, stage_name: str) -> None:
        console.print(f"  [dim]·[/dim] [cyan]{target.skill_name}[/cyan] → {stage_name}")

    def _on_target_done(r: TargetRunResult) -> None:
        style = _LABEL_STYLE.get(r.verdict.label, "")
        console.print(
            f"  → [bold]{r.verdict.skill_name}[/bold] · "
            f"[{style}]{r.verdict.label}[/{style}]"
        )

    # 6. Run the pipeline (per-target failures isolated)
    try:
        result = run_pipeline(
            adapter,
            skill_io=skill_io,
            llm=llm_instance,            # explicit provider; None for dry-run
            top_n=top_n,
            fix_sample=fix_sample,
            baseline_sample=baseline_sample,
            outputs_root=outputs_root,
            dry_run=dry_run,
            raise_on_error=False,
            on_stage=None if dry_run else _on_stage,
            on_target_done=None if dry_run else _on_target_done,
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Pipeline error:[/red] {type(exc).__name__}: {exc}")
        sys.exit(1)

    # 7. Summary
    _print_summary(console, result)


# ── Summary table ───────────────────────────────────────────────────────────

def _print_summary(console: Console, result: PipelineRunResult) -> None:
    """Final post-run summary — one table + the path to summary.md."""
    if result.dry_run:
        table = Table(title=f"Dry-run · {len(result.targets)} target(s) loaded",
                      show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Skill", style="bold cyan")
        table.add_column("Evidence", justify="right")
        table.add_column("Fix sessions", justify="right")
        table.add_column("Baselines", justify="right")
        for i, t in enumerate(result.targets, 1):
            table.add_row(
                str(i), t.skill_name,
                str(len(t.evidence)),
                str(len(t.fix_session_ids)),
                str(len(t.regression_baseline_ids)),
            )
        console.print(table)
        console.print(
            f"[dim]No LLM calls made.[/dim]  "
            f"summary.md → [bold]{result.outputs_root / result.run_id}/summary.md[/bold]"
        )
        return

    table = Table(
        title=f"Run {result.run_id} · {result.elapsed_seconds:.1f}s",
        show_header=True, header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("Skill", style="bold cyan")
    table.add_column("Action")
    table.add_column("Critic")
    table.add_column("Fix", justify="right")
    table.add_column("Regr", justify="right")
    table.add_column("Verdict")

    for i, r in enumerate(result.target_results, 1):
        v = r.verdict
        style = _LABEL_STYLE.get(v.label, "")
        fix = f"{v.fix_target_score:.0%}" if v.fix_target_score is not None else "—"
        regr = f"{v.regression_score:.0%}" if v.regression_score is not None else "—"
        table.add_row(
            str(i), v.skill_name,
            v.propose_action or "—",
            v.critic_verdict or "—",
            fix, regr,
            f"[{style}]{v.label}[/{style}]",
        )
    console.print(table)

    console.print(
        f"[bold]ACCEPT[/bold]: {result.n_accept}  ·  "
        f"[bold yellow]HUMAN_REVIEW[/bold yellow]: {result.n_review}  ·  "
        f"[bold red]REJECT[/bold red]: {result.n_reject}  ·  "
        f"[dim]SKIP[/dim]: {result.n_skip}"
    )
    console.print(
        f"summary.md → [bold]{result.outputs_root / result.run_id}/summary.md[/bold]"
    )


if __name__ == "__main__":
    cli()
