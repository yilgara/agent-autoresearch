"""Strategy implementations — versioned for easy comparison.

Each subdirectory is one complete implementation of the autoresearch
loop's LLM-driven stages. Different versions can change ANY stage
(program, propose, critic, responder, judge, verdict) — that's why
they're versioned together rather than mixed at the file level.

Versions currently in this package:

  - **v1** — the original loop. One bundled edit per round, freeform
    LLM judgment for both critic and judge, fix_target_score +
    regression_score thresholds.

  - **v2** — atomic-mutation discipline. Propose one change per
    evidence with retries (max 3 attempts), then a final combined
    validation with recursive rollback. Critic / judge / verdict
    unchanged from v1.

  - **v3** — v2 + rubric-based program.md. The program LLM emits a
    3-axis rubric (1–3 scoring) plus ≥5 binary checks. Judge
    produces all three signals (winner / per-axis scores / per-check
    pass/fail) in one call. Verdict combines four aggregate rates:
    fix_rate, regression_rate, rubric_improvement_rate,
    binary_checks_pass_rate.

Pick a strategy on the CLI: `autoresearch run --strategy v3`.
Use one programmatically:

    from agent_autoresearch.strategies import v3
    from agent_autoresearch.strategies.registry import get_strategy

    v3 = get_strategy("v3")    # equivalent — for dynamic dispatch
"""

# Per-version sentinel — each strategy's __init__ also sets this so
# the pipeline can dispatch without importing all three eagerly.
__all__ = ["registry", "v1", "v2", "v3"]
