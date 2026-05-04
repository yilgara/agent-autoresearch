"""Strategy implementations — versioned for easy comparison.

Each subdirectory is one complete implementation of the autoresearch
loop's LLM-driven stages. Different versions can change ANY stage
(program, propose, critic, responder, judge, verdict) — that's why
they're versioned together rather than mixed at the file level.

Versions currently in this package:

  - **v1** — the original loop. One bundled edit per round, freeform
    LLM judgment for both critic and judge, single verdict per
    target. This is what `--strategy v1` selects.

  - (future) **v2** — adds rubric in program.md (binary success
    criteria), single-mutation discipline (one atomic change per
    round, ratcheting through a list), explicit two-axis verdict.

To pick a strategy on the CLI: `autoresearch run --strategy v1`.
To use one programmatically: `from agent_autoresearch.strategies import v1`.
"""
