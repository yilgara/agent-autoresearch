"""Strategy v1 — the original autoresearch loop.

One file per LLM-driven stage; each function is independently
callable for partial-pipeline runs (e.g. testing one prompt change
in isolation).

Stages currently implemented (more landing as Phase 3 + 4 of the
public refactor port):

  Stage     Step  Module       Function          Result
  ────────  ────  ───────────  ────────────────  ──────────────
  Program   4     program.py   build_program     ProgramResult
  Propose   5     propose.py   propose           ProposeResult
  Critic    6     critic.py    critic            CriticResult

  (Pending)
  Responder 7a    responder.py — soft replay LLM #4
  Judge     7b    judge.py     — soft replay LLM #5
  Replay    7     replay.py    — orchestrates 7a + 7b per session
  Verdict   8     verdict.py   — deterministic combine

Each prompt template lives in `prompts/` next to this code so v1
is self-contained — copy the whole folder to fork as v2.
"""

from agent_autoresearch.strategies.v1.critic import (
    CRITIC_MAX_TOKENS,
    CriticResult,
    CriticVerdict,
    critic,
)
from agent_autoresearch.strategies.v1.program import (
    EVIDENCE_MAX_TOTAL,
    EVIDENCE_PER_CATEGORY,
    PROGRAM_MAX_TOKENS,
    ProgramResult,
    build_program,
    format_evidence_block,
)
from agent_autoresearch.strategies.v1.propose import (
    PROPOSE_MAX_TOKENS,
    ProposeAction,
    ProposeResult,
    propose,
)


__all__ = [
    # program (step 4)
    "build_program",
    "ProgramResult",
    "format_evidence_block",
    "PROGRAM_MAX_TOKENS",
    "EVIDENCE_PER_CATEGORY",
    "EVIDENCE_MAX_TOTAL",
    # propose (step 5)
    "propose",
    "ProposeResult",
    "ProposeAction",
    "PROPOSE_MAX_TOKENS",
    # critic (step 6)
    "critic",
    "CriticResult",
    "CriticVerdict",
    "CRITIC_MAX_TOKENS",
]
