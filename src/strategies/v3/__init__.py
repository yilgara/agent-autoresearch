"""Strategy v1 — the original autoresearch loop.

One file per stage; each function is independently callable for
partial-pipeline runs (e.g. testing one prompt change in isolation).

Stages in this strategy version:

  Stage      Step  Module          Function         Result
  ─────────  ────  ──────────────  ───────────────  ───────────────
  Program    4     program.py      build_program    ProgramResult
  Propose    5     propose.py      propose          ProposeResult
  Critic     6     critic.py       critic           CriticResult
  Responder  7a    responder.py    run_responder    ResponderResult
  Judge      7b    judge.py        run_judge        JudgeResult
  Replay     7     replay.py       soft_replay      ReplayResult
  Verdict    8     verdict.py      compute_verdict  Verdict

Each prompt template lives in `prompts/` next to this code so v1 is
self-contained — copy the whole folder to fork as v2.
"""


STRATEGY_VERSION = "v3"

from agent_autoresearch.strategies.v3.critic import (
    CRITIC_MAX_TOKENS,
    CriticResult,
    CriticVerdict,
    critic,
)
from agent_autoresearch.strategies.v3.judge import (
    JUDGE_MAX_TOKENS,
    CheckOutcome,
    CheckResult,
    JudgeResult,
    RubricVote,
    RubricWinner,
    run_judge,
)
from agent_autoresearch.strategies.v3.program import (
    EVIDENCE_MAX_TOTAL,
    EVIDENCE_PER_CATEGORY,
    MIN_BINARY_CHECKS,
    PROGRAM_MAX_TOKENS,
    RUBRIC_AXIS_COUNT,
    BinaryCheck,
    ProgramResult,
    RubricAxis,
    build_program,
    format_evidence_block,
)
from agent_autoresearch.strategies.v3.propose import (
    MAX_ATTEMPTS_PER_EVIDENCE,
    PROPOSE_MAX_TOKENS,
    AtomicAction,
    AtomicAttempt,
    ProposeAction,
    ProposeResult,
    propose,
)
from agent_autoresearch.strategies.v3.replay import (
    DEFAULT_BASELINE_SAMPLE,
    DEFAULT_FIX_SAMPLE,
    ReplayResult,
    SessionReplay,
    SessionRole,
    soft_replay,
)
from agent_autoresearch.strategies.v3.responder import (
    RESPONDER_MAX_TOKENS,
    ResponderResult,
    run_responder,
)
from agent_autoresearch.strategies.v3.verdict import (
    THRESHOLDS,
    Verdict,
    VerdictLabel,
    compute_verdict,
)


__all__ = [
    # program (step 4) — v3 with rubric + binary checks
    "build_program",
    "ProgramResult",
    "RubricAxis",
    "BinaryCheck",
    "format_evidence_block",
    "PROGRAM_MAX_TOKENS",
    "EVIDENCE_PER_CATEGORY",
    "EVIDENCE_MAX_TOTAL",
    "RUBRIC_AXIS_COUNT",
    "MIN_BINARY_CHECKS",
    # propose (step 5) — v2 atomic-mutation
    "propose",
    "ProposeResult",
    "ProposeAction",
    "AtomicAction",
    "AtomicAttempt",
    "PROPOSE_MAX_TOKENS",
    "MAX_ATTEMPTS_PER_EVIDENCE",
    # critic (step 6)
    "critic",
    "CriticResult",
    "CriticVerdict",
    "CRITIC_MAX_TOKENS",
    # responder (step 7a)
    "run_responder",
    "ResponderResult",
    "RESPONDER_MAX_TOKENS",
    # judge (step 7b) — v3 with 3 signals (new_passes / rubric votes / checks)
    "run_judge",
    "JudgeResult",
    "RubricVote",
    "RubricWinner",
    "CheckOutcome",
    "CheckResult",
    "JUDGE_MAX_TOKENS",
    # replay orchestrator (step 7)
    "soft_replay",
    "ReplayResult",
    "SessionReplay",
    "SessionRole",
    "DEFAULT_FIX_SAMPLE",
    "DEFAULT_BASELINE_SAMPLE",
    # verdict (step 8)
    "compute_verdict",
    "Verdict",
    "VerdictLabel",
    "THRESHOLDS",
]
