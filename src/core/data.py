"""Neutral data shapes the pipeline operates on.

Adapters populate these from whatever your eval system actually has;
the rest of the library never looks at adapter-specific types. See
[`docs/writing_an_adapter.md`](../../docs/writing_an_adapter.md) for
worked examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Conversation primitives ──────────────────────────────────────────────────

@dataclass
class ToolCall:
    """One tool invocation made by the agent during a turn.

    Optional in `Turn.tool_calls`. Replay benefits from seeing them
    (e.g. judging whether the new skill would have called the right
    tool), but they're not strictly required — leave the list empty
    if your eval system doesn't capture tool spans.
    """
    name: str
    args: Any = None        # JSON-safe; usually a dict
    output: Any = None      # JSON-safe; truncate large outputs at adapter time
    error: str | None = None


@dataclass
class Turn:
    """One round-trip exchange — user message + agent reply.

    Convention: 1 turn = 1 user message paired with 1 agent response.
    If your transcript stores them separately (alternating user/agent
    rows), the adapter pairs them up before constructing Turns.

    Either `user` or `agent` may be the empty string when one side
    is absent (e.g. an event-triggered session has no user message at
    turn 1; an aborted reply has no agent text). Replay tolerates both.
    """
    turn: int                                # 1-indexed
    user: str = ""
    agent: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Conversation:
    """One session — a sequence of `Turn`s plus optional metadata.

    `session_id` is the primary key used by `Target.fix_session_ids`
    and `Target.regression_baseline_ids` to look conversations up.
    Adapter authors are responsible for keeping these IDs consistent
    between targets and conversations.
    """
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# ── Target primitives ────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """One structured failure finding from your eval pipeline.

    `category` is a short stable string (e.g. `"wrong_information"`,
    `"missing_tool_call"`). Use the same value across runs — the LLM
    treats it as a label.

    `details` is a free-form dict that ends up in the prompt as JSON.
    Common fields: `summary`, `quote`, `session_id`, `turn`, `rule`,
    `what_agent_did`. No schema enforcement — make it specific.

    `confidence` is optional. If you have one (e.g. your eval LLM
    emits one), pass it through; the strategy LLM uses it to weigh
    evidence.
    """
    category: str
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None


@dataclass
class Target:
    """One skill the pipeline should attempt to improve.

    Built by `Adapter.load_targets()`. Carries the evidence (why),
    the failing sessions (replay against), and the passing sessions
    (regression baseline).

    The `rank` field is optional; use it to give the CLI's `--top-n`
    selector a meaningful order. Lower rank = higher priority. If you
    don't set it, targets are processed in adapter order.
    """
    skill_name: str
    evidence: list[Evidence] = field(default_factory=list)
    fix_session_ids: list[str] = field(default_factory=list)
    regression_baseline_ids: list[str] = field(default_factory=list)
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_baselines(self) -> bool:
        return len(self.regression_baseline_ids) > 0

    @property
    def n_evidence(self) -> int:
        return len(self.evidence)
