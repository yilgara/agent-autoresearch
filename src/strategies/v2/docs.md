# Strategy v2 — atomic-mutation propose

v2 keeps v1's metrics, replay, and verdict logic — but replaces the
single-shot `propose` with an **atomic-mutation loop**: one change
per evidence, validated by a per-attempt critic, with the survivors
stacked into the final state.

The motivation: v1 bundles every evidence item into a single edit.
When that edit fails, it's hard to tell which part is the problem —
maybe 3 of the 5 changes are good and 2 are bad, but the whole diff
gets rejected. v2 makes each evidence its own atomic mutation with
its own critic gate, then assembles the survivors.

## What's new in v2

| Change | Why |
|---|---|
| **One atomic edit per `propose` LLM call** | Smaller diffs are easier for the critic to evaluate. Each call has a single, scoped purpose. |
| **Per-evidence retry budget** (max 3 attempts) | If the LLM produces a bad edit, the next attempt sees the failure reason and tries a different angle. |
| **Per-attempt critic gate** | Each accepted change is critic-validated before being added to the cumulative state. |
| **`<action>done</action>` early-exit** | The proposer can signal "no more useful edits" mid-loop. |
| **One final replay over the cumulative state** | Result is captured for verdict — orchestrator does **not** re-run replay. |
| **No final critic, no rollback** | Per-attempt critics already validated each accepted change. If an evidence's 3 attempts all fail, it's skipped — previously-accepted changes stand. |

Critic prompt, replay prompt, and verdict logic are unchanged from
v1. Only `propose.py` (and how the orchestrator wires it) is
different.

## Flow

```mermaid
flowchart TB
    PROG[/"<b>program.md</b><br/>strategy doc"/]:::io
    SK[/"Current SKILL.md"/]:::io
    EV[/"target.evidence"/]:::io

    INIT["<b>state</b> ← SKILL.md<br/><b>accepted_log</b> ← [ ]<br/><b>idx</b> ← 0 · <b>attempt</b> ← 1"]:::nollm
    PROG --> INIT
    SK --> INIT
    EV --> INIT

    P1["<b>propose_atomic</b><br/>1 LLM call<br/><i>sees: state, evidence[idx],<br/>accepted_log, previous attempts</i>"]:::llm
    INIT --> P1

    ACT{{"<b>action?</b>"}}:::decision
    P1 --> ACT

    ACT -- "edit" --> CR
    ACT -. "skip<br/>(this evidence)" .-> NEXT
    ACT -. "done<br/>(stop entire loop)" .-> FR

    CR["<b>critic_per_attempt</b><br/>1 LLM call · small diff"]:::llm
    CR --> CRDEC{{"approves?"}}:::decision
    CRDEC == "yes" ==> ACCEPT
    CRDEC == "no" ==> RETRY

    RETRY{{"attempt &lt; 3?"}}:::decision
    RETRY == "yes (retry · attempt += 1)" ==> P1
    RETRY == "no (3 strikes — skip this evidence)" ==> NEXT

    ACCEPT["<b>accept</b><br/>state += change<br/>accepted_log += change"]:::nollm
    ACCEPT --> NEXT

    NEXT{{"more<br/>evidence?"}}:::decision
    NEXT == "yes (idx += 1, attempt = 1)" ==> P1
    NEXT == "no" ==> FR

    FR["<b>final_replay</b><br/>full fix + baseline sample<br/><i>result captured for verdict</i>"]:::llm
    FR --> RESULT["<b>ProposeResult</b><br/>action=edit · accepted_log"]:::nollm

    RESULT --> V{{"<b>compute_verdict</b><br/>fix_target_score &gt; 0<br/>regression_score ≥ 90%"}}:::verdict

    classDef nollm fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    classDef llm fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100
    classDef io fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef verdict fill:#FFF9C4,stroke:#F9A825,stroke-width:3px,color:#F57F17
    classDef decision fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
```

## Metrics & verdict — same as v1

| Metric | Population | Per-session signal | Aggregate | Threshold |
|---|---|---|---|---|
| `fix_target_score` | fix sessions | `new_passes` | fraction where new passes | strict `> 0` |
| `regression_score` | baseline sessions | `new_passes` | fraction where new passes | `≥ 0.90` |

```python
THRESHOLDS = {
    "fix_target_min":   0.0,   # strict `>` — any improvement counts
    "regression_min":   0.9,
}
```

Verdict labels are identical to v1:

| Outcome | Conditions |
|---|---|
| `ACCEPT` | `fix_target_score > 0` AND `regression_score ≥ 0.9` |
| `REJECT` | `regression_score < 0.9` |
| `HUMAN_REVIEW` | regression OK but `fix_target_score == 0` |
| `SKIP` | propose returned skip (no accepted changes) |

**No critic gate at verdict time.** Per-attempt critics already
validated each accepted change inside `propose`; the orchestrator
passes `critic_result=None` to `compute_verdict`. There is no
`critic.md` artifact for v2.

## LLM call count

For E evidence (default), all accepted on first try, `fix_sample=3,
baseline_sample=3`:

| Stage | Calls |
|---|---|
| `build_program` | 1 |
| Per evidence: `propose_atomic` + `critic_per_attempt` | 2 × E |
| `final_replay` (responder + judge × 6 sessions) | 12 |
| **Total** | **2E + 13** |

Worst case adds up to 2 retries per evidence (3 attempts × 2 calls
each). No final critic, no rollback re-runs.

## Public contract

`propose()` still returns one `ProposeResult` with the cumulative
`new_skill_md`. v2 adds two informational fields:

- `accepted_log: list[AtomicAttempt]` — one entry per accepted change
- `attempts_log: list[AtomicAttempt]` — every attempt, accepted or not

Neither affects verdict logic; they're for the markdown trace and
debugging.

## When v2 is the right choice

- **Skills with multiple distinct failure modes.** v1 bundles them
  all into one diff; v2 keeps each evidence's change separable and
  individually critic-validated.
- **You care about attribution.** Each accepted change has its own
  reasoning + the evidence it targets, logged in `accepted_log`.
- **You want recoverable failures.** A bad edit doesn't kill the
  whole run — the loop just moves to the next evidence.

## When to use v3 instead

v2 still uses a single `new_passes` boolean per session — same shape
as v1. If you need **per-axis quality signal** ("the new prompt is
better at dietary handling but worse at result grounding") or
**invariant assertions** ("the new prompt must never refuse a
reasonable lookup"), use v3.
