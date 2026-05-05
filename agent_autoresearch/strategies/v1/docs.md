# Strategy v1 — the original autoresearch loop

The first cut of the propose → validate → verdict loop. One bundled
edit per round, freeform LLM judgment for both critic and judge, two
aggregate rates feeding a 5-label verdict.

This is the simplest of the three versions and the cheapest to run
(~$0.07 per target with default sample sizes). It's the right
starting point if you've never run autoresearch on your data before
— the output is easy to read and the failure modes are obvious.

## What's in v1

| Stage | LLM calls | Output |
|---|---:|---|
| `build_program` | 1 | `program.md` — strategy doc |
| `propose` | 1 | One bundled `v_new.md` (or `<action>skip</action>`) |
| `critic` | 1 | `<verdict>APPROVE / REQUEST_CHANGES</verdict>` |
| `responder` (per session) | 1 × N | `<reply>` under the new skill |
| `judge` (per session) | 1 × N | `<winner>new / old / tie</winner>` |
| `compute_verdict` | 0 | One of `ACCEPT / HUMAN_REVIEW / REJECT / SKIP / NO_VALIDATION` |

Total per target with 3 fix + 3 baseline replays: **3 + 12 = 15 LLM
calls** ≈ $0.07 with Sonnet 4.5.

## Flow

```mermaid
flowchart TB
    %% ── Inputs ──
    PROG_IN[/"Target<br/><i>evidence + fix + baselines</i>"/]:::io
    SKILL_IN[/"Current SKILL.md"/]:::io

    %% ── Phase B · Propose (3 LLM calls) ──
    subgraph PHB[" Phase B · Propose — 3 LLM calls "]
        direction TB
        BP1["<b>build_program</b><br/>1 LLM call<br/>→ program.md"]:::llm
        BP2["<b>propose</b><br/>1 LLM call<br/>→ v_new.md or skip"]:::llm
        BP3["<b>critic</b><br/>1 LLM call<br/>→ APPROVE / REQUEST_CHANGES"]:::llm
        BP1 --> BP2 --> BP3
    end
    PROG_IN --> BP1
    SKILL_IN --> BP2

    %% ── Action gate ──
    DEC{{"<b>propose action?</b>"}}:::decision
    BP2 --> DEC

    %% ── Phase C · Replay (2 LLM calls × N sessions) ──
    subgraph PHC[" Phase C · Soft replay — 2 LLM calls × N sessions "]
        direction LR
        R1["<b>responder</b><br/>reply under v_new.md"]:::llm
        R2["<b>judge</b><br/>winner: new / old / tie"]:::llm
        AGG["<b>aggregate</b><br/>fix_target_score<br/>regression_score"]:::nollm
        R1 --> R2 --> AGG
    end
    DEC == "edit" ==> BP3
    DEC == "edit" ==> R1

    %% ── Verdict (deterministic) ──
    V{{"<b>compute_verdict</b><br/><i>2-axis thresholds</i><br/>fix_target_score ≥ 70%<br/>regression_score ≥ 90%"}}:::verdict
    DEC -. "skip" .-> V
    BP3 --> V
    AGG --> V

    OUT[/"<b>outputs/run/skill/</b><br/>program · v_old · v_new · diff<br/>critic · replay · verdict"/]:::io
    V ==> OUT

    %% ── Styling ──
    classDef nollm fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    classDef llm fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100
    classDef io fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1
    classDef verdict fill:#FFF9C4,stroke:#F9A825,stroke-width:3px,color:#F57F17
    classDef decision fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C
    style PHB fill:#FAFAFA,stroke:#9E9E9E,stroke-dasharray:5 5
    style PHC fill:#FAFAFA,stroke:#9E9E9E,stroke-dasharray:5 5
```

## Verdict thresholds

```python
THRESHOLDS = {
    "fix_target_min":   0.70,    # ≥ 70% of fix sessions: new wins
    "regression_min":   0.90,    # ≥ 90% of baselines: new doesn't lose
    "fix_reject_below": 0.50,    # below this → hard REJECT
}
```

| | Acceptance | Hard reject |
|---|---|---|
| `fix_target_score` | ≥ 0.70 | < 0.50 |
| `regression_score` | ≥ 0.90 | — |
| `critic` | APPROVE | REQUEST_CHANGES |

Between `fix_reject_below` and `fix_target_min` → `HUMAN_REVIEW`.

## When v1 is the right choice

- **First-time runs.** You're trying autoresearch on your eval data
  for the first time. Read v1 outputs, make sure the propose stage
  produces edits that look reasonable, then graduate to v2/v3.
- **Cost-sensitive runs.** Quick top-N sweeps where you mainly care
  about which skills *might* benefit from edits, less about
  surgical change quality.
- **Debugging adapters.** When you suspect your adapter is the
  problem (wrong evidence, wrong session IDs), v1's simpler flow
  produces cleaner failure messages.

## When to graduate

- **Multi-issue skills.** v1 bundles all evidence into one
  proposed edit. If your skills have 3+ distinct failure modes, the
  resulting diffs get sprawling and hard to review. → use v2.
- **Need attribution.** "Did the dietary-filter fix help?" can't be
  answered in v1 (one edit either passes or fails as a whole). → v3
  with rubric scoring gives you per-axis signal.
