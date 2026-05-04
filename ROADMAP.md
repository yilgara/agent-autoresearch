# Roadmap

What's shipping when, and why in this order. The README's [Roadmap
section](./README.md#roadmap) is the short version; this is where the
reasoning lives.

---

## v0.1 — current

The shape we have today. Single proposed edit per skill, ACCEPT /
HUMAN_REVIEW / REJECT / SKIP based on freeform LLM judgment.

**What works end-to-end:**
- Pipeline: parse → target → propose → critic → soft replay → verdict
- 5 LLM calls per target (program / propose / critic / responder / judge)
- Per-target output folder with full audit trail
- Conservative thresholds — burden of proof on the new skill

**Known limitations** (carried into v0.2 because the pipeline is the
right shape; refinement comes after pluggability):
- One bundled edit per round (no decomposition into atomic changes)
- LLM-as-judge for both regression and fix scoring (no explicit binary
  criteria)
- Run-to-run variance on borderline cases

---


## v0.2 — Rubric + single-mutation discipline

### A. Strategy LLM auto-generates a rubric per target

Today's `program.md` describes the strategy in prose ("what to fix",
"what NOT to change"). v0.3 adds an explicit **binary success
criteria** section, derived from the same evidence:

```markdown
## Success criteria (auto-generated, used by the judge)
- [ ] Does the new tool plan call `get_extended_profile` first?
- [ ] Does the new reply respect the user's stated location?
- [ ] Did the new reply skip the redundant clarifying question? (yes = good)
```

These are checks a per-session LLM call can answer reliably (yes/no),
unlike "is the new reply better overall?" which is freeform judgment.
Judge step gets both — the freeform call AND scores against each
criterion.

**Why:** ties the judge's signal directly to what the strategy
asked for. Less stochastic, more auditable, and the criteria are
visible in the output folder so a reviewer can sanity-check.

### B. Single mutation per round (decompose the strategy)

Today's `propose` step makes one bundled edit even when `program.md`
identifies multiple weaknesses. v0.2 splits the strategy into atomic
changes and proposes ONE at a time, validates it, and only then moves
on to the next.

Loop becomes:
1. Step 4 emits a list of mutations prioritized by frequency
2. For each mutation: propose → critic → replay → verdict
3. If accepted, the next mutation runs against the *modified* skill
4. Stop when list exhausted, budget hit, or N consecutive rejects

**Why:** smaller diffs, cleaner attribution per axis, easier rollback,
matches "isolate one variable per experiment" discipline.
Cost goes up 3-5× per skill in exchange for better signal quality.

### C. Verdict logic uses both axes explicitly

```
ACCEPT iff
  regression_tests_still_pass            (must-pass guardrail)
  AND fix_target_rubric_score_improved   (axis to maximize)
```

Today this is implicit in our `regression_score ≥ 90%` AND
`fix_target_score ≥ 70%` thresholds. v0.2 makes it the explicit
two-axis structure: tests must pass; rubric
score must rise. Same idea, but now the rubric is per-target binary
criteria instead of a single freeform judge call.

### What is added, what kept same

| Added | Kept |
|---|---|
| Two-axis evaluation (tests + rubric) | Production data as the test set, not synthetic test cases |
| Binary criteria on each axis | Strategy LLM authors them; humans don't |
| Single-mutation discipline | One round per day, not hundreds overnight |
| Keep-iff-improves ratchet | LLM judge, not deterministic check (criteria narrow it down) |

**What we don't add:** the hundreds-of-mutations-overnight pattern
or the human-authored test cases. Both run counter to "low setup,
production-driven, daily cadence" which is our core philosophy.

---

## v0.3 — Multi-turn replay + cross-run memory

**Ships:**
- True multi-turn replay (responder generates the agent's reply for
  *every* turn in a session, not just the focus turn). Catches
  failures that emerge across multiple turns.
- Cross-run dedupe: if yesterday's verdict was REJECT on skill X with
  a particular edit shape, don't propose the same shape today.
- Library of "known-bad edit patterns" the proposer learns to avoid.

**Why deferred to v0.3:** multi-turn replay is more expensive and
complex; the focus-turn approach catches most failures at lower cost.
Worth doing once we know which failure modes the focus-turn approach
misses in practice.

---

## v0.4 — Multi-LLM-provider support

**Ships:**
- OpenAI provider
- Bedrock provider
- OpenRouter provider (one-stop for many models)
- Per-step model config (e.g. cheap model for critic, strong model for
  judge)
- `AUTORESEARCH_LLM_PROVIDER` env var + per-step overrides

**Why this and not earlier:** Anthropic-only is fine for v0.x. Adding
providers before the API contract is stable creates churn for users.

---

## v1.0 — Stable contract

**Ships:**
- Frozen public API for adapters (no breaking changes after this)
- Performance benchmarks


---

## Possible future directions (not committed)

These are interesting but not yet on the path:

- **Hybrid pipeline.** Use v0.2's pipeline to
  identify which skills are broken in production, then run 
  hundreds-of-mutations on just those skills. Depth where it matters, 
  breadth everywhere else.
- **Auto-merge with safety guardrails.** If a verdict is ACCEPT and
  identical edits land 3 days in a row across N runs, optionally
  auto-merge to a configured branch. Off by default.
- **Live A/B test adapter.** Wire the proposed v_new SKILL.md into a
  production canary for X% of traffic, observe real eval scores
  tomorrow, accept based on production data rather than soft replay.
- **GHA workflow templates.** Drop-in YAML for running autoresearch
  in CI after a daily eval pipeline finishes. Currently each user
  writes their own.

---

## How decisions get made

- **Issue or discussion** before code on anything bigger than a bug fix
- **Real user feedback > internal speculation** — we'll skip features
  nobody actually uses, even if they're on this list
- **Honest caveats stay honest** — we don't ship "ACCEPT means safe to
  merge" until we've observed enough runs to trust it

If something on this list matters to you, [open an
issue](https://github.com/yilgara/agent-autoresearch/issues) — that's
the strongest signal for what we work on next.
