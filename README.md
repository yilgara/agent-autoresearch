# agent-autoresearch

> Auto-improve agent skill prompts from your eval pipeline output.

Take one batch of agent-evaluation findings, propose targeted edits to
the underlying skill prompts, validate the proposals via simulated
replay against real failed sessions, and hand the verified diffs to a
human for review.

Inspired by [karpathy/autoresearch][karpathy] (autonomous overnight ML
research), but for agent **skill prompts** rather than ML training
code.

[karpathy]: https://github.com/karpathy/autoresearch

> **⚠️ v0.x — early access.** The pipeline runs end-to-end and produces
> useful proposals, but the prompts and thresholds are still being
> calibrated against real eval data.

---

## What it does

You bring:
- Skill prompts on disk (`skills/<name>/SKILL.md`)
- Either pass/fail labels for past sessions, or just raw transcripts
- Session transcripts in JSONL or JSON

The loop produces, per skill:
1. A focused improvement strategy (`program.md`) — generated from your evidence
2. A proposed `v_new.md` — an Edit, or an explicit Skip
3. A critic's audit of the diff (Validation Layer A)
4. A soft-replay run against real failed sessions + passing baselines (Validation Layer B)
5. A deterministic verdict: **ACCEPT** / **HUMAN_REVIEW** / **REJECT** / **SKIP**

Approved diffs land in `outputs/<run_id>/<skill>/` for manual review.
The library **never auto-merges** to your skills directory — every
shipped change goes through human eyes.

## Pipeline

```
Phase A · Parse + Combine    (no LLM)
    1. Adapter loads targets + transcripts
    2. Pipeline picks top-N by failure count

Phase B · Propose            (3 LLM calls per target)
    3. program  — strategy doc per target
    4. propose  — Edit / Skip + new SKILL.md
    5. critic   — audit the diff (Validation Layer A)

Phase C · Validate           (2 LLM calls × N sessions per target)
    6. soft replay — responder + judge per session (Validation Layer B)

Step 7 · Verdict             (deterministic — no LLM)
    Combine critic + replay scores → ACCEPT / HUMAN_REVIEW / REJECT
```

Full architecture details: [`docs/pipeline.md`](docs/pipeline.md).

## Quickstart

### 1. Install

```bash
pip install agent-autoresearch    # or: pip install -e . from the cloned repo
```

### 2. Try the bundled synthetic demo (full pipeline, real LLM)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
autoresearch run --adapter synthetic \
                 --skills-root examples/synthetic-skills \
                 --top-n 1
```

Runs the full 8-step pipeline against bundled hardcoded fixtures + 2
example skill files. Expect ~$0.10 in API cost. Output lands in
`outputs/<run_id>/find-restaurant/` — read `verdict.md`, then `diff.txt`.

For a no-network sanity check, drop `--top-n 1` and add `--dry-run`:

```bash
autoresearch run --adapter synthetic --dry-run
```

### 3. Live run with your data

Pick the adapter that matches what you have:

| You have… | Use |
|---|---|
| Pre-labeled CSV + transcripts | `--adapter csv` |
| Just transcripts (no labels) | `--adapter jsonl_judge` (LLM-judges them, then reuses csv) |
| A custom eval pipeline | Write your own — see [`docs/writing_an_adapter.md`](docs/writing_an_adapter.md) |

```bash
autoresearch run --adapter csv \
                 --skills-root ./skills \
                 --top-n 3
```

See [`docs/csv_adapter.md`](docs/csv_adapter.md) for the CSV schema +
file layout.

## Architecture

The library is split into a **stable core** (the pipeline + LLM calls)
and **pluggable adapters** (your eval format, your skill storage, your
LLM provider).

### Core abstractions

- **`Adapter`** — loads `Target` + `Conversation` shapes from your
  data source (eval reports, JSONL, DB, anywhere)
- **`SkillIO`** — loads + writes skill prompts. Default
  `FilesystemSkillIO` covers `skills/<name>/SKILL.md` layouts
- **`LLMProvider`** — chat-completion wrapper. Default is Anthropic
  Sonnet 4.5

### Writing your own adapter

Most teams only need to subclass `Adapter` — `SkillIO` and
`LLMProvider` have sensible defaults. The adapter implements two
methods: `load_targets()` and `load_conversations()`.

See [`docs/writing_an_adapter.md`](docs/writing_an_adapter.md) for a
full worked example.

## Configuration

| Env var | Required | What it's for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes (live runs) | Sonnet 4.5 for all stages |

Everything else is a CLI flag — `autoresearch run --help` for the
full list. Verdict thresholds live in
[`agent_autoresearch/strategies/v1/verdict.py`](agent_autoresearch/strategies/v1/verdict.py)
(the `THRESHOLDS` dict at the top).

## What's deliberately NOT in the library

- **Auto-merge.** Approved edits stay in `outputs/`; you copy them to
  your skills repo via your normal review process. The cost-of-getting-
  it-wrong on a skill prompt is too high to skip humans.
- **Multi-round ratchet across days.** The library does parallel top-N
  per single run. True ratcheting (re-rank after each accepted edit)
  requires re-running your upstream eval pipeline between rounds —
  that's your eval system's job.
- **Cross-run dedupe.** If yesterday's report flagged X and today's
  also flags X, the library proposes for both. A future cross-run
  memory layer is on the roadmap.
- **Skill discovery.** This is an *improvement* loop for skills that
  already have prompts. Inventing new skills from scratch is a
  different problem.

## Honest limitations

- **Validation is LLM-grading-LLM.** The replay step is one model
  imagining what another would do, judged by a third. It catches
  obvious wins and clear regressions, but is no substitute for
  observing the actual edit in production.
- **Run-to-run variance.** The proposer LLM doesn't always produce
  identical edits — different runs may yield different verdicts on
  the same input. By design (more exploration); if it's a problem
  for you, run twice and compare.
- **Conservative thresholds.** Default `fix_target_min=70%`,
  `regression_min=90%`. Tune after observing 5–10 real runs.

## Cost

Roughly per top-N target (using Sonnet 4.5):
- Phase B (program + propose + critic): ~$0.05–0.08
- Phase C (replay): ~$0.05–0.40 depending on session count

A typical run with `top_n=3, fix_sample=3, baseline_sample=3` lands
around $1 total — trivial vs. the value of one accepted skill
improvement.

## Contributing

PRs welcome. Before opening one, open an issue describing the change.
For prompt edits especially, please include a before/after run on real
or synthetic eval data so reviewers can compare outputs side-by-side.

## License

MIT — see [`LICENSE`](LICENSE).
