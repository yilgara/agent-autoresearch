# agent-autoresearch

> Auto-improve agent skill prompts from your eval pipeline output.

Take one day's worth of agent-evaluation findings, propose targeted edits
to the underlying skill prompts, validate the proposals via simulated
replay against real failed sessions, and hand the verified diffs to a
human for review. End-to-end LLM cost is ~$1 per run on top-3 targets.

Inspired by [karpathy/autoresearch][karpathy] (autonomous overnight ML
research), but for agent **skill prompts** rather than ML training
code.

[karpathy]: https://github.com/karpathy/autoresearch

> **⚠️ v0.x — early access.** The pipeline runs end-to-end and produces
> useful proposals, but the prompts and thresholds are still being
> calibrated. 

---

## What it does

You bring:
- A list of skills (or any agent prompts) on disk
- A daily eval report identifying which skills are failing
- Transcripts of the failing sessions

The loop produces, per skill:
1. A focused improvement strategy (`program.md`) — generated from your eval evidence
2. A proposed `v_new.md` for the skill — an Edit, a Create, or an explicit Skip
3. A critic's audit of the diff (Validation Layer A)
4. A soft-replay run against real failed sessions + passing baselines (Validation Layer B)
5. A deterministic verdict: **ACCEPT** / **HUMAN_REVIEW** / **REJECT** / **SKIP**

Approved diffs land in `outputs/run_<ts>/<skill>/` for manual review.
The library **never auto-merges** to your skills directory — every
shipped change goes through human eyes.

## Pipeline

The 8 steps split across three phases:

```
Phase A · Parse + Combine    (no LLM)
    1. Parse eval reports
    2. Load session transcripts
    3. Build per-skill Target bundles

Phase B · Propose            (3 LLM calls per target)
    4. build_program — strategy doc per target
    5. propose      — Edit / Create / Skip + new SKILL.md
    6. critic       — audit the diff (Validation Layer A)

Phase C · Validate           (2 LLM calls × N sessions per target)
    7. soft replay  — responder + judge per session (Validation Layer B)

Step 8 · Verdict             (deterministic — no LLM)
    8. Combine critic + replay scores → ACCEPT / HUMAN_REVIEW / REJECT
```

Full architecture details: [`docs/pipeline.md`](docs/pipeline.md).

## Quickstart

### 1. Install

```bash
pip install agent-autoresearch
```

### 2. Try a dry-run with synthetic data

```bash
autoresearch run --adapter synthetic --dry-run
```

No API key needed — parses bundled example data and prints the targets
the pipeline would operate on.

### 3. Live run with your data

```bash
export ANTHROPIC_API_KEY=sk-ant-...
autoresearch run --adapter <your-adapter> --top-n 3
```

Outputs land in `outputs/run_<ts>/<repo>/<skill>/`. Read the
`verdict.md` first; if ACCEPT, eyeball `diff.txt`, then copy
`v_new.md` to your skills repo manually.

## Architecture

The library is split into a **stable core** (the pipeline + LLM
calls) and **pluggable adapters** (your eval format, your skill
storage, your LLM provider).

### Core abstractions

- **`Target`** — one skill the pipeline is trying to improve. Carries
  `evidence` (your eval findings), `fix_sessions` (where the skill
  broke), `regression_baselines` (where it worked).
- **`SkillIO`** — abstract interface for loading and writing skill
  prompts. Default `FilesystemSkillIO` covers most teams.
- **`EvidenceSource`** — abstract interface for "give me the failures
  and transcripts for run X." Each eval pipeline implements its own.
- **`Conversation`** — schema-stable representation of a session
  transcript. Adapters convert their own session format into this.
- **`LLMProvider`** — abstract interface; default impl is Anthropic.


### Writing your own adapter

To wire autoresearch into your own eval pipeline, you implement three
classes:

- **`Adapter`** — translates your eval pipeline's output (reports,
  database rows, JSON, whatever) into the library's `Target` and
  `Conversation` shapes.
- **`SkillIO`** — tells the library how to load and write your skill
  files. The default `FilesystemSkillIO` works for most layouts; only
  override if your skills live in something exotic (S3, a DB, a git
  service).
- **`LLMProvider`** — optional. The default Anthropic implementation
  ships with the library. Override only if you need to call a different
  model provider.

Most teams only need to write the `Adapter` — the other two have
sensible defaults. See [`docs/writing_an_adapter.md`](docs/writing_an_adapter.md)
for the full guide with a worked example.

## Configuration

| Env var | Required | What it's for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | All five LLM calls (Sonnet 4.5 by default) |
| `AUTORESEARCH_INPUTS_DIR` | no (default `./inputs`) | Where adapters look for input files |
| `AUTORESEARCH_OUTPUTS_DIR` | no (default `./outputs`) | Where results land |

CLI flags override env defaults — `autoresearch run --help` for the
full list.

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

## Honest limitations

- **Validation is LLM-grading-LLM.** The replay step is one model
  imagining what another would do, judged by a third. It catches
  obvious wins and clear regressions, but is no substitute for
  observing the actual edit in production.
- **Run-to-run variance.** The proposer LLM doesn't always produce
  identical edits — different runs may yield different verdicts on
  the same input. This is by design (more exploration); if it's a
  problem for you, run twice and compare, or pin to a single seed.
- **Conservative thresholds.** Default `fix_target_min=70%`,
  `regression_min=90%`. Tunable in [`config.py`](agent_autoresearch/config.py).

## Cost

Per top-N target:
- Strategy: ~$0.02
- Propose: ~$0.03
- Critic: ~$0.01
- Replay: ~$0.05–0.40 (depends on session count)
- **Total: ~$0.10–0.50 per target**

Top-3 targets per run with 3 fix-target + 3 baseline replays each:
~$1 per autoresearch run. Trivial vs. the value of a single accepted
skill improvement.


## Contributing

PRs welcome. Before opening one:

1. Open an issue describing the problem / change
2. For prompt changes, run the bundled comparison harness so we can
   eyeball before/after on the same eval data
3. New adapters: add tests against your own synthetic data; we don't
   require you to share your real eval pipeline

## License

MIT — see [`LICENSE`](LICENSE).


