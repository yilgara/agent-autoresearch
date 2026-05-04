# The CSV adapter

The simplest way to use `autoresearch` without writing any Python.

You give it a CSV with one row per `(session, skill)` pass/fail label, plus a folder of conversation transcripts. It loads them, builds Targets, and runs the pipeline. No code on your side.

## When to use this

Pick the CSV adapter if **any** of these is true:

- You already eval your agent and have results in a spreadsheet, a script, Promptfoo, DeepEval, or anywhere else that can export a CSV.
- You want to manually label a small number of failing sessions and try `autoresearch` on them before investing in pipeline plumbing.
- You don't have an eval pipeline yet but you have transcripts — use [`--adapter jsonl_judge`](#starting-from-just-transcripts) to generate the CSV automatically.
- You want to experiment with hand-edited labels (the LLM judge got something wrong; you fix it in the CSV; the next `run` uses the fixed labels).

If your eval pipeline is custom enough that producing a CSV would be lossy, write your own [adapter](./writing_an_adapter.md) instead.

---

## Two paths in

Either you bring your own labels (use `--adapter csv`), or you only have transcripts and let an LLM produce the labels (use `--adapter jsonl_judge`, which is just `csv` with a judging step in front).

### Path A — you already have labels

```bash
# 1. Lay out your data:
#    eval_data/
#    ├── results.csv         (your labels)
#    └── transcripts/
#        ├── sess_001.jsonl
#        ├── sess_002.jsonl
#        └── ...
# 2. Run:
autoresearch run --adapter csv \
    --skills-root ./skills \
    --top-n 3
```

By default the adapter looks for `./results.csv` and `./transcripts/`. Set them anywhere you want by constructing the adapter from Python instead of the CLI (see [Programmatic use](#programmatic-use) below).

### Path B — starting from just transcripts

Use `--adapter jsonl_judge` instead. It runs the judge on first use, writes the labeled CSV, then internally delegates to `CSVAdapter`:

```bash
# 1. Judge + view targets, no pipeline yet (lets you review results.csv first):
autoresearch run --adapter jsonl_judge --dry-run --skills-root ./skills

# 2. Open results.csv, hand-edit any mislabels (optional but recommended on first run).

# 3. Real run — judge step is now cached, no extra LLM calls before the pipeline:
autoresearch run --adapter jsonl_judge --skills-root ./skills --top-n 3
```

The judge is one LLM call per transcript. Results are cached in `results.csv` plus a `results.csv.judged` sidecar — re-running is free, and adding new transcripts later only judges the new ones.

The judge can be wrong. Open `results.csv` in a spreadsheet, fix any mislabels, and re-run. Hand-edited rows behave identically to LLM-generated ones — and you can switch to `--adapter csv` for subsequent runs to be sure no judging happens at all.

To force re-judging (after editing the prompt or skill list), delete `results.csv` and `results.csv.judged`, or construct `JSONLJudgeAdapter(force_rejudge=True)` from Python.

---

## CSV schema

Three required columns, four optional. Extra columns are ignored.

| Column | Required | Notes |
|---|---|---|
| `session_id` | yes | Primary key. Must match the transcript filename stem (`sess_001` ↔ `sess_001.jsonl`). |
| `skill` | yes | Skill name as it appears on disk. No auto-conversion between kebab/snake — match exactly. |
| `score` | yes | `pass` / `fail` (also accepts `0`/`1`, `true`/`false`, case-insensitive). |
| `failure_category` | optional | Short snake_case label like `wrong_arguments`. Used for clustering related failures. |
| `failure_summary` | optional | One or two sentences explaining the failure. Free text. |
| `failure_turn` | optional | 1-indexed turn the failure happened on. Used by replay to focus on the right turn. |
| `transcript_path` | optional | Per-row override of the default `<transcripts_dir>/<session_id>.{jsonl,json}` lookup. |

Pass rows can leave the failure-* columns blank.

### One session, multiple skills

A session can exercise more than one skill. Emit one row per `(session_id, skill)` pair:

```csv
session_id,skill,score,failure_category,failure_summary,failure_turn,transcript_path
sess_42,find-restaurant,pass,,,,
sess_42,book-table,fail,wrong_arguments,"Date passed as MM/DD/YYYY; tool expects ISO.",4,
```

Here `sess_42` is a baseline for `find-restaurant` (the search part worked) **and** a fix-target for `book-table` (the booking part broke). The pipeline runs each skill independently. Replay focuses on the right turn per skill — turn 4 for `book-table`'s replay, last-turn (default) for `find-restaurant`'s baseline.

### `failure_category` is a clustering label

It's a short stable string used to **bucket similar failures** so the program-builder LLM can see "this skill has 5 wrong_arguments failures and 2 missing_filter failures" instead of seven undifferentiated examples.

Reuse the same category across similar failures. Free-form is fine; we don't enforce a taxonomy.

Common values to start with:

```
wrong_tool · wrong_arguments · missing_filter · ignored_constraint
hallucination · format_error · incomplete_answer · refused_reasonable_request
```

---

## Transcript file format

Auto-detected by extension. Two shapes accepted:

### `.jsonl` — one turn per line

```jsonl
{"turn": 1, "user": "Find me dinner downtown.", "agent": "Sure, any preferences?"}
{"turn": 2, "user": "Vegan please.", "agent": "Bob's Steakhouse is great!", "tool_calls": [{"name": "search_restaurants", "args": {"q": "downtown"}, "output": [{"name": "Bob's Steakhouse"}]}]}
```

### `.json` — single object with a `turns` array

```json
{
  "session_id": "sess_001",
  "turns": [
    {"turn": 1, "user": "...", "agent": "..."},
    {"turn": 2, "user": "...", "agent": "...", "tool_calls": [{"name": "...", "args": {...}, "output": {...}}]}
  ]
}
```

### Per-turn fields

| Field | Type | Notes |
|---|---|---|
| `turn` | int | 1-indexed. |
| `user` | string | The user's message at this turn. Empty allowed (event-triggered turns). |
| `agent` | string | The agent's reply. Empty allowed (aborted replies). |
| `tool_calls` | array | Optional. List of `{name, args, output, error}` objects for any tool calls made during this turn. |

Tool calls are optional. If your transcripts don't have them, leave the field out — the pipeline degrades gracefully (replay still works; the judge has slightly less context).

### File naming

By convention the filename **stem** matches the `session_id`:

```
transcripts/
├── sess_001.jsonl    ← session_id = "sess_001"
├── sess_002.json     ← session_id = "sess_002"
└── 2025-04-15_a8c3.jsonl   ← session_id = "2025-04-15_a8c3"
```

If you can't match that convention, set `transcript_path` per-row in the CSV instead.

---

## A complete worked example

Here's a minimal `eval_data/` folder you can copy and adapt:

```
eval_data/
├── results.csv
├── transcripts/
│   ├── sess_001.jsonl
│   ├── sess_002.jsonl
│   ├── sess_003.jsonl
│   └── sess_004.jsonl
└── (your skills/ folder lives elsewhere, e.g. ../skills/)
```

`results.csv`:

```csv
session_id,skill,score,failure_category,failure_summary,failure_turn,transcript_path
sess_001,find-restaurant,fail,ignored_constraint,"User asked for vegan; agent suggested a steakhouse.",2,
sess_002,find-restaurant,fail,refused_reasonable_request,"Agent said 'I cannot help' for a normal restaurant lookup.",1,
sess_003,find-restaurant,pass,,,,
sess_004,book-table,fail,wrong_arguments,"Passed date as 05/05/2026 (MM/DD); tool wanted ISO.",3,
```

Run:

```bash
autoresearch run --adapter csv \
    --skills-root ./skills \
    --outputs-root ./outputs \
    --top-n 3
```

What happens:

1. CSV loaded → 2 Targets built (`find-restaurant` rank 0 with 2 fails, `book-table` rank 1 with 1 fail).
2. For each Target, the 8-step pipeline runs: program → propose → critic + replay → verdict.
3. Output lands at `outputs/<run_id>/<skill_name>/`:

```
outputs/run_2026-05-04_19-30-12/
├── summary.md
├── find-restaurant/
│   ├── program.md
│   ├── v_old.md
│   ├── v_new.md
│   ├── diff.txt
│   ├── critic.md
│   ├── replay.md
│   └── verdict.md
└── book-table/
    └── ...
```

`summary.md` is the top-level. Open it to see verdicts across all skills + paths to the per-skill detail.

---

## CLI reference

```bash
# Required
--adapter csv

# Skill resolution
--skills-root <dir>          (default: skills)
--skill-path-template <tmpl> (default: {root}/{name}/SKILL.md)

# Sampling
--top-n <int>                (default: 3)
--fix-sample <int>           (default: 3)
--baseline-sample <int>      (default: 3)

# Output
--outputs-root <dir>         (default: outputs)

# Sanity check without spending tokens
--dry-run
```

The CSV adapter doesn't have its own CLI flags — its file paths are configured via construction args. From the CLI, defaults apply (`./results.csv` + `./transcripts/`). To customize paths, use Python:

---

## Programmatic use

```python
from pathlib import Path
from agent_autoresearch.adapters.csv import CSVAdapter
from agent_autoresearch.pipeline import run_pipeline
from agent_autoresearch.core.skill_io import FilesystemSkillIO

adapter = CSVAdapter(
    results_csv="/data/myteam/labels.csv",
    transcripts_dir="/data/myteam/transcripts",
)
skill_io = FilesystemSkillIO(root="/code/agent/skills")

result = run_pipeline(
    adapter,
    skill_io=skill_io,
    top_n=5,
    outputs_root=Path("/tmp/autoresearch-runs"),
)

print(f"{result.n_accept} accepts, {result.n_reject} rejects")
for r in result.target_results:
    print(f"  {r.verdict.skill_name}: {r.verdict.label}")
```

The CLI is just a thin wrapper over `run_pipeline()`. Anything the CLI does, you can do directly from Python. Useful for embedding in tests, notebooks, or your own tooling.

---

## Common pitfalls

### Skill names don't match the disk

The CSV column `skill` must match exactly what's on disk under `skills/`. The library does **no** kebab↔snake conversion. If your CSV says `find-restaurant` but your folder is `find_restaurant`, the run will REJECT every target with "Skill not found".

Fix it at CSV-write time (or use a hand-edit pass). The `--skill-path-template` flag can help if your layout is unusual (`{root}/{category}/{name}.md` etc.) but it can't fix name mismatch.

### Transcript files use a different ID convention than the CSV

If your CSV says `session_id = abc-123` but your file is `transcripts/abc_123.jsonl`, the load will fail. Either align the two conventions, or fill in `transcript_path` per-row.

### Sessions appearing in both fix and baseline lists for the same skill

If you (or the LLM judge) emit two rows for the same `(session_id, skill)` pair with conflicting verdicts, the adapter currently treats them as separate entries — the session ends up in both `fix_session_ids` and `regression_baseline_ids` for that skill. Replay will then judge it twice with inconsistent expectations.

The judge dedupes per skill on the way in (fail wins over pass), so this only happens with hand-edits. Easy to spot in the CSV; easy to fix.

### `failure_turn` is the agent's failure turn, not the user's complaint turn

If a user complains on turn 5 about something the agent did on turn 3, set `failure_turn=3`. The replay step uses this to pick which turn to regenerate under the proposed skill change — it needs to point at the actual failure, not the downstream complaint.

### Score parsing is strict

Empty / unrecognized values raise `ValueError` rather than silently being treated as fail. The accepted spellings are documented above. This is deliberate — a typo shouldn't quietly drop a session out of the run.

---

## What this adapter is **not**

- **Not an eval framework.** It doesn't run your agent or compute scores. It assumes you (or `--adapter jsonl_judge`) already produced labels.
- **Not a session storage system.** Transcripts live as files; the adapter just reads them. If your sessions live in a database, write your own adapter.
- **Not for skill discovery.** It works on existing skills the agent already has prompts for. The improvement loop assumes a `skills/<name>/SKILL.md` to read and propose edits against. New-skill discovery is a different problem.

If any of those don't fit, look at [`docs/writing_an_adapter.md`](./writing_an_adapter.md) for the abstract `Adapter` interface — most non-CSV-shaped data sources are a couple-hundred-line adapter away.
