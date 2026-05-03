# Writing an adapter

This guide walks through plugging your own evaluation pipeline into
autoresearch. Start with [pipeline.md](./pipeline.md) for the
architectural overview; this is the practical "how do I wire this in"
companion.

If you only have 10 minutes, jump to [Quickstart](#quickstart) — it
covers the minimum viable adapter end to end.

---

## What an adapter is

autoresearch is agnostic about where your eval signal comes from.
Your eval pipeline could be Braintrust, GCP Cloud Logs, a
custom SQLite database, a JSON file someone emails you, or a REST
API — the library doesn't care.

An **adapter** is the thin translation layer between your specific
output format and autoresearch's neutral data classes. Once you've
written the adapter, every downstream step (program → propose →
critic → replay → verdict) works without modification.

Three classes are involved:

| Class | What it does | Do you need to implement? |
|---|---|---|
| **`Adapter`** | Translates your eval output into `Target` and `Conversation` objects | **Yes — always.** This is the main thing. |
| **`SkillIO`** | Loads & writes skill prompt files | Probably not — `FilesystemSkillIO` (default) covers most layouts |
| **`LLMProvider`** | Calls the LLM API | Probably not — Anthropic Sonnet is the default |

In practice, **most teams only write the `Adapter`**.

---

## Quickstart

We'll wire autoresearch into a fictional eval system called **MyEval**
that writes one JSON file per nightly run.

### 1. The eval output you have to parse

Pretend `MyEval` writes `eval_runs/run_<id>.json` after each nightly,
with this shape:

```json
{
  "run_id": "2026-05-04",
  "broken_skills": [
    {
      "skill_name": "find-restaurant",
      "n_failures": 12,
      "failure_summary": "Agent kept booking wrong restaurant when user said 'the second one'",
      "failed_session_ids": ["abc123", "def456", "ghi789"],
      "passed_session_ids": ["xyz111", "uvw222", "rst333", "pqr444"]
    },
    {
      "skill_name": "send-email",
      "n_failures": 3,
      "failure_summary": "Agent attached the wrong document",
      "failed_session_ids": ["mno555"],
      "passed_session_ids": ["jkl666", "ghi777"]
    }
  ],
  "sessions": [
    {
      "id": "abc123",
      "transcript": [
        {"role": "user", "text": "Find me Italian"},
        {"role": "agent", "text": "Where would you like to eat?"},
        {"role": "user", "text": "Amsterdam center"},
        {"role": "agent", "text": "Here are 5 options..."},
        {"role": "user", "text": "Book the second one"},
        {"role": "agent", "text": "Booked at Bistro Verde"}
      ]
    }
  ]
}
```

### 2. The adapter

Save as `myeval_adapter.py`:

```python
import json
from pathlib import Path
from agent_autoresearch import Adapter, Target, Conversation, Evidence


class MyEvalAdapter(Adapter):
    name = "myeval"

    def __init__(self, run_id: str, eval_dir: Path = Path("eval_runs")):
        self.run_id = run_id
        self.data = json.loads((eval_dir / f"run_{run_id}.json").read_text())

    def load_targets(self) -> list[Target]:
        return [
            Target(
                skill_name=skill["skill_name"],
                evidence=[
                    Evidence(
                        category="failure_pattern",
                        details={"summary": skill["failure_summary"],
                                 "n_failures": skill["n_failures"]},
                    )
                ],
                fix_session_ids=skill["failed_session_ids"],
                regression_baseline_ids=skill["passed_session_ids"],
            )
            for skill in self.data["broken_skills"]
        ]

    def load_conversations(self) -> list[Conversation]:
        return [
            Conversation(
                session_id=s["id"],
                turns=self._convert_turns(s["transcript"]),
            )
            for s in self.data["sessions"]
        ]

    @staticmethod
    def _convert_turns(transcript):
        # Pair user/agent messages into turns
        turns = []
        i = 0
        while i < len(transcript):
            user_msg = transcript[i]["text"] if transcript[i]["role"] == "user" else ""
            agent_msg = transcript[i + 1]["text"] if i + 1 < len(transcript) and transcript[i + 1]["role"] == "agent" else ""
            turns.append({"turn": len(turns) + 1, "user": user_msg, "agent": agent_msg})
            i += 2
        return turns
```

### 3. Run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...
autoresearch run --adapter myeval --run-id 2026-05-04
```

That's it. autoresearch parses your JSON via the adapter, builds
targets, runs the propose → critic → replay → verdict pipeline, and
writes outputs to `outputs/run_<ts>/find-restaurant/` etc.

---

## The data classes

What your adapter populates.

### `Target`

One per skill the pipeline should consider improving.

```python
@dataclass
class Target:
    skill_name: str                                     # the skill identifier
    evidence: list[Evidence]                            # your findings (see below)
    fix_session_ids: list[str]                          # sessions where this skill broke
    regression_baseline_ids: list[str] = field(default_factory=list)
                                                         # sessions where it worked (for regression check)
    rank: int = 0                                       # optional: your priority order
```

Notes:
- `skill_name` must match a skill that `SkillIO.load(name)` can find.
- `fix_session_ids` should reference sessions returned by
  `load_conversations()`. If an ID is missing from the conversation
  set, replay skips that session for that target.
- `regression_baseline_ids` is optional but **strongly recommended**.
  Without baselines, the regression score is undefined and verdicts
  default to HUMAN_REVIEW.

### `Conversation`

One per session your replay phase will need.

```python
@dataclass
class Conversation:
    session_id: str                                     # primary key
    turns: list[Turn]                                   # ordered list of exchanges
    metadata: dict = field(default_factory=dict)        # anything else (env, channel, …)


@dataclass
class Turn:
    turn: int                                           # 1-indexed
    user: str                                           # what the user said
    agent: str                                          # what the agent replied
    tool_calls: list[ToolCall] = field(default_factory=list)
                                                         # optional, for richer replay
```

A "turn" in autoresearch is **one user message paired with one agent
response** — a single back-and-forth exchange. If your transcript is
flat ("user", "agent", "user", "agent", ...) you'll need to pair them
up; the quickstart adapter shows this pattern.

### `Evidence`

Free-form structured failure data. autoresearch passes it to the LLM
in step 4 (`build_program`) as part of the strategy generation.

```python
@dataclass
class Evidence:
    category: str                                       # your label, e.g. "wrong_information", "tool_error"
    details: dict                                       # whatever fields are useful — kept as JSON
    confidence: float | None = None                     # optional 0..1
```

The LLM sees this as JSON. Common shapes:

```python
# Pattern 1: short summary + a quote
Evidence(category="wrong_information",
         details={"summary": "Agent claimed 9pm showing, tool returned 8:30pm",
                  "quote": "Pathé Tuschinski has a 9pm Marvel show"})

# Pattern 2: structured rule violation
Evidence(category="step_violation",
         details={"step": "3", "rule": "must call get_extended_profile first",
                  "what_agent_did": "skipped step 3 entirely"})

# Pattern 3: pure narrative
Evidence(category="general",
         details={"summary": "Multiple users reported the agent forgot context across turns."})
```

There's no schema — `details` is just a dict the LLM reads. Make it
specific enough to ground the strategy.

---

## Where the adapter plugs in

### Option 1 — Local module (simplest)

For internal use, just put `myeval_adapter.py` in your working
directory and pass `--adapter myeval_adapter:MyEvalAdapter` on the
CLI. autoresearch imports the class by path.

### Option 2 — Installable Python package

For sharing across projects, register the adapter via Python entry
points. In your `pyproject.toml`:

```toml
[project.entry-points."agent_autoresearch.adapters"]
myeval = "myeval_adapter:MyEvalAdapter"
```

Now `autoresearch run --adapter myeval` works after `pip install
your-package`. autoresearch's CLI discovers the adapter by name.

---

## What you get for free

### `FilesystemSkillIO` (default)

Loads & writes skill files matching a configurable path template.
Defaults to `skills/<name>/SKILL.md`.

```python
from agent_autoresearch import FilesystemSkillIO

# default — skills live at skills/<name>/SKILL.md
skill_io = FilesystemSkillIO(root="skills")

# custom layout — flat files, e.g. skills/<category>/<name>.md
skill_io = FilesystemSkillIO(path_template="skills/{category}/{name}.md")
```

If your skills live somewhere exotic (S3, a DB, a git API), implement
your own `SkillIO` subclass and pass it to the CLI via `--skill-io`.
Most teams never need to.

### `AnthropicLLMProvider` (default)

Wraps Sonnet 4.5 via the Anthropic SDK. Reads `ANTHROPIC_API_KEY` from
the environment. Subclass `LLMProvider` if you want to swap to OpenAI,
Bedrock, OpenRouter, etc.

---

## Common patterns

### Pattern A — File-based eval

The MyEval quickstart above. Best when your eval pipeline produces
artifacts (JSON, MD, JSONL) that get uploaded as GHA artifacts or
landed on S3.

```python
class MyAdapter(Adapter):
    def __init__(self, run_id):
        self.data = json.loads(Path(f"runs/{run_id}.json").read_text())
    ...
```

### Pattern B — Database-backed eval

If your eval results live in Postgres / SQLite / DynamoDB, query in
the constructor or lazy-load in `load_targets()`.

```python
import sqlite3

class DBAdapter(Adapter):
    def __init__(self, run_id, db_path="eval.db"):
        self.run_id = run_id
        self.conn = sqlite3.connect(db_path)

    def load_targets(self):
        rows = self.conn.execute(
            "SELECT skill_name, summary, failed_ids, passed_ids "
            "FROM eval_findings WHERE run_id = ?", (self.run_id,)
        ).fetchall()
        return [Target(...) for row in rows]
```

### Pattern C — API-backed eval

If you fetch eval results from a service (Braintrust, your own REST
endpoint), wrap that in the constructor.

```python
import requests

class APIAdapter(Adapter):
    def __init__(self, run_id, api_url, token):
        resp = requests.get(f"{api_url}/runs/{run_id}",
                            headers={"Authorization": f"Bearer {token}"})
        self.data = resp.json()
```

---

## Testing your adapter

Two layers of testing we recommend.

### 1. Unit test against synthetic data

Hand-craft a tiny fake input that exercises edge cases:

```python
def test_adapter_handles_skill_with_no_baselines():
    fake_data = {
        "broken_skills": [
            {"skill_name": "x", "n_failures": 1,
             "failure_summary": "...", "failed_session_ids": ["a"],
             "passed_session_ids": []},                  # no baselines
        ],
        "sessions": [{"id": "a", "transcript": []}],
    }
    adapter = MyEvalAdapter.from_dict(fake_data)         # add a from_dict helper
    targets = adapter.load_targets()
    assert len(targets) == 1
    assert targets[0].regression_baseline_ids == []      # adapter passes empty through
```

### 2. Dry-run on real data

Before burning LLM tokens, verify the adapter parses correctly:

```bash
autoresearch run --adapter myeval --run-id 2026-05-04 --dry-run
```

`--dry-run` runs Phase A only (parsing + target build) and prints
the targets. No LLM calls.

---

## Common pitfalls

### Session IDs must match

`fix_session_ids` and `regression_baseline_ids` reference sessions
that `load_conversations()` returns. If a target's `fix_session_ids`
contains an ID that's not in the conversations set, replay skips
that session silently — your fix score then comes from a smaller-
than-expected sample.

The adapter is responsible for keeping these consistent.

### Empty `regression_baseline_ids` defaults to HUMAN_REVIEW

If a target has no baseline sessions, `regression_score` is undefined
and the verdict logic can't ACCEPT (we have nothing to verify the
edit doesn't regress). It defaults to HUMAN_REVIEW. To get clean
ACCEPTs, your adapter must surface enough passing sessions per
target.

### Evidence categories should be stable strings

The LLM uses `evidence[*].category` as a label when discussing the
failure. Stable, descriptive strings (e.g. `"wrong_tool_call"`,
`"missing_step"`) help the LLM reason; freeform strings or English
sentences make the strategy doc messy.

### Don't paraphrase the agent's words

When constructing `Conversation.turns`, copy the user/agent text
**verbatim** from your logs. Replay needs the actual words, not a
summary. If your eval system stores summaries, fall back to a no-op
adapter for replay (`load_conversations` returns empty) — autoresearch
will skip the validation phase and verdicts default to NO_VALIDATION.

---

## Reference: built-in adapters

Both ship in the library — read their source for full examples.

- **[`skilleval`](../agent_autoresearch/adapters/skilleval.py)** — for
  the [skilleval](https://github.com/yilgara/skilleval) eval pipeline
  (myT, flume). Parses markdown reports + a JSONL transcript sidecar.
  ~150 lines.
- **[`synthetic`](../agent_autoresearch/adapters/synthetic.py)** — generates
  fake targets and conversations for testing the pipeline without any
  real eval data. Useful as a tutorial reference.

---

## What's NOT covered here

- **Custom LLM providers** — separate doc when we add v0.3
  multi-provider support.
- **Custom strategies** — overriding the prompts is possible
  (`--strategy v2`) but considered advanced; see `agent_autoresearch/
  strategies/v1.py` for the structure.
- **GHA workflow integration** — example workflows for running
  autoresearch in CI live in [`docs/ci_examples.md`](./ci_examples.md)
  (TODO).

---

## Getting help

Open an issue at
[github.com/yilgara/agent-autoresearch](https://github.com/yilgara/agent-autoresearch)
with:

1. Which eval system you're trying to wire in
2. The shape of one input file or DB row (sanitised)
3. What you've tried so far

Adapter contributions to the built-in set are welcome — if your eval
system is reasonably common (Braintrust, LangSmith, OpenTelemetry
spans, etc.), we'll merge a reference adapter that everyone benefits
from.
