"""CSV adapter — for teams that already eval their agent.

If your eval pipeline (or a teammate filling in a Google Sheet, or a
script you wrote, or Promptfoo, or DeepEval) already produces
pass/fail labels and a short reason for each failure, this is the
adapter you want. It does **no LLM call** — the library trusts your
labels.

If you only have raw transcripts and no labels yet, use
`jsonl_judge` instead — it runs an LLM judge to generate the labels
this adapter expects you to provide.

## File layout

By convention this adapter expects two things in the same folder:

    eval_data/
    ├── results.csv         # one row per session
    └── transcripts/        # one transcript per session
        ├── sess_001.jsonl
        ├── sess_002.jsonl
        └── ...

`transcripts_dir` and `results_csv` are both configurable.

## results.csv schema

Three required columns + four optional. Any extra columns are ignored.

| Column              | Required | Notes |
|---------------------|----------|-------|
| `session_id`        | yes      | Primary key. Must match the transcript filename stem. |
| `skill`             | yes      | The skill this session exercises. Use the same string the disk uses (no kebab/snake auto-conversion). |
| `score`             | yes      | `0`/`1` or `fail`/`pass` or `false`/`true`. |
| `failure_category`  | optional | Short label e.g. `wrong_tool`, `missing_filter`. Used as `Evidence.category`. |
| `failure_summary`   | optional | Free-text reason. Goes into `Evidence.details["summary"]`. |
| `failure_turn`      | optional | 1-indexed turn the failure happened on. Helps replay focus. |
| `transcript_path`   | optional | Override the default `<transcripts_dir>/<session_id>.{jsonl,json}` lookup. |

Pass rows can leave the failure-* columns blank.

## Transcript file shapes

Auto-detected by extension:

- **`.jsonl`** — one turn per line, each line a JSON object with
  `turn`, `user`, `agent`, optional `tool_calls`.
- **`.json`** — a single object with `session_id` + `turns: [...]`,
  same per-turn shape as JSONL.

Tool calls are optional. If your transcripts don't have them, leave
the field out — replay degrades gracefully.

## What targets look like after parsing

One `Target` per distinct skill. Within each target:

  - `fix_session_ids`         = sessions where score=fail
  - `regression_baseline_ids` = sessions where score=pass
  - `evidence`                = one `Evidence` per failing row that
                                 carried a `failure_summary`

Targets are ranked by failure count (more failures = lower `rank`,
i.e. higher priority).

## Example usage

    from agent_autoresearch.adapters.csv import CSVAdapter
    adapter = CSVAdapter(
        results_csv="eval_data/results.csv",
        transcripts_dir="eval_data/transcripts",
    )

Or via the CLI:

    autoresearch run --adapter csv --top-n 3

(Defaults are `./results.csv` + `./transcripts/`.)
"""

from __future__ import annotations

import csv as _csv
import json
from pathlib import Path
from typing import Any

from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import (
    Conversation,
    Evidence,
    Target,
    ToolCall,
    Turn,
)


# ── Score parsing ────────────────────────────────────────────────────────────

_PASS_TOKENS = {"1", "pass", "passed", "true", "ok", "success"}
_FAIL_TOKENS = {"0", "fail", "failed", "false", "ko", "failure"}


def _parse_score(raw: str) -> bool:
    """Return True if pass, False if fail. Raises on unknown values.

    Accepts the common spellings; case-insensitive. Empty / unrecognized
    values raise ValueError so a typo in the CSV doesn't silently
    drop sessions out of the run.
    """
    v = (raw or "").strip().lower()
    if v in _PASS_TOKENS:
        return True
    if v in _FAIL_TOKENS:
        return False
    raise ValueError(
        f"Unrecognized score value {raw!r}. "
        f"Expected one of {sorted(_PASS_TOKENS | _FAIL_TOKENS)}."
    )


# ── Transcript loading ──────────────────────────────────────────────────────

def _load_transcript(path: Path, session_id: str) -> Conversation:
    """Read a JSON or JSONL transcript and return a `Conversation`.

    Format auto-detected by suffix. Tool calls are optional per turn.
    Errors raise — the adapter would rather fail loudly than silently
    return a partially-loaded session.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Transcript for session {session_id!r} not found at {path}"
        )

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        turns = [_turn_from_dict(json.loads(line))
                 for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        turns = [_turn_from_dict(t) for t in obj.get("turns", [])]
    else:
        raise ValueError(
            f"Unsupported transcript extension {suffix!r} for {path}. "
            "Expected .jsonl or .json."
        )

    return Conversation(session_id=session_id, turns=turns)


def _turn_from_dict(d: dict[str, Any]) -> Turn:
    """Build a `Turn` from a dict — robust to missing optional fields."""
    raw_calls = d.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            name=c.get("name", ""),
            args=c.get("args"),
            output=c.get("output"),
            error=c.get("error"),
        )
        for c in raw_calls
    ]
    return Turn(
        turn=int(d.get("turn", 0)),
        user=d.get("user", "") or "",
        agent=d.get("agent", "") or "",
        tool_calls=tool_calls,
    )


# ── Adapter ──────────────────────────────────────────────────────────────────

class CSVAdapter(Adapter):
    """Adapter for pre-labeled eval results in a CSV + a transcripts folder.

    Construction is cheap; the actual file reads happen in
    `load_targets()` / `load_conversations()`. Typical usage is one
    adapter per `autoresearch run` invocation, so we don't bother
    caching across calls.
    """

    name = "csv"

    REQUIRED_COLUMNS = ("session_id", "skill", "score")

    def __init__(
        self,
        results_csv: str | Path = "results.csv",
        transcripts_dir: str | Path = "transcripts",
    ):
        """
        Args:
            results_csv: path to the CSV. Default `./results.csv`.
            transcripts_dir: folder containing transcript files.
                Default `./transcripts`. Lookup convention:
                `<transcripts_dir>/<session_id>.jsonl` first, then
                `.json`. Override per-row with the optional
                `transcript_path` column.
        """
        self.results_csv = Path(results_csv)
        self.transcripts_dir = Path(transcripts_dir)

        # Filled by load_targets / cached for load_conversations
        self._rows: list[dict[str, str]] | None = None

    # ── Adapter contract ────────────────────────────────────────────────────

    def load_targets(self) -> list[Target]:
        """Group CSV rows by skill, build one `Target` per skill.

        Targets are ordered by descending failure count, so `--top-n`
        on the CLI naturally picks the most-broken skills first.
        """
        rows = self._read_rows()

        # Group by skill
        by_skill: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_skill.setdefault(row["skill"], []).append(row)

        # Build a Target per skill
        targets: list[Target] = []
        for skill_name, skill_rows in by_skill.items():
            fix_ids: list[str] = []
            base_ids: list[str] = []
            evidence: list[Evidence] = []

            for row in skill_rows:
                sid = row["session_id"]
                passed = _parse_score(row["score"])
                if passed:
                    base_ids.append(sid)
                    continue

                fix_ids.append(sid)

                # Evidence — only built from rows that carry a summary
                summary = (row.get("failure_summary") or "").strip()
                if not summary:
                    continue

                category = (row.get("failure_category") or "failure").strip() or "failure"
                details: dict[str, Any] = {
                    "summary": summary,
                    "session_id": sid,
                }
                turn_raw = (row.get("failure_turn") or "").strip()
                if turn_raw:
                    try:
                        details["turn"] = int(turn_raw)
                    except ValueError:
                        # Non-integer turn is a CSV typo — keep it as the raw
                        # string so the LLM still sees something useful.
                        details["turn"] = turn_raw

                evidence.append(Evidence(category=category, details=details))

            targets.append(Target(
                skill_name=skill_name,
                evidence=evidence,
                fix_session_ids=fix_ids,
                regression_baseline_ids=base_ids,
            ))

        # Rank by failure count, descending — the CLI's --top-n will
        # naturally pick the most-broken skills first.
        targets.sort(key=lambda t: len(t.fix_session_ids), reverse=True)
        for i, t in enumerate(targets):
            t.rank = i

        return targets

    def load_conversations(self) -> list[Conversation]:
        """Load every transcript referenced by the CSV (pass + fail)."""
        rows = self._read_rows()
        out: list[Conversation] = []
        seen: set[str] = set()
        for row in rows:
            sid = row["session_id"]
            if sid in seen:
                continue
            seen.add(sid)
            path = self._resolve_transcript_path(row)
            out.append(_load_transcript(path, sid))
        return out

    # ── Internals ───────────────────────────────────────────────────────────

    def _read_rows(self) -> list[dict[str, str]]:
        """Read + validate the CSV, with caching across method calls.

        Validates required columns once. Per-row validation (score
        parsing, transcript existence) happens lazily where each
        field is consumed.
        """
        if self._rows is not None:
            return self._rows

        if not self.results_csv.exists():
            raise FileNotFoundError(
                f"results CSV not found: {self.results_csv}"
            )

        with self.results_csv.open(encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {self.results_csv} appears to be empty.")

            missing = [c for c in self.REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"CSV {self.results_csv} missing required columns: {missing}. "
                    f"Found: {list(reader.fieldnames)}."
                )

            rows = [r for r in reader if (r.get("session_id") or "").strip()]

        self._rows = rows
        return rows

    def _resolve_transcript_path(self, row: dict[str, str]) -> Path:
        """Pick the transcript file for one row.

        Precedence:
          1. Explicit `transcript_path` column (if set), resolved
             relative to the CSV's directory if not absolute.
          2. `<transcripts_dir>/<session_id>.jsonl`
          3. `<transcripts_dir>/<session_id>.json`
        """
        explicit = (row.get("transcript_path") or "").strip()
        if explicit:
            p = Path(explicit)
            if not p.is_absolute():
                p = self.results_csv.parent / p
            return p

        sid = row["session_id"]
        for ext in (".jsonl", ".json"):
            p = self.transcripts_dir / f"{sid}{ext}"
            if p.exists():
                return p
        # Return the .jsonl candidate — _load_transcript will raise
        # FileNotFoundError with a clear message.
        return self.transcripts_dir / f"{sid}.jsonl"
