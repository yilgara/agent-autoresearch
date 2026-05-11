"""JSONL+judge adapter — for teams that have transcripts but no eval pipeline.

This is the **second** built-in adapter shape, alongside `csv.py`:

  - `csv.py`           — bring-your-own labels (you have a results.csv)
  - `jsonl_judge.py`   — bring-your-own transcripts (we LLM-judge them)

It is **not** a separate pipeline — it is a thin wrapper that:

  1. Runs `autoresearch.judging.judge_transcripts()` once to produce a
     labeled CSV from your raw JSONL transcripts.
  2. Delegates to a regular `CSVAdapter` from there on.

The reason for keeping the two shapes separate (rather than merging
them) is caching: the judge is one LLM call per session, and the CSV
is the cache. Re-running this adapter without `force_rejudge=True` is
free for sessions that have already been judged.

## Two ways to use it

### From the CLI

```bash
# zero-config: looks for ./transcripts/, ./skills/, writes ./results.csv
autoresearch run --adapter jsonl_judge --top-n 3
```

The judge runs the first time (one LLM call per transcript). Output
CSV gets cached so subsequent `autoresearch run` invocations are
free. To force re-judging, either delete `results.csv` and its
`.judged` sidecar, or construct the adapter with `force_rejudge=True`
from Python (see below).

To stop after judging — review the CSV before paying for the full
pipeline — pass `--dry-run`:

```bash
autoresearch run --adapter jsonl_judge --dry-run    # judges + summary, no pipeline
# review results.csv
autoresearch run --adapter jsonl_judge              # full run; no re-judging
```

### From Python

```python
from agent_autoresearch.adapters.jsonl_judge import JSONLJudgeAdapter
from agent_autoresearch.pipeline import run_pipeline

adapter = JSONLJudgeAdapter(
    transcripts_dir="/data/myteam/logs",
    skills_dir="/code/agent/skills",
    results_csv="/tmp/myteam_labels.csv",
    force_rejudge=False,
)
result = run_pipeline(adapter, top_n=5)
```

If you want more control over the judging step (custom LLM provider,
progress callbacks, custom skill list), call
`agent_autoresearch.judging.judge_transcripts()` directly and then
pass the resulting CSV to `CSVAdapter`. This wrapper is the convenience
path for the common case.
"""

from __future__ import annotations

from pathlib import Path

from agent_autoresearch.adapters.csv import CSVAdapter
from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import Conversation, Target
from agent_autoresearch.core.llm import LLMProvider
from agent_autoresearch.judging import judge_transcripts


class JSONLJudgeAdapter(Adapter):
    """Adapter for raw JSONL transcripts — judges sessions on first use, then
    behaves like `CSVAdapter`.

    Construction is cheap (no I/O). The judge runs lazily on the first
    call to `load_targets()` or `load_conversations()`. If the output
    CSV already exists, only newly-added transcripts are judged.
    """

    name = "jsonl_judge"

    def __init__(
        self,
        transcripts_dir: str | Path = "transcripts",
        skills_dir: str | Path = "skills",
        results_csv: str | Path = "results.csv",
        *,
        force_rejudge: bool = False,
        llm: LLMProvider | None = None,
    ):
        """
        Args:
            transcripts_dir: folder of `.jsonl` / `.json` session files.
                Default `./transcripts`. Filename stem becomes session_id.
            skills_dir: folder containing per-skill prompts. Subfolder
                names (or `.md` filenames) become the canonical skill
                list the judge attributes failures to. Default `./skills`.
            results_csv: where to read/write the labeled CSV. Default
                `./results.csv`. Acts as the judge cache — re-runs are
                free for sessions already in the file.
            force_rejudge: if True, re-judge every transcript even if
                a row exists. Useful when you've changed the prompt or
                the skill list. Default False.
            llm: optional LLMProvider for the judge calls. Defaults to
                Anthropic Sonnet via `default_llm_provider()`.
        """
        self.transcripts_dir = Path(transcripts_dir)
        self.skills_dir = Path(skills_dir)
        self.results_csv = Path(results_csv)
        self.force_rejudge = force_rejudge
        self._llm = llm

        # Lazy — judge + delegate built on first contract method call.
        self._delegate: CSVAdapter | None = None

    # ── Adapter contract ────────────────────────────────────────────────────

    def load_targets(self) -> list[Target]:
        return self._ensure_delegate().load_targets()

    def load_conversations(self) -> list[Conversation]:
        return self._ensure_delegate().load_conversations()

    # ── Internals ───────────────────────────────────────────────────────────

    def _ensure_delegate(self) -> CSVAdapter:
        """Run the judge once if needed, then return a cached `CSVAdapter`.

        Subsequent calls reuse the same delegate and never re-judge —
        if you want to refresh, construct a new adapter (or delete the
        results CSV).
        """
        if self._delegate is not None:
            return self._delegate

        skills = self._discover_skill_names()
        if not skills:
            raise FileNotFoundError(
                f"No skills found under {self.skills_dir}. Expected per-skill "
                "folders (with SKILL.md inside) or `*.md` files. The judge "
                "needs to know which skill names to attribute failures to."
            )

        if not self.transcripts_dir.exists():
            raise FileNotFoundError(
                f"Transcripts dir not found: {self.transcripts_dir}"
            )

        # If forcing a re-judge, wipe the cache files so judge_transcripts
        # treats every session as fresh. (Don't delete user's hand-edits
        # silently — only when force_rejudge is explicitly set.)
        if self.force_rejudge:
            self._clear_cache()

        # Run the judge — this is a no-op for sessions already in the CSV.
        judge_transcripts(
            transcripts_dir=self.transcripts_dir,
            skills=skills,
            output_csv=self.results_csv,
            llm=self._llm,
            skip_existing=True,
        )

        self._delegate = CSVAdapter(
            results_csv=self.results_csv,
            transcripts_dir=self.transcripts_dir,
        )
        return self._delegate

    def _discover_skill_names(self) -> list[str]:
        """List skill names under `self.skills_dir`.

        Two layouts supported:
          1. `<skills_dir>/<name>/SKILL.md` — Anthropic-style folders
          2. `<skills_dir>/<name>.md`       — flat `.md` files

        Hidden entries (`.` prefix) are skipped. Order is alphabetical
        for a stable judge prompt across runs.
        """
        if not self.skills_dir.exists():
            return []

        names: set[str] = set()
        for entry in self.skills_dir.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and (entry / "SKILL.md").exists():
                names.add(entry.name)
            elif entry.is_file() and entry.suffix.lower() == ".md":
                names.add(entry.stem)
        return sorted(names)

    def _clear_cache(self) -> None:
        """Remove the results CSV + judged-log sidecar. Best-effort."""
        for p in (
            self.results_csv,
            self.results_csv.parent / (self.results_csv.name + ".judged"),
        ):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
