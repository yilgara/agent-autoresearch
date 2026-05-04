"""LLM-driven session judging — turn raw transcripts into a labeled CSV.

This module is the "Tier 0" entry point for users who only have
agent conversation logs and no eval pipeline yet. Point it at a
folder of JSONL/JSON transcripts; it will:

  1. Load each session
  2. Ask an LLM to judge pass/fail + attribute failures to one of
     your known skills
  3. Write a `results.csv` in the schema the CSV adapter consumes

After this runs you have everything `CSVAdapter` needs:

    autoresearch judge --transcripts ./logs --skills ./skills --out ./results.csv
    autoresearch run --adapter csv --top-n 3

Why a separate command (not an adapter): the labels are expensive
to produce (one LLM call per session) and worth caching. The CSV
becomes the cache; subsequent `autoresearch run` invocations are
free. The user can also open the CSV in a spreadsheet and fix
labels the LLM got wrong before running propose+critic+replay.

This module exposes one function — `judge_transcripts()` — and a
small report dataclass. The CLI subcommand is just a thin wrapper.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_autoresearch._prompts import format_prompt
from agent_autoresearch.core.data import Conversation, ToolCall, Turn
from agent_autoresearch.core.llm import LLMProvider, default_llm_provider


_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_session.md"

# Token cap — verdict + 1-2 sentence reasoning, never long.
JUDGE_MAX_TOKENS = 500

# Truncation budget when rendering one session for the judge.
# Smaller than replay's transcript budget; the judge only needs
# enough to verdict, not to imagine a hypothetical reply.
JUDGE_TRANSCRIPT_MAX_CHARS = 10_000
JUDGE_USER_MAX_CHARS       = 1200
JUDGE_REPLY_MAX_CHARS      = 1500
JUDGE_TOOL_OUTPUT_CHARS    = 400
JUDGE_TOOL_ARGS_CHARS      = 200


# CSV columns we emit — matches CSVAdapter's expected schema exactly.
CSV_COLUMNS = (
    "session_id",
    "skill",
    "score",
    "failure_category",
    "failure_summary",
    "failure_turn",
    "transcript_path",
)


# ── Result types ────────────────────────────────────────────────────────────

@dataclass
class JudgeReport:
    """Summary of one `judge_transcripts()` call.

    Counts split into session-level and row-level because one session
    can produce multiple CSV rows (one per skill exercised).

    Session-level:
      - `n_transcripts`        files found in the transcripts dir
      - `n_judged`             sessions where the judge ran successfully
                                 (regardless of how many judgments came back)
      - `n_skipped_existing`   sessions skipped because they were already
                                 in the CSV / judged-log sidecar
      - `n_unrelated`          judged sessions that produced zero
                                 valid skill attributions
      - `n_errors`             LLM/parse failures (logged to .errors.log)

    Row-level:
      - `n_rows_written`       total CSV rows added this run
      - `n_pass`               of those, pass rows
      - `n_fail`               of those, fail rows
    """
    n_transcripts: int
    n_judged: int
    n_skipped_existing: int
    n_unrelated: int
    n_errors: int
    n_rows_written: int
    n_pass: int
    n_fail: int
    output_csv: Path


# ── Public entry point ──────────────────────────────────────────────────────

ProgressHook = Callable[[str, int, int], None]   # (session_id, current, total)


def judge_transcripts(
    *,
    transcripts_dir: Path | str,
    skills: list[str],
    output_csv: Path | str,
    llm: LLMProvider | None = None,
    skip_existing: bool = True,
    on_progress: ProgressHook | None = None,
) -> JudgeReport:
    """Judge every transcript in `transcripts_dir`, append rows to `output_csv`.

    Args:
        transcripts_dir: folder of `.jsonl` / `.json` session files.
            Filename stem becomes the session_id.
        skills: list of known skill names. The judge attributes
            failures to exactly one of these (or `none` if the
            session doesn't exercise any). Case-sensitive — pass the
            same names your disk uses.
        output_csv: destination CSV. Created if missing; appended to
            (deduplicated by session_id) if it already exists.
        llm: provider for the judge LLM. Defaults to Anthropic Sonnet
            via `default_llm_provider()`.
        skip_existing: if True (default), session_ids already present
            in `output_csv` are skipped. Set False to force re-judging.
        on_progress: optional hook called once per session before the
            LLM call. Useful for the CLI's Rich progress bar.

    Returns:
        `JudgeReport` summarizing the run. Raises only on truly
        unrecoverable errors (missing transcripts dir, malformed
        skills list); per-session failures are caught and counted
        in `n_errors`.
    """
    transcripts_dir = Path(transcripts_dir)
    output_csv = Path(output_csv)

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts dir not found: {transcripts_dir}")
    if not skills:
        raise ValueError(
            "`skills` list is empty. The judge needs to know which "
            "skill names to attribute failures to."
        )

    llm = llm or default_llm_provider()

    transcript_paths = sorted(_list_transcripts(transcripts_dir))
    n_total = len(transcript_paths)

    # Load already-judged session_ids if appending. We track *every*
    # judged session — including skill=none rows that don't make it
    # into the CSV — via a sidecar log, so re-runs don't re-pay for
    # off-topic sessions.
    judged_log = _judged_log_path(output_csv)
    already_done: set[str] = set()
    if skip_existing:
        already_done |= _read_existing_session_ids(output_csv)
        already_done |= _read_judged_log(judged_log)

    rows_to_write: list[dict[str, str]] = []
    n_judged = n_unrelated = n_errors = 0
    n_skipped_existing = 0
    n_pass = n_fail = 0

    for i, path in enumerate(transcript_paths, 1):
        sid = path.stem
        if on_progress:
            on_progress(sid, i, n_total)

        if sid in already_done:
            n_skipped_existing += 1
            continue

        try:
            conv = _load_transcript(path, sid)
        except Exception as exc:  # noqa: BLE001 — surface in report, keep going
            n_errors += 1
            _record_error(output_csv, sid, f"load failed: {type(exc).__name__}: {exc}")
            continue

        try:
            judgments = _judge_one(conv=conv, skills=skills, llm=llm)
        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            _record_error(output_csv, sid, f"judge failed: {type(exc).__name__}: {exc}")
            continue

        # Record this session as judged regardless of outcome —
        # protects the next run from re-paying for unrelated sessions.
        _append_judged(judged_log, sid)
        n_judged += 1

        if not judgments:
            n_unrelated += 1
            continue

        # One CSV row per judgment (one per skill the session exercised)
        for j in judgments:
            rows_to_write.append({
                "session_id":       sid,
                "skill":            j["skill"],
                "score":            j["score"],
                "failure_category": j["category"],
                "failure_summary":  j["summary"],
                "failure_turn":     j["turn"],
                "transcript_path":  str(path.resolve()),
            })
            if j["score"] == "pass":
                n_pass += 1
            else:
                n_fail += 1

    if rows_to_write:
        _append_rows(output_csv, rows_to_write)

    return JudgeReport(
        n_transcripts=n_total,
        n_judged=n_judged,
        n_skipped_existing=n_skipped_existing,
        n_unrelated=n_unrelated,
        n_errors=n_errors,
        n_rows_written=len(rows_to_write),
        n_pass=n_pass,
        n_fail=n_fail,
        output_csv=output_csv,
    )


# ── Per-session judge call ──────────────────────────────────────────────────

def _judge_one(
    *,
    conv: Conversation,
    skills: list[str],
    llm: LLMProvider,
) -> list[dict[str, str]]:
    """Call the judge LLM on one session; return one dict per skill exercised.

    Each dict has keys: `score`, `skill`, `category`, `turn`, `summary`.
    Returns an empty list when the session exercises no listed skills
    (the LLM emitted zero `<judgement>` blocks, or all blocks were
    `skill=none`).
    """
    skills_block = "\n".join(f"- `{s}`" for s in skills)
    transcript = _render_transcript_for_judge(conv)

    system, user = format_prompt(
        _PROMPT_PATH,
        skills_block=skills_block,
        session_id=conv.session_id,
        transcript=transcript,
    )
    resp = llm.call(system=system, user=user, max_tokens=JUDGE_MAX_TOKENS)
    return _parse_judge_response(resp.text, allowed_skills=skills)


# ── Response parsing ────────────────────────────────────────────────────────

_TAG_RE_CACHE: dict[str, re.Pattern] = {}

_JUDGEMENT_BLOCK_RE = re.compile(
    r"<judgement\s*>\s*(.*?)\s*</judgement\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _xml_tag(name: str) -> re.Pattern:
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(
            rf"<{name}\s*>\s*(.*?)\s*</{name}\s*>",
            re.DOTALL | re.IGNORECASE,
        )
    return _TAG_RE_CACHE[name]


def _extract(text: str, tag: str) -> str:
    m = _xml_tag(tag).search(text)
    return (m.group(1).strip() if m else "")


def _parse_judge_response(
    raw: str,
    *,
    allowed_skills: list[str],
) -> list[dict[str, str]]:
    """Extract every `<judgement>` block, validate, dedupe by skill.

    Returns one dict per distinct skill the judge attributed. Empty
    list means the session exercises no listed skills (or the LLM
    output was unparseable — defensive: prefer skipping over
    flagging a hallucinated skill).

    Dedup rule: if the LLM emits two judgments for the same skill
    (shouldn't happen but tolerated), the **fail** one wins so we
    don't lose signal; otherwise the first one wins.
    """
    allowed_lower = {s.lower(): s for s in allowed_skills}
    blocks = _JUDGEMENT_BLOCK_RE.findall(raw)

    by_skill: dict[str, dict[str, str]] = {}
    for block in blocks:
        skill_raw = _extract(block, "skill").strip().lower()

        # Skip explicit "none" markers
        if skill_raw in ("", "none", "n/a", "(none)"):
            continue

        # Reject skills outside the allowed list — protects against
        # the model hallucinating a skill name that doesn't exist on disk
        if skill_raw not in allowed_lower:
            continue

        verdict = _extract(block, "verdict").strip().lower()
        score = "fail" if verdict == "fail" else "pass"

        category = _extract(block, "category").strip()
        turn = _extract(block, "turn").strip()
        summary = _extract(block, "summary").strip()

        # Pass blocks shouldn't carry failure metadata
        if score == "pass":
            category = ""
            turn = ""
            summary = ""

        # Validate turn is a positive integer if provided
        if turn and not turn.isdigit():
            turn = ""

        skill = allowed_lower[skill_raw]
        new_row = {
            "score":    score,
            "skill":    skill,
            "category": category,
            "turn":     turn,
            "summary":  summary,
        }

        # Dedup — fail wins over pass
        existing = by_skill.get(skill)
        if existing is None or (existing["score"] == "pass" and score == "fail"):
            by_skill[skill] = new_row

    return list(by_skill.values())


# ── Transcript I/O ──────────────────────────────────────────────────────────

def _list_transcripts(d: Path) -> list[Path]:
    """All `.jsonl` and `.json` files directly under `d`."""
    return [p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in (".jsonl", ".json")]


def _load_transcript(path: Path, session_id: str) -> Conversation:
    """Mirror of CSVAdapter's transcript loader; kept local to avoid coupling."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        turns = [_turn_from_dict(json.loads(line))
                 for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        turns = [_turn_from_dict(t) for t in obj.get("turns", [])]
    else:
        raise ValueError(f"Unsupported extension {suffix!r} for {path}")
    return Conversation(session_id=session_id, turns=turns)


def _turn_from_dict(d: dict[str, Any]) -> Turn:
    raw_calls = d.get("tool_calls") or []
    return Turn(
        turn=int(d.get("turn", 0)),
        user=d.get("user", "") or "",
        agent=d.get("agent", "") or "",
        tool_calls=[
            ToolCall(
                name=c.get("name", ""),
                args=c.get("args"),
                output=c.get("output"),
                error=c.get("error"),
            )
            for c in raw_calls
        ],
    )


def _render_transcript_for_judge(conv: Conversation) -> str:
    """Compact full-session render for the judge.

    Simpler than v1's replay transcript renderer: the judge looks at
    the whole session at once, so we don't need a focus-turn marker.
    """
    lines: list[str] = []
    for turn in conv.turns:
        lines.append(f"### Turn {turn.turn}")
        if turn.user:
            lines.append(f"user: {_truncate(turn.user, JUDGE_USER_MAX_CHARS)}")
        for tc in turn.tool_calls:
            args = _short_json(tc.args, JUDGE_TOOL_ARGS_CHARS)
            if tc.error:
                result = f"ERROR: {_truncate(str(tc.error), 200)}"
            else:
                result = _short_json(tc.output, JUDGE_TOOL_OUTPUT_CHARS) or "(no output)"
            lines.append(f"[tool] {tc.name}({args}) → {result}")
        if turn.agent:
            lines.append(f"agent: {_truncate(turn.agent, JUDGE_REPLY_MAX_CHARS)}")
        lines.append("")

    transcript = "\n".join(lines).rstrip()
    if len(transcript) > JUDGE_TRANSCRIPT_MAX_CHARS:
        head = transcript[: JUDGE_TRANSCRIPT_MAX_CHARS // 3]
        tail = transcript[-(2 * JUDGE_TRANSCRIPT_MAX_CHARS // 3):]
        transcript = (
            head
            + "\n\n[… session truncated; middle turns omitted …]\n\n"
            + tail
        )
    return transcript


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _short_json(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = str(value)
    return _truncate(s, limit)


# ── CSV I/O ─────────────────────────────────────────────────────────────────

def _read_existing_session_ids(csv_path: Path) -> set[str]:
    """Read the session_id column from an existing CSV, ignoring errors."""
    try:
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {(r.get("session_id") or "").strip()
                    for r in reader
                    if (r.get("session_id") or "").strip()}
    except Exception:
        return set()


def _append_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows to an existing CSV, or create a new one with the header."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _judged_log_path(csv_path: Path) -> Path:
    """Sidecar tracking every session_id ever judged (incl. skill=none)."""
    return csv_path.parent / (csv_path.name + ".judged")


def _read_judged_log(log_path: Path) -> set[str]:
    """Read the sidecar log; returns empty set if missing or unreadable."""
    if not log_path.exists():
        return set()
    try:
        return {line.strip()
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    except Exception:
        return set()


def _append_judged(log_path: Path, session_id: str) -> None:
    """Append one session_id to the judged-log sidecar; never raises."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{session_id}\n")
    except Exception:
        pass


def _record_error(csv_path: Path, session_id: str, message: str) -> None:
    """Best-effort error sidecar — never raises; helps the user spot bad sessions.

    Writes one line per error to `<output_csv>.errors.log` next to the
    main CSV. The CSV itself stays clean.
    """
    try:
        log_path = csv_path.parent / (csv_path.name + ".errors.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{session_id}\t{message}\n")
    except Exception:
        pass


__all__ = [
    "judge_transcripts",
    "JudgeReport",
    "JUDGE_MAX_TOKENS",
    "CSV_COLUMNS",
]
