"""Shared helpers used across v1 stages.

Private to the v1 strategy; not part of the public API. v2 will get
its own copy if/when it forks — keeps each version self-contained.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_autoresearch.core.data import Conversation, Evidence, ToolCall


# ── XML-tag extraction (propose + critic + responder + judge parsers) ───────

_TAG_RE_CACHE: dict[str, re.Pattern] = {}


def xml_tag(name: str) -> re.Pattern:
    """Compile (and cache) a tolerant regex for `<name>...</name>` blocks."""
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(
            rf"<{name}\s*>\s*(.*?)\s*</{name}\s*>",
            re.DOTALL | re.IGNORECASE,
        )
    return _TAG_RE_CACHE[name]


def extract_tag(text: str, name: str) -> str | None:
    """Pull the contents of one XML tag from a free-form string. Returns
    None if the tag is missing — callers fall back to defaults."""
    m = xml_tag(name).search(text)
    return m.group(1).strip() if m else None


# ── Output cleaning (build_program parser) ──────────────────────────────────

def strip_chatter(text: str) -> str:
    """Strip optional code fences + leading prose before the first `# H1`.

    The prompts ask the model to start with a specific heading, but
    Sonnet sometimes wraps the output in ```markdown ... ``` or adds
    a preamble like "Here's the program:". This normalises both.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[: -3].rstrip()
    idx = text.find("\n# ")
    if idx > 0 and not text.startswith("# "):
        text = text[idx + 1:]
    return text.strip()


# ── Replay-stage helpers (responder + judge + soft_replay) ──────────────────

# Truncation limits for fields fed into the replay LLMs. Conversations
# can have multi-thousand-char user messages or agent replies; these
# limits keep replay calls bounded without losing the gist.
USER_MAX_CHARS         = 1500
REPLY_MAX_CHARS        = 2000
TRANSCRIPT_MAX_CHARS   = 15000
TOOL_OUTPUT_MAX_CHARS  = 600
TOOL_ARGS_MAX_CHARS    = 250


def truncate(s: str, n: int) -> str:
    """Trim with an ellipsis if longer than `n` chars."""
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def pick_focus_turn(
    conversation: Conversation,
    evidence: Evidence | None = None,
) -> int:
    """Decide which turn of the session to focus the replay on (1-indexed).

    Priority:
      1. `evidence.details["focus_turn"]` if the adapter set it
      2. Last turn of the session (most failures bubble up to the
         agent's final user-visible reply)

    Adapter authors who want session-specific focus can tag evidence
    with `details={"session_id": "...", "focus_turn": 3}` and the
    replay will pick that turn. Otherwise the default works fine.
    """
    n_turns = len(conversation.turns)
    if n_turns == 0:
        return 1

    if evidence is not None:
        focus = (evidence.details or {}).get("focus_turn")
        if focus:
            try:
                return min(int(focus), n_turns)
            except (TypeError, ValueError):
                pass

    return n_turns


def _short_json(value: Any, limit: int) -> str:
    """JSON-dump a value, truncating the rendered string at `limit`."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = str(value)
    return truncate(s, limit)


def _format_tool_call(tc: ToolCall) -> str:
    """One tool call rendered for the transcript."""
    args_str = _short_json(tc.args, TOOL_ARGS_MAX_CHARS)
    if tc.error:
        result = f"ERROR: {truncate(str(tc.error), 200)}"
    else:
        result = _short_json(tc.output, TOOL_OUTPUT_MAX_CHARS) or "(no output)"
    return f"[tool] {tc.name}({args_str}) → {result}"


def format_session_transcript(
    conversation: Conversation,
    *,
    focus_turn: int,
) -> str:
    """Render the whole session as a readable transcript.

    Marks the focus turn with `← FOCUS` so the responder/judge know
    which turn is the one being replayed. Tool calls are interleaved
    verbatim (with output truncation per `TOOL_OUTPUT_MAX_CHARS`).
    Long transcripts are clipped at `TRANSCRIPT_MAX_CHARS` (head + tail
    kept; middle dropped).
    """
    lines: list[str] = []
    for turn in conversation.turns:
        idx = turn.turn
        marker = "  ← FOCUS" if idx == focus_turn else ""
        lines.append(f"### Turn {idx}{marker}")
        if turn.user:
            lines.append(f"user: {truncate(turn.user, USER_MAX_CHARS)}")
        for tc in turn.tool_calls:
            lines.append(_format_tool_call(tc))
        if turn.agent:
            agent_label = "agent (original reply)" if idx == focus_turn else "agent"
            lines.append(f"{agent_label}: {truncate(turn.agent, REPLY_MAX_CHARS)}")
        lines.append("")  # blank between turns

    transcript = "\n".join(lines).rstrip()
    if len(transcript) > TRANSCRIPT_MAX_CHARS:
        # Head + tail; drop the middle. Preserves "what they were
        # doing" + "where it broke" for long sessions.
        head = transcript[: TRANSCRIPT_MAX_CHARS // 3]
        tail = transcript[-(2 * TRANSCRIPT_MAX_CHARS // 3):]
        transcript = (
            head
            + "\n\n[… session truncated; middle turns omitted …]\n\n"
            + tail
        )
    return transcript


def focus_turn_user(conversation: Conversation, focus_turn: int) -> str:
    """The user message at the focus turn — what the responder must reply to."""
    for turn in conversation.turns:
        if turn.turn == focus_turn:
            return truncate(turn.user, USER_MAX_CHARS) or "(no user message)"
    return "(no user message)"


def focus_turn_old_reply(conversation: Conversation, focus_turn: int) -> str:
    """The original agent reply at the focus turn — what we compare against."""
    for turn in conversation.turns:
        if turn.turn == focus_turn:
            return truncate(turn.agent, REPLY_MAX_CHARS) or "(no agent reply)"
    return "(no agent reply)"


def evidence_for_session(target_evidence: list[Evidence], session_id: str) -> Evidence | None:
    """Find the first Evidence whose `details["session_id"]` matches.

    Returns None if no evidence was tagged for this session — replay
    falls back to defaults (last-turn focus, no extra context).
    """
    for e in target_evidence:
        if (e.details or {}).get("session_id") == session_id:
            return e
    return None
