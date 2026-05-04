"""Shared helpers used across stages — XML-tag extraction, output cleaning.

Private to the stages package; not part of the public API.
"""

from __future__ import annotations

import re


# ── XML-tag extraction (used by propose + critic parsers) ───────────────────

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
    """Pull the contents of one XML tag from a free-form string. Returns None
    if the tag is missing — callers fall back to defaults."""
    m = xml_tag(name).search(text)
    return m.group(1).strip() if m else None


# ── Output cleaning (used by build_program parser) ──────────────────────────

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
