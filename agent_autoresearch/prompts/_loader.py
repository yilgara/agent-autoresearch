"""Minimal markdown prompt loader (mirrors evals-myT / evals-flume's pattern).

Each prompt lives in `<name>.md` with two top-level sections:

    # System

    <system text — usually placeholder-free>

    # User

    <user text with {placeholders}>

Use `format_prompt(name, **kwargs)` to load + substitute. Literal braces
in JSON examples must be doubled (`{{`, `}}`) — same convention as the
upstream repos.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

_SECTION_RE = re.compile(
    r"^#\s+(system|user)\s*\n(.*?)(?=^#\s+(?:system|user)\s*\n|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> tuple[str, str]:
    """Return (system_text, user_text) from `<name>.md` (cached per process)."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for m in _SECTION_RE.finditer(text):
        sections[m.group(1).lower()] = m.group(2).strip()

    if "system" not in sections or "user" not in sections:
        raise ValueError(
            f"{path} must contain both `# System` and `# User` sections "
            f"(found: {list(sections.keys())})"
        )
    return sections["system"], sections["user"]


def format_prompt(name: str, **kwargs) -> tuple[str, str]:
    """Load and substitute `{placeholder}` fields in the user section.

    System prompt is returned verbatim (shouldn't carry placeholders).
    """
    system, user = load_prompt(name)
    return system, user.format(**kwargs)
