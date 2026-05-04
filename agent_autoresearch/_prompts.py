"""Markdown prompt loader — shared utility used by all strategy versions.

Each strategy version (`strategies/v1/`, `strategies/v2/`, …) keeps
its own `prompts/` directory with version-specific markdown files.
Stage code calls `format_prompt(path, **kwargs)` pointing at the
specific file it wants.

Prompt file format:

    # System

    <system prompt text>

    # User

    <user prompt text with {placeholders}>

`load_prompt(path)` returns (system_text, user_text) cached per file.
`format_prompt(path, **kwargs)` substitutes `{placeholder}` fields in
the user section using `str.format()`. Literal braces inside the
template (e.g. JSON examples) must be doubled: `{{` and `}}`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_SECTION_RE = re.compile(
    r"^#\s+(system|user)\s*\n(.*?)(?=^#\s+(?:system|user)\s*\n|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@lru_cache(maxsize=None)
def load_prompt(path: str | Path) -> tuple[str, str]:
    """Return (system_text, user_text) from a prompt markdown file.

    Cached per file path — prompts are read once per process. Edit
    the .md file and restart the process to pick up changes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")

    text = p.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for match in _SECTION_RE.finditer(text):
        heading = match.group(1).lower()
        body = match.group(2).strip()
        sections[heading] = body

    if "system" not in sections or "user" not in sections:
        raise ValueError(
            f"{p} must contain both `# System` and `# User` sections "
            f"(found: {list(sections.keys())})"
        )
    return sections["system"], sections["user"]


def format_prompt(path: str | Path, **kwargs) -> tuple[str, str]:
    """Load a prompt and substitute `{placeholder}` fields in the user section.

    The system prompt is returned verbatim — it shouldn't carry
    placeholders. Raises `KeyError` if a `{placeholder}` in the
    template isn't supplied in `kwargs`.
    """
    system, user = load_prompt(path)
    return system, user.format(**kwargs)
