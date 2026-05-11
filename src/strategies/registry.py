"""Strategy version registry.

Maps short names ("v1", "v2", "v3", or just "1"/"2"/"3") to the
matching strategy module so the pipeline can pick which version's
stages to run without an `if/elif` chain over imports.

Usage:

    from agent_autoresearch.strategies.registry import get_strategy
    strategy = get_strategy("v2")
    result = strategy.build_program(target, ...)

The pipeline orchestrator branches on `strategy.STRATEGY_VERSION` to
handle the per-version differences (e.g. v2/v3 propose takes
validators that v1 doesn't).
"""

from __future__ import annotations

import importlib
from types import ModuleType


_STRATEGIES: dict[str, str] = {
    "v1": "agent_autoresearch.strategies.v1",
    "v2": "agent_autoresearch.strategies.v2",
    "v3": "agent_autoresearch.strategies.v3",
}


# Cache loaded modules so repeated lookups don't re-import.
_CACHE: dict[str, ModuleType] = {}


def list_strategies() -> list[str]:
    """Names of all registered strategies, in version order."""
    return list(_STRATEGIES.keys())


def get_strategy(name: str) -> ModuleType:
    """Look up a strategy module by name.

    Accepts `"v1"`, `"v2"`, `"v3"` or the bare `"1"`, `"2"`, `"3"`.
    Raises `ValueError` with a friendly listing on unknown input.
    """
    key = _normalize(name)
    if key not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {name!r}. "
            f"Available: {', '.join(_STRATEGIES.keys())}."
        )
    if key not in _CACHE:
        _CACHE[key] = importlib.import_module(_STRATEGIES[key])
    return _CACHE[key]


def _normalize(name: str) -> str:
    n = (name or "").strip().lower()
    if n.startswith("v"):
        return n
    if n.isdigit():
        return f"v{n}"
    return n


__all__ = ["get_strategy", "list_strategies"]
