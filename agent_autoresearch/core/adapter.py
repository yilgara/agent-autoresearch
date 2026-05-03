"""The `Adapter` ABC — the seam between your eval pipeline and autoresearch.

Subclass this and implement `load_targets()` and `load_conversations()`.
Everything else in the library works against the abstract interface,
so once your adapter is in place every downstream step (program →
propose → critic → replay → verdict) runs without modification.

See [`docs/writing_an_adapter.md`](../../docs/writing_an_adapter.md)
for a complete worked example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from agent_autoresearch.core.data import Conversation, Target


class Adapter(ABC):
    """Abstract base for eval-pipeline adapters.

    Subclasses must set the class variable `name` to a unique string
    used by the CLI (`autoresearch run --adapter <name>`) and implement
    the two abstract methods.

    Adapter instances are short-lived — usually constructed once per
    `autoresearch run` invocation. Heavy work (DB connections, file
    reads, API fetches) happens in `__init__`; the loader methods just
    return already-prepared data.
    """

    #: Unique short identifier — used in entry-point registration and on the CLI.
    #: Must be set by subclasses.
    name: ClassVar[str] = ""

    @abstractmethod
    def load_targets(self) -> list[Target]:
        """Return one `Target` per skill the pipeline should consider.

        Targets carry your eval pipeline's findings — which skills
        broke, with what evidence, and which sessions to replay
        against. The pipeline picks the top-N by `rank` (or insertion
        order if you don't set rank).

        See `agent_autoresearch.core.data.Target` for the full shape.
        """

    @abstractmethod
    def load_conversations(self) -> list[Conversation]:
        """Return all `Conversation` objects referenced by `Target`
        `fix_session_ids` and `regression_baseline_ids`.

        Loaded once per run. The pipeline indexes by `session_id`
        and looks up per-session data as replay needs it.

        Sessions whose IDs aren't referenced by any target are
        ignored (no harm in returning extras), but missing IDs
        referenced by targets are silently skipped during replay —
        keep the lists consistent.
        """

    # ── Convenience properties subclasses usually don't need to override ──

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"
