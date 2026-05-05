"""Synthetic adapter — hardcoded fixtures for tests + first-run smoke checks.

This adapter returns the same in-memory data every time. It exists for
three reasons:

  1. **Install verification.** After `pip install agent-autoresearch`,
     a user can run `autoresearch run --adapter synthetic --dry-run`
     and confirm the wiring works without setting up any data.
  2. **Tests.** Pipeline / strategy / verdict tests need *some*
     adapter to drive them. Rather than every test file rolling its
     own mock, they use this one.
  3. **Documentation.** The fixtures here illustrate exactly what
     shapes `Target`, `Conversation`, and `Evidence` are supposed to
     take when you write your own adapter.

The data describes a fictional restaurant-booking agent with two
skills: `find-restaurant` and `book-table`. Five sessions across them,
including one multi-skill session (`sess_004`) that's a baseline for
one skill and a fix-target for the other — handy for verifying the
multi-skill path works.

This adapter is **not** for serious use. The fixtures are too small
and too uniform to produce meaningful prompt edits — running the full
LLM pipeline against them mostly tests that the wiring works, not
that the output is good.

## CLI usage

```bash
# Smoke check — exercises adapter + summary, no LLM calls
autoresearch run --adapter synthetic --dry-run

# Full pipeline — needs `skills/find-restaurant/SKILL.md` +
# `skills/book-table/SKILL.md` on disk. Add minimal fixture skills
# under `./skills/` first, or point `--skills-root` at your own.
autoresearch run --adapter synthetic --top-n 2
```
"""

from __future__ import annotations

from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import (
    Conversation,
    Evidence,
    Target,
    ToolCall,
    Turn,
)


# ── Conversations ────────────────────────────────────────────────────────────

_CONVERSATIONS: list[Conversation] = [
    # sess_001 — fail on find-restaurant (vegan filter ignored on turn 2)
    Conversation(
        session_id="sess_001",
        turns=[
            Turn(turn=1,
                 user="Find me dinner downtown.",
                 agent="Sure, any preferences?"),
            Turn(turn=2,
                 user="Vegan please.",
                 agent="Bob's Steakhouse is great!",
                 tool_calls=[ToolCall(
                     name="search_restaurants",
                     args={"q": "downtown"},   # ← missing the vegan filter
                     output=[{"name": "Bob's Steakhouse"}],
                 )]),
        ],
    ),

    # sess_002 — fail on find-restaurant (refused a reasonable request, turn 1)
    Conversation(
        session_id="sess_002",
        turns=[
            Turn(turn=1,
                 user="Find me a place to eat downtown.",
                 agent="I think you should check Yelp."),
        ],
    ),

    # sess_003 — clean baseline for find-restaurant
    Conversation(
        session_id="sess_003",
        turns=[
            Turn(turn=1,
                 user="Vegan dinner downtown?",
                 agent="Found Green Leaf — vegan-friendly, downtown.",
                 tool_calls=[ToolCall(
                     name="search_restaurants",
                     args={"q": "vegan downtown", "filter": "vegan"},
                     output=[{"name": "Green Leaf"}],
                 )]),
        ],
    ),

    # sess_004 — multi-skill: passes find-restaurant, fails book-table
    Conversation(
        session_id="sess_004",
        turns=[
            Turn(turn=1,
                 user="Find a vegan restaurant for tomorrow night.",
                 agent="Green Leaf is vegan, downtown, open tomorrow.",
                 tool_calls=[ToolCall(
                     name="search_restaurants",
                     args={"q": "vegan downtown", "filter": "vegan"},
                     output=[{"name": "Green Leaf"}],
                 )]),
            Turn(turn=2,
                 user="Yes, book it for 7pm tomorrow, party of 2.",
                 agent="Booked!",   # ← agent claims success despite the tool error
                 tool_calls=[ToolCall(
                     name="book_table",
                     args={"restaurant": "Green Leaf",
                           "party": 2,
                           "date": "05/06/2026",   # ← wrong format
                           "time": "19:00"},
                     error="Invalid date format - expected ISO (YYYY-MM-DD)",
                 )]),
        ],
    ),

    # sess_005 — clean baseline for book-table
    Conversation(
        session_id="sess_005",
        turns=[
            Turn(turn=1,
                 user="Book a table at Green Leaf for tomorrow 7pm, party of 2.",
                 agent="Done — confirmation #ABC123.",
                 tool_calls=[ToolCall(
                     name="book_table",
                     args={"restaurant": "Green Leaf",
                           "party": 2,
                           "date": "2026-05-06",
                           "time": "19:00"},
                     output={"confirmation": "ABC123"},
                 )]),
        ],
    ),
]


# ── Targets ──────────────────────────────────────────────────────────────────

_TARGETS: list[Target] = [
    Target(
        skill_name="find-restaurant",
        rank=0,
        fix_session_ids=["sess_001", "sess_002"],
        regression_baseline_ids=["sess_003", "sess_004"],
        evidence=[
            Evidence(
                category="ignored_constraint",
                details={
                    "summary": "Agent ignored the user's explicit vegan "
                               "constraint and recommended a steakhouse.",
                    "session_id": "sess_001",
                    "focus_turn": 2,
                },
            ),
            Evidence(
                category="refused_reasonable_request",
                details={
                    "summary": "Agent refused a normal restaurant lookup "
                               "with 'check Yelp' instead of using the skill.",
                    "session_id": "sess_002",
                    "focus_turn": 1,
                },
            ),
        ],
    ),
    Target(
        skill_name="book-table",
        rank=1,
        fix_session_ids=["sess_004"],
        regression_baseline_ids=["sess_005"],
        evidence=[
            Evidence(
                category="wrong_arguments",
                details={
                    "summary": "Agent passed date as MM/DD/YYYY when the tool "
                               "expects ISO (YYYY-MM-DD), then hallucinated "
                               "'Booked!' despite the tool returning an error.",
                    "session_id": "sess_004",
                    "focus_turn": 2,
                },
            ),
        ],
    ),
]


# ── Adapter ──────────────────────────────────────────────────────────────────

class SyntheticAdapter(Adapter):
    """Returns hardcoded fixtures every time. See module docstring."""

    name = "synthetic"

    def load_targets(self) -> list[Target]:
        return list(_TARGETS)         # defensive copy

    def load_conversations(self) -> list[Conversation]:
        return list(_CONVERSATIONS)
