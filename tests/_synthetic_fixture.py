"""Synthetic adapter — TEST FIXTURE ONLY.

This file lives under `tests/` because the library no longer ships
generic adapters. It's the in-memory fixture pipeline / strategy /
verdict tests use to drive `run_target` / `run_pipeline` without
external data or LLM calls.

The data describes a fictional restaurant-booking agent with two
skills: `find-restaurant` and `book-table`. Five sessions across them,
including one multi-skill session (`sess_004`) that's a baseline for
one skill and a fix-target for the other — handy for verifying the
multi-skill path works.

Not exposed via entry-points; tests import `SyntheticAdapter` directly:

    from tests._synthetic_fixture import SyntheticAdapter
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
