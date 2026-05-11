"""agent-autoresearch — auto-improve agent skill prompts from your eval pipeline.

The public API a user (most importantly: an adapter author) imports:

    from agent_autoresearch import (
        Adapter, Target, Conversation, Turn, ToolCall, Evidence,
        SkillIO, FilesystemSkillIO,
        LLMProvider, AnthropicLLMProvider,
    )

Everything else (strategies, prompts, validate, verdict, pipeline,
CLI) is implementation. Adapter authors should never need to touch
those modules directly — only the data classes + base interfaces
above.
"""

from agent_autoresearch.core.adapter import Adapter
from agent_autoresearch.core.data import (
    Conversation,
    Evidence,
    Target,
    ToolCall,
    Turn,
)
from agent_autoresearch.core.llm import (
    AnthropicLLMProvider,
    LLMProvider,
    LLMResponse,
    OpenAILLMProvider,
)
from agent_autoresearch.core.skill_io import (
    FilesystemSkillIO,
    SkillIO,
    UNATTRIBUTED,
)


__version__ = "0.0.1"

__all__ = [
    # data
    "Target",
    "Conversation",
    "Turn",
    "ToolCall",
    "Evidence",
    # adapter
    "Adapter",
    # skill IO
    "SkillIO",
    "FilesystemSkillIO",
    "UNATTRIBUTED",
    # LLM
    "LLMProvider",
    "AnthropicLLMProvider",
    "OpenAILLMProvider",
    "LLMResponse",
    # version
    "__version__",
]
