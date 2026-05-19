"""State schema for the strategy agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class StrategyState(MessagesState):
    feature_flags: NotRequired[list[str]]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
    patterns_context: NotRequired[list[dict[str, Any]] | None]
