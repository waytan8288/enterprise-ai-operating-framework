"""State schema for the govern agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class GovernState(MessagesState):
    feature_flags: NotRequired[list[str]]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
