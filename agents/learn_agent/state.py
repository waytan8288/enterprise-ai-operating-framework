"""State schema for the learn agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class LearnState(MessagesState):
    feature_flags: NotRequired[list[str]]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
