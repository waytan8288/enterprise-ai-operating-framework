"""State schema for the measure agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class MeasureState(MessagesState):
    feature_flags: NotRequired[list[str]]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
