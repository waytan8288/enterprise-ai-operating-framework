"""State schema for the insight agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class InsightState(MessagesState):
    feature_flags: NotRequired[list[str]]
    sql_failed: NotRequired[bool]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
    data_context: NotRequired[dict[str, Any] | None]
