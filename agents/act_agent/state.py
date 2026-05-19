"""State schema for the act agent."""

from __future__ import annotations

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class ActState(MessagesState):
    feature_flags: NotRequired[list[str]]
    next_agent: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
    pending_action: NotRequired[dict[str, Any] | None]
    action_approved: NotRequired[bool]
