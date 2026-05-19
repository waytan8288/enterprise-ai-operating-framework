"""State schema for the orchestrator (COORDINATE) agent."""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, Required, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class MultiAgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    active_agent: NotRequired[
        Literal[
            "insight_agent",
            "strategy_agent",
            "act_agent",
            "measure_agent",
            "learn_agent",
            "govern_agent",
        ]
    ]
    feature_flags: NotRequired[list[str]]
    user_display_name: NotRequired[str | None]
    user_timezone: NotRequired[str | None]
    session_id: NotRequired[str | None]
    active_policies: NotRequired[list[str]]
