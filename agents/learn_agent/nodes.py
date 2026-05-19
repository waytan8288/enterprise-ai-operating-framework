"""Node functions for the learn agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from governance.middleware import governed_model
from learn_agent.prompts import LEARN_AGENT_PROMPT
from learn_agent.state import LearnState
from learn_agent.tools import (
    get_recent_outcomes,
    incorporate_feedback,
    record_pattern,
    search_existing_patterns,
    transfer_to_measure_agent,
    transfer_to_strategy_agent,
    update_pattern_confidence,
)

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS = [
    get_recent_outcomes,
    search_existing_patterns,
    record_pattern,
    update_pattern_confidence,
    incorporate_feedback,
    transfer_to_measure_agent,
    transfer_to_strategy_agent,
]


async def learn_node(
    state: LearnState, config: RunnableConfig
) -> dict[str, Any]:
    """Main learning and pattern detection node."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "learn_agent")
    model_with_tools = model.bind_tools(_TOOLS)

    messages = [SystemMessage(content=LEARN_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    next_agent = None
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ("transfer_to_measure_agent", "transfer_to_strategy_agent"):
                next_agent = tc["name"].replace("transfer_to_", "")
                break

    return {"messages": [response], "next_agent": next_agent}
