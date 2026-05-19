"""Node functions for the strategy agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from governance.middleware import governed_model
from strategy_agent.prompts import STRATEGY_AGENT_PROMPT
from strategy_agent.state import StrategyState
from strategy_agent.tools import (
    get_outcome_history,
    record_recommendation,
    search_knowledge_patterns,
    transfer_to_act_agent,
    transfer_to_insight_agent,
)

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS = [search_knowledge_patterns, get_outcome_history, record_recommendation, transfer_to_insight_agent, transfer_to_act_agent]


async def strategy_node(
    state: StrategyState, config: RunnableConfig
) -> dict[str, Any]:
    """Main strategy analysis node."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "strategy_agent")
    model_with_tools = model.bind_tools(_TOOLS)

    messages = [SystemMessage(content=STRATEGY_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    next_agent = None
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ("transfer_to_insight_agent", "transfer_to_act_agent"):
                next_agent = tc["name"].replace("transfer_to_", "")
                break

    return {
        "messages": [response],
        "next_agent": next_agent,
    }
