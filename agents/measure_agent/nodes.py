"""Node functions for the measure agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from governance.middleware import governed_model
from measure_agent.prompts import MEASURE_AGENT_PROMPT
from measure_agent.state import MeasureState
from measure_agent.tools import (
    get_recent_outcomes,
    measure_metric,
    record_outcome,
    transfer_to_insight_agent,
    transfer_to_learn_agent,
)

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS = [measure_metric, record_outcome, get_recent_outcomes, transfer_to_learn_agent, transfer_to_insight_agent]


async def measure_node(
    state: MeasureState, config: RunnableConfig
) -> dict[str, Any]:
    """Main measurement node."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "measure_agent")
    model_with_tools = model.bind_tools(_TOOLS)

    messages = [SystemMessage(content=MEASURE_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    next_agent = None
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ("transfer_to_learn_agent", "transfer_to_insight_agent"):
                next_agent = tc["name"].replace("transfer_to_", "")
                break

    return {"messages": [response], "next_agent": next_agent}
