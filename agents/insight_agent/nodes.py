"""Node functions for the insight agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from governance.middleware import governed_model
from insight_agent.handoff_tools import transfer_to_act_agent, transfer_to_strategy_agent
from insight_agent.prompts import build_insight_prompt
from insight_agent.state import InsightState
from insight_agent.tools import execute_query, get_data_schema

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


async def insight_analysis_node(
    state: InsightState, config: RunnableConfig
) -> dict[str, Any]:
    """Main insight analysis node — LLM with data tools and handoff tools."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "insight_agent")
    tools = [execute_query, get_data_schema, transfer_to_strategy_agent, transfer_to_act_agent]
    model_with_tools = model.bind_tools(tools)

    prompt = build_insight_prompt(state.get("user_timezone"))
    messages = [SystemMessage(content=prompt)] + state["messages"]

    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}
