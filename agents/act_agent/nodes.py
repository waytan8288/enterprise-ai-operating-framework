"""Node functions for the act agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from governance.middleware import governed_model
from act_agent.prompts import ACT_AGENT_PROMPT
from act_agent.state import ActState
from act_agent.tools import (
    execute_action,
    plan_action,
    record_action_result,
    transfer_to_insight_agent,
    transfer_to_measure_agent,
)

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS = [plan_action, execute_action, record_action_result, transfer_to_measure_agent, transfer_to_insight_agent]


async def action_planning_node(
    state: ActState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM plans the action and calls plan_action tool."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "act_agent")
    model_with_tools = model.bind_tools(_TOOLS)

    messages = [SystemMessage(content=ACT_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    next_agent = None
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ("transfer_to_measure_agent", "transfer_to_insight_agent"):
                next_agent = tc["name"].replace("transfer_to_", "")
                break

    return {
        "messages": [response],
        "next_agent": next_agent,
    }


def verify_action_node(state: ActState, config: RunnableConfig) -> dict[str, Any]:
    """Human-in-the-loop approval gate.

    Pauses execution and presents the action plan to the user for approval.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    action_summary = "Action pending approval"
    if last_msg and hasattr(last_msg, "content"):
        action_summary = str(last_msg.content)[:500]

    approval = interrupt(value={
        "action_summary": action_summary,
        "message": "Please approve or reject this action.",
    })

    return {"action_approved": bool(approval)}
