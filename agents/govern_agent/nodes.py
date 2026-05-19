"""Node functions for the govern agent graph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from governance.middleware import governed_model
from govern_agent.prompts import GOVERN_AGENT_PROMPT
from govern_agent.state import GovernState
from govern_agent.tools import (
    check_compliance,
    get_audit_trail,
    list_policies,
    validate_data_quality,
)

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS = [get_audit_trail, check_compliance, validate_data_quality, list_policies]


async def govern_node(
    state: GovernState, config: RunnableConfig
) -> dict[str, Any]:
    """Main governance node."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "govern_agent")
    model_with_tools = model.bind_tools(_TOOLS)

    messages = [SystemMessage(content=GOVERN_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    return {"messages": [response]}
