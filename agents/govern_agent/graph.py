"""Govern Agent graph — GOVERN pillar.

Handles audit trails, compliance checking, data quality validation,
and policy enforcement queries.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from govern_agent.nodes import govern_node
from govern_agent.state import GovernState
from govern_agent.tools import (
    check_compliance,
    get_audit_trail,
    list_policies,
    validate_data_quality,
)

_tools = [get_audit_trail, check_compliance, validate_data_quality, list_policies]


def should_continue(state: GovernState) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


_builder = StateGraph(GovernState)
_builder.add_node("govern", govern_node)
_builder.add_node("tools", ToolNode(_tools))

_builder.add_edge(START, "govern")
_builder.add_conditional_edges("govern", should_continue, ["tools", END])
_builder.add_edge("tools", "govern")

graph = _builder.compile()
graph.name = "govern_agent"
