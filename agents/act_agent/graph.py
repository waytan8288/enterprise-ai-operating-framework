"""Act Agent graph — ACT pillar with HITL approval gates.

Graph topology:
    START → action_planning → [tools | verify | end]
    tools → action_planning (loop)
    verify → action_planning (after approval)
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from act_agent.nodes import action_planning_node, verify_action_node
from act_agent.state import ActState
from act_agent.tools import (
    execute_action,
    plan_action,
    record_action_result,
    transfer_to_insight_agent,
    transfer_to_measure_agent,
)

_tools = [plan_action, execute_action, record_action_result, transfer_to_measure_agent, transfer_to_insight_agent]


def route_after_planning(state: ActState) -> str:
    if state.get("next_agent"):
        return "__end__"

    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        has_execute = any(tc["name"] == "execute_action" for tc in last_msg.tool_calls)
        if has_execute and not state.get("action_approved"):
            return "verify"
        return "tools"

    return "__end__"


_builder = StateGraph(ActState)
_builder.add_node("action_planning", action_planning_node)
_builder.add_node("tools", ToolNode(_tools))
_builder.add_node("verify", verify_action_node)

_builder.add_edge(START, "action_planning")
_builder.add_conditional_edges(
    "action_planning",
    route_after_planning,
    ["tools", "verify", END],
)
_builder.add_edge("tools", "action_planning")
_builder.add_edge("verify", "action_planning")

graph = _builder.compile()
graph.name = "act_agent"
