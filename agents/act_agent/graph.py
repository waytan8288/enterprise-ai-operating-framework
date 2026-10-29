"""Act Agent graph — ACT pillar with HITL approval gates.

Graph topology:
    START → action_planning → [tools | verify | end]
    tools → action_planning (loop)
    verify → tools (approved) | action_planning (rejected)

State-changing tools (execute_action and all MCP tools) go through verify.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from act_agent.nodes import (
    action_planning_node,
    requires_approval,
    run_tools_node,
    verify_action_node,
)
from act_agent.state import ActState


def route_after_planning(state: ActState) -> str:
    if state.get("next_agent"):
        return "__end__"

    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        needs_approval = any(requires_approval(tc["name"]) for tc in last_msg.tool_calls)
        if needs_approval and not state.get("action_approved"):
            return "verify"
        return "tools"

    return "__end__"


def route_after_verify(state: ActState) -> str:
    return "tools" if state.get("action_approved") else "action_planning"


_builder = StateGraph(ActState)
_builder.add_node("action_planning", action_planning_node)
_builder.add_node("tools", run_tools_node)
_builder.add_node("verify", verify_action_node)

_builder.add_edge(START, "action_planning")
_builder.add_conditional_edges(
    "action_planning",
    route_after_planning,
    ["tools", "verify", END],
)
_builder.add_edge("tools", "action_planning")
_builder.add_conditional_edges("verify", route_after_verify, ["tools", "action_planning"])

graph = _builder.compile()
graph.name = "act_agent"
