"""Learn Agent graph — LEARN pillar.

Captures outcomes, detects patterns, and enriches the Knowledge Graph
to make the framework's flywheel compound over time.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from learn_agent.nodes import learn_node
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

_tools = [
    get_recent_outcomes,
    search_existing_patterns,
    record_pattern,
    update_pattern_confidence,
    incorporate_feedback,
    transfer_to_measure_agent,
    transfer_to_strategy_agent,
]


def should_continue(state: LearnState) -> str:
    if state.get("next_agent"):
        return "__end__"
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


_builder = StateGraph(LearnState)
_builder.add_node("learn", learn_node)
_builder.add_node("tools", ToolNode(_tools))

_builder.add_edge(START, "learn")
_builder.add_conditional_edges("learn", should_continue, ["tools", END])
_builder.add_edge("tools", "learn")

graph = _builder.compile()
graph.name = "learn_agent"
