"""Measure Agent graph — MEASURE pillar.

Quantifies business outcomes, attributes impact to actions,
and records outcomes in the Knowledge Graph.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from measure_agent.nodes import measure_node
from measure_agent.state import MeasureState
from measure_agent.tools import (
    get_recent_outcomes,
    measure_metric,
    record_outcome,
    transfer_to_insight_agent,
    transfer_to_learn_agent,
)

_tools = [measure_metric, record_outcome, get_recent_outcomes, transfer_to_learn_agent, transfer_to_insight_agent]


def should_continue(state: MeasureState) -> str:
    if state.get("next_agent"):
        return "__end__"
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


_builder = StateGraph(MeasureState)
_builder.add_node("measure", measure_node)
_builder.add_node("tools", ToolNode(_tools))

_builder.add_edge(START, "measure")
_builder.add_conditional_edges("measure", should_continue, ["tools", END])
_builder.add_edge("tools", "measure")

graph = _builder.compile()
graph.name = "measure_agent"
