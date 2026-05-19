"""Strategy Agent graph — STRATEGY pillar.

Searches the Knowledge Graph for patterns, ranks recommendations,
and hands off to act_agent or insight_agent as needed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from strategy_agent.nodes import strategy_node
from strategy_agent.state import StrategyState
from strategy_agent.tools import (
    get_outcome_history,
    record_recommendation,
    search_knowledge_patterns,
    transfer_to_act_agent,
    transfer_to_insight_agent,
)

_tools = [search_knowledge_patterns, get_outcome_history, record_recommendation, transfer_to_insight_agent, transfer_to_act_agent]


def should_continue(state: StrategyState) -> str:
    messages = state.get("messages", [])
    if state.get("next_agent"):
        return "__end__"
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


_builder = StateGraph(StrategyState)
_builder.add_node("strategy", strategy_node)
_builder.add_node("tools", ToolNode(_tools))

_builder.add_edge(START, "strategy")
_builder.add_conditional_edges("strategy", should_continue, ["tools", END])
_builder.add_edge("tools", "strategy")

graph = _builder.compile()
graph.name = "strategy_agent"
