"""Insight Agent graph — INSIGHT pillar.

Graph topology:
    START → route_entry → semantic_sql_agent --(ok)--> insight_analysis → END
                      ↘                     ↘(failed)→ insight_analysis → END
                       direct_analysis → END

Routes to the semantic SQL sub-agent for data queries, with fallback to
direct analysis. Non-data questions trigger handoffs via next_agent.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from governance.middleware import governed_model
from insight_agent.handoff_tools import transfer_to_act_agent, transfer_to_strategy_agent
from insight_agent.prompts import build_insight_prompt
from insight_agent.state import InsightState
from insight_agent.tools import execute_query, get_data_schema

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def semantic_sql_node(
    state: InsightState, config: RunnableConfig
) -> dict[str, Any]:
    """Run the semantic SQL sub-agent for data queries."""
    from insight_agent.semantic_sql_agent.graph import graph as sql_graph

    result = await sql_graph.ainvoke(
        {**state, "sql_failed": False, "next_agent": None},
        config,
    )
    messages = result.get("messages", [])
    sql_failed = False
    for msg in reversed(messages):
        if hasattr(msg, "content") and "error" in str(msg.content).lower():
            sql_failed = True
            break

    return {
        "messages": messages,
        "sql_failed": sql_failed,
        "next_agent": result.get("next_agent"),
    }


async def insight_analysis_node(
    state: InsightState, config: RunnableConfig
) -> dict[str, Any]:
    """Main insight analysis — LLM with data tools and handoff tools."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), "insight_agent")
    tools = [execute_query, get_data_schema, transfer_to_strategy_agent, transfer_to_act_agent]
    model_with_tools = model.bind_tools(tools)

    prompt = build_insight_prompt(state.get("user_timezone"))
    messages = [SystemMessage(content=prompt)] + state["messages"]

    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def tool_executor_node(
    state: InsightState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute tools called by the insight analysis node."""
    tool_node = ToolNode([execute_query, get_data_schema, transfer_to_strategy_agent, transfer_to_act_agent])
    result = await tool_node.ainvoke(state, config)
    return result


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_semantic(state: InsightState) -> str:
    if state.get("next_agent"):
        return "__end__"
    if state.get("sql_failed"):
        return "insight_analysis"
    return "__end__"


def should_continue_analysis(state: InsightState) -> str:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_builder = StateGraph(InsightState)
_builder.add_node("semantic_sql", semantic_sql_node)
_builder.add_node("insight_analysis", insight_analysis_node)
_builder.add_node("tools", tool_executor_node)

_builder.add_edge(START, "semantic_sql")
_builder.add_conditional_edges(
    "semantic_sql", route_after_semantic, ["insight_analysis", END]
)
_builder.add_conditional_edges(
    "insight_analysis", should_continue_analysis, ["tools", END]
)
_builder.add_edge("tools", "insight_analysis")

graph = _builder.compile()
graph.name = "insight_agent"
