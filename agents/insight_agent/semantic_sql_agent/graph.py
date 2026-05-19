"""Semantic SQL agent graph — NL to SQL with schema-in-prompt."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from insight_agent.semantic_sql_agent.prompts import build_semantic_sql_prompt
from insight_agent.semantic_sql_agent.tools import execute_query, summarize_results

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

_tools = [execute_query, summarize_results]


async def sql_agent_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """LLM node that generates SQL and calls tools."""
    model = ChatOpenAI(model=_MODEL_NAME, temperature=0)
    model_with_tools = model.bind_tools(_tools)

    user_timezone = state.get("user_timezone")
    prompt = build_semantic_sql_prompt(user_timezone)
    messages = [SystemMessage(content=prompt)] + state["messages"]

    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "__end__"


_builder = StateGraph(MessagesState)
_builder.add_node("agent", sql_agent_node)
_builder.add_node("tools", ToolNode(_tools))

_builder.add_edge(START, "agent")
_builder.add_conditional_edges("agent", should_continue, ["tools", END])
_builder.add_edge("tools", "agent")

graph = _builder.compile()
graph.name = "semantic_sql_agent"
