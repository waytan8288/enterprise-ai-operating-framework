"""Enterprise AI Operating Framework Orchestrator Agent — COORDINATE pillar.

Supervisor graph that routes user requests to the appropriate specialist agent.

Architecture:
    START → auth_router → [agent nodes] → route_after_agent → END

The auth_router reads feature flags per-turn and overwrites state to prevent
privilege escalation on resumed threads.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from governance.middleware import redact_pii
from governance.policies import DEFAULT_POLICIES
from orchestrator_agent.prompts import build_orchestrator_prompt
from orchestrator_agent.state import MultiAgentState
from security.feature_flags import (
    AGENT_FLAG_MAP,
    has_feature,
    resolve_feature_flags,
)

logger = logging.getLogger(__name__)

_AGENT_NAMES = Literal[
    "insight_agent",
    "strategy_agent",
    "act_agent",
    "measure_agent",
    "learn_agent",
    "govern_agent",
]


# ---------------------------------------------------------------------------
# Govern gate — policy checks on every request
# ---------------------------------------------------------------------------


async def govern_gate(state: MultiAgentState, config: RunnableConfig) -> Command:
    """Apply governance policies before routing.

    Redacts PII from incoming messages and injects active policy context
    into state for downstream agents.
    """
    messages = list(state.get("messages", []))
    redacted = False
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            cleaned = redact_pii(msg.content)
            if cleaned != msg.content:
                messages[i] = HumanMessage(content=cleaned, id=msg.id)
                redacted = True

    active_policies = [p.id for p in DEFAULT_POLICIES if p.enabled]
    update: dict[str, Any] = {"active_policies": active_policies}
    if redacted:
        update["messages"] = messages
        logger.info("PII redacted from user input before routing")

    return Command(goto="auth_router", update=update)


# ---------------------------------------------------------------------------
# Auth router — per-turn feature flag resolution
# ---------------------------------------------------------------------------


async def auth_router(state: MultiAgentState, config: RunnableConfig) -> Command:
    """Per-turn state-of-truth for feature_flags and active_agent.

    Reads feature_flags from the authenticated user context on every
    invocation and overwrites state. Revoked flags downgrade a resumed
    active_agent in the same turn.
    """
    flags = resolve_feature_flags(config)
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}

    resumed = state.get("active_agent")
    if resumed:
        flag = AGENT_FLAG_MAP.get(resumed)
        if flag and not has_feature({"feature_flags": flags}, flag):
            resumed = None

    update = {
        "feature_flags": flags,
        "user_display_name": configurable.get("user_display_name"),
        "user_timezone": configurable.get("user_timezone"),
    }

    if resumed:
        update["active_agent"] = resumed
        return Command(goto=resumed, update=update)

    return Command(goto="llm_router", update=update)


# ---------------------------------------------------------------------------
# LLM router node — uses LLM to classify intent and route
# ---------------------------------------------------------------------------

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@tool
def transfer_to_insight_agent(query: str) -> str:
    """Transfer to the Insight Agent for data analysis, trend detection, anomaly detection, or root cause analysis.

    Args:
        query: The user's analytics question or data request.
    """
    return f"Transferring to insight_agent: {query}"


@tool
def transfer_to_strategy_agent(query: str) -> str:
    """Transfer to the Strategy Agent for recommendations, action prioritization, or pattern-based advice.

    Args:
        query: The user's strategy or recommendation request.
    """
    return f"Transferring to strategy_agent: {query}"


@tool
def transfer_to_act_agent(query: str) -> str:
    """Transfer to the Act Agent to execute workflows, make API calls, or take approved actions.

    Args:
        query: The user's action request.
    """
    return f"Transferring to act_agent: {query}"


@tool
def transfer_to_measure_agent(query: str) -> str:
    """Transfer to the Measure Agent to quantify outcomes, track KPIs, or generate impact reports.

    Args:
        query: The user's measurement or reporting request.
    """
    return f"Transferring to measure_agent: {query}"


@tool
def transfer_to_learn_agent(query: str) -> str:
    """Transfer to the Learn Agent to capture outcomes, detect patterns, or enrich the Knowledge Graph.

    Args:
        query: The user's learning or knowledge capture request.
    """
    return f"Transferring to learn_agent: {query}"


@tool
def transfer_to_govern_agent(query: str) -> str:
    """Transfer to the Govern Agent for audit trails, compliance checks, data quality, or policy queries.

    Args:
        query: The user's governance or compliance request.
    """
    return f"Transferring to govern_agent: {query}"


_TRANSFER_TOOLS = {
    "transfer_to_insight_agent": ("insight_agent", transfer_to_insight_agent),
    "transfer_to_strategy_agent": ("strategy_agent", transfer_to_strategy_agent),
    "transfer_to_act_agent": ("act_agent", transfer_to_act_agent),
    "transfer_to_measure_agent": ("measure_agent", transfer_to_measure_agent),
    "transfer_to_learn_agent": ("learn_agent", transfer_to_learn_agent),
    "transfer_to_govern_agent": ("govern_agent", transfer_to_govern_agent),
}

_TOOL_TO_FLAG = {
    "transfer_to_insight_agent": AGENT_FLAG_MAP["insight_agent"],
    "transfer_to_strategy_agent": AGENT_FLAG_MAP["strategy_agent"],
    "transfer_to_act_agent": AGENT_FLAG_MAP["act_agent"],
    "transfer_to_measure_agent": AGENT_FLAG_MAP["measure_agent"],
    "transfer_to_learn_agent": AGENT_FLAG_MAP["learn_agent"],
    "transfer_to_govern_agent": AGENT_FLAG_MAP["govern_agent"],
}


async def llm_router(state: MultiAgentState, config: RunnableConfig) -> Command:
    """LLM-based intent classification and routing."""
    flags = state.get("feature_flags", [])
    flag_source = {"feature_flags": flags}

    available_tools = []
    for tool_name, (agent_name, tool_fn) in _TRANSFER_TOOLS.items():
        flag = _TOOL_TO_FLAG.get(tool_name)
        if flag is None or has_feature(flag_source, flag):
            available_tools.append(tool_fn)

    prompt = build_orchestrator_prompt(
        feature_flags=flags,
        user_display_name=state.get("user_display_name"),
        user_timezone=state.get("user_timezone"),
    )

    model = ChatOpenAI(model=_MODEL_NAME, temperature=0)
    model_with_tools = model.bind_tools(available_tools)

    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        agent_info = _TRANSFER_TOOLS.get(tool_name)
        if agent_info:
            target_agent, _ = agent_info
            return Command(
                goto=target_agent,
                update={
                    "active_agent": target_agent,
                    "messages": [response],
                },
            )

    return Command(
        goto=END,
        update={"messages": [response]},
    )


# ---------------------------------------------------------------------------
# Agent subgraph node wrappers
# ---------------------------------------------------------------------------


def _clean_messages_for_subgraph(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Filter messages to only pass clean context to subgraphs.

    Removes orphaned tool calls from the orchestrator's LLM router
    that would cause OpenAI API errors in subgraphs.
    """
    clean: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            clean.append(msg)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            clean.append(msg)
        elif isinstance(msg, ToolMessage):
            pass
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            pass
    if not clean:
        clean = [m for m in messages if isinstance(m, HumanMessage)]
    return clean


async def call_insight_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from insight_agent.graph import graph as insight_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await insight_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


async def call_strategy_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from strategy_agent.graph import graph as strategy_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await strategy_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


async def call_act_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from act_agent.graph import graph as act_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await act_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


async def call_measure_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from measure_agent.graph import graph as measure_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await measure_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


async def call_learn_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from learn_agent.graph import graph as learn_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await learn_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


async def call_govern_agent(
    state: MultiAgentState, config: RunnableConfig
) -> dict[str, Any]:
    from govern_agent.graph import graph as govern_graph

    subgraph_state = {**state, "messages": _clean_messages_for_subgraph(state["messages"])}
    result = await govern_graph.ainvoke(subgraph_state, config)
    return {
        "messages": result.get("messages", []),
        "active_agent": result.get("next_agent", state.get("active_agent")),
    }


# ---------------------------------------------------------------------------
# Route after agent — check if done or handoff
# ---------------------------------------------------------------------------


def route_after_agent(
    state: MultiAgentState,
) -> str:
    """Route based on active_agent, or END if the agent finished without handoff."""
    messages = state.get("messages", [])

    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    active = state.get("active_agent", "insight_agent")
    flag_source = {"feature_flags": state.get("feature_flags", [])}
    flag = AGENT_FLAG_MAP.get(active)
    if flag and not has_feature(flag_source, flag):
        return "insight_agent"

    return active


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(MultiAgentState)

builder.add_node("govern_gate", govern_gate)
builder.add_node("auth_router", auth_router)
builder.add_node("llm_router", llm_router)
builder.add_node("insight_agent", call_insight_agent)
builder.add_node("strategy_agent", call_strategy_agent)
builder.add_node("act_agent", call_act_agent)
builder.add_node("measure_agent", call_measure_agent)
builder.add_node("learn_agent", call_learn_agent)
builder.add_node("govern_agent", call_govern_agent)

builder.add_edge(START, "govern_gate")

_ALL_AGENTS = [
    "insight_agent",
    "strategy_agent",
    "act_agent",
    "measure_agent",
    "learn_agent",
    "govern_agent",
    END,
]

for agent_name in [
    "insight_agent", "strategy_agent", "act_agent",
    "measure_agent", "learn_agent", "govern_agent",
]:
    builder.add_conditional_edges(agent_name, route_after_agent, _ALL_AGENTS)

graph = builder.compile()
graph.name = "orchestrator_agent"
