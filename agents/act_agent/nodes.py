"""Node functions for the act agent graph."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from governance.middleware import governed_model
from mcp_tools import get_mcp_tools, is_mcp_tool, parse_mcp_tool_name
from security.feature_flags import MCP_TOOLS_FLAG, has_feature, resolve_feature_flags
from act_agent.prompts import ACT_AGENT_PROMPT
from act_agent.state import ActState
from act_agent.tools import (
    _get_kg_writer,
    execute_action,
    plan_action,
    record_action_result,
    transfer_to_insight_agent,
    transfer_to_measure_agent,
)

logger = logging.getLogger(__name__)

_AGENT_NAME = "act_agent"
_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TOOLS: list[BaseTool] = [plan_action, execute_action, record_action_result, transfer_to_measure_agent, transfer_to_insight_agent]


def requires_approval(tool_name: str) -> bool:
    """State-changing tools that must pass the HITL gate before running."""
    return tool_name == "execute_action" or is_mcp_tool(tool_name)


def _mcp_enabled(state: ActState, config: RunnableConfig) -> bool:
    return has_feature(
        {"feature_flags": resolve_feature_flags(config)}, MCP_TOOLS_FLAG
    ) or has_feature(state, MCP_TOOLS_FLAG)


async def _available_tools(state: ActState, config: RunnableConfig) -> list[BaseTool]:
    if not _mcp_enabled(state, config):
        return _TOOLS
    return _TOOLS + await get_mcp_tools(_AGENT_NAME)


def _record_mcp_call(tool_name: str, args: dict[str, Any], status: str) -> None:
    """Audit every MCP tool call in the Knowledge Graph."""
    server, tool = parse_mcp_tool_name(tool_name)
    try:
        writer = _get_kg_writer()
        decision = writer.record_decision(
            agent=_AGENT_NAME,
            pillar="ACT",
            context=f"User approved MCP tool call: {server}.{tool}",
            decision_text=f"Call {tool} on MCP server {server}",
            confidence=1.0,
        )
        writer.record_action(
            decision_id=decision.id,
            agent=_AGENT_NAME,
            action_type="mcp_tool_call",
            action_detail=json.dumps({"tool": tool, "args": args}, default=str),
            target_system=server,
            status=status,
        )
    except Exception:
        logger.exception("Failed to record MCP call %s in Knowledge Graph", tool_name)


async def action_planning_node(
    state: ActState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM plans the action and calls plan_action tool."""
    model = governed_model(ChatOpenAI(model=_MODEL_NAME, temperature=0), _AGENT_NAME)
    model_with_tools = model.bind_tools(await _available_tools(state, config))

    messages = [SystemMessage(content=ACT_AGENT_PROMPT)] + state["messages"]
    response = await model_with_tools.ainvoke(messages)

    next_agent = None
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in ("transfer_to_measure_agent", "transfer_to_insight_agent"):
                next_agent = tc["name"].replace("transfer_to_", "")
                break

    return {
        "messages": [response],
        "next_agent": next_agent,
    }


async def run_tools_node(state: ActState, config: RunnableConfig) -> dict[str, Any]:
    """Execute the last AI message's tool calls (native and MCP).

    Approval is consumed once gated tools run, so each state-changing
    step needs its own approval.
    """
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls if isinstance(last_msg, AIMessage) else []
    tools_by_name = {t.name: t for t in await _available_tools(state, config)}

    results: list[ToolMessage] = []
    for tc in tool_calls:
        tool = tools_by_name.get(tc["name"])
        if tool is None:
            results.append(ToolMessage(
                content=f"Error: tool {tc['name']!r} is not available.",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            ))
            continue

        try:
            result = await tool.ainvoke(tc, config)
            status = "completed"
        except Exception as exc:
            logger.exception("Tool %s failed", tc["name"])
            result = ToolMessage(
                content=f"Error: {exc}",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            )
            status = "failed"

        if is_mcp_tool(tc["name"]):
            await asyncio.to_thread(_record_mcp_call, tc["name"], tc["args"], status)
        results.append(result)

    update: dict[str, Any] = {"messages": results}
    if any(requires_approval(tc["name"]) for tc in tool_calls):
        update["action_approved"] = False
    return update


def verify_action_node(state: ActState, config: RunnableConfig) -> dict[str, Any]:
    """Human-in-the-loop approval gate.

    Pauses execution and presents the action plan to the user for approval.
    On rejection, answers the pending tool calls so the LLM can re-plan.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    tool_calls = last_msg.tool_calls if isinstance(last_msg, AIMessage) else []

    action_summary = "Action pending approval"
    if last_msg and last_msg.content:
        action_summary = str(last_msg.content)[:500]

    approval = interrupt(value={
        "action_summary": action_summary,
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]}
            for tc in tool_calls
            if requires_approval(tc["name"])
        ],
        "message": "Please approve or reject this action.",
    })

    if approval:
        return {"action_approved": True}

    return {
        "action_approved": False,
        "messages": [
            ToolMessage(
                content="Rejected by user. Do not retry without a revised plan.",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            )
            for tc in tool_calls
        ],
    }
