"""Handoff tools for the insight agent sub-agents.

These set ``next_agent`` in state so the insight_agent parent graph
and the orchestrator can route accordingly.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command


@tool
def transfer_to_strategy_agent(query: str) -> Command:
    """Transfer to the Strategy Agent for recommendations based on insights.

    Use when the user wants to know what to do about the data findings,
    or needs prioritized recommendations.

    Args:
        query: The recommendation or strategy question.
    """
    return Command(
        update={
            "next_agent": "strategy_agent",
            "messages": [
                ToolMessage(
                    content=f"Transferring to Strategy Agent: {query}",
                    tool_call_id="transfer",
                )
            ],
        }
    )


@tool
def transfer_to_act_agent(query: str) -> Command:
    """Transfer to the Act Agent to execute an action based on insights.

    Use when the user wants to take action on the findings.

    Args:
        query: The action request.
    """
    return Command(
        update={
            "next_agent": "act_agent",
            "messages": [
                ToolMessage(
                    content=f"Transferring to Act Agent: {query}",
                    tool_call_id="transfer",
                )
            ],
        }
    )
