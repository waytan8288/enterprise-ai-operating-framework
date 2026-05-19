"""Tools for the act agent."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


def _get_kg_writer() -> KGWriter:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    return KGWriter(store)


@tool
def plan_action(
    action_type: str,
    action_detail: str,
    target_system: str = "",
    steps: list[str] | None = None,
) -> str:
    """Plan an action and present it for user approval.

    This creates an action plan that will be presented to the user
    for approval before execution.

    Args:
        action_type: Type of action (e.g., api_call, workflow, report).
        action_detail: Detailed description of what will be done.
        target_system: The external system that will be affected.
        steps: List of steps that will be executed.
    """
    plan = {
        "action_type": action_type,
        "action_detail": action_detail,
        "target_system": target_system,
        "steps": steps or [action_detail],
        "status": "pending_approval",
        "requires_approval": True,
    }
    return json.dumps(plan, indent=2)


@tool
def execute_action(
    action_type: str,
    action_detail: str,
    target_system: str = "",
) -> str:
    """Execute an approved action.

    This should only be called after the user has approved the action plan.
    The action is recorded in the Knowledge Graph for audit trail.

    Args:
        action_type: Type of action being executed.
        action_detail: What is being done.
        target_system: The external system being affected.
    """
    writer = _get_kg_writer()
    decision = writer.record_decision(
        agent="act_agent",
        pillar="ACT",
        context=f"User approved action: {action_detail}",
        decision_text=f"Execute {action_type} on {target_system}",
        confidence=1.0,
    )
    action = writer.record_action(
        decision_id=decision.id,
        agent="act_agent",
        action_type=action_type,
        action_detail=action_detail,
        target_system=target_system,
        status="completed",
    )
    return json.dumps({
        "status": "completed",
        "action_id": action.id,
        "decision_id": decision.id,
        "message": f"Action executed: {action_detail}",
    })


@tool
def record_action_result(
    action_id: str,
    status: str,
    result_detail: str,
) -> str:
    """Record the result of an executed action.

    Args:
        action_id: The ID of the action from execute_action.
        status: Final status (completed, failed, partial).
        result_detail: Description of what happened.
    """
    return json.dumps({
        "action_id": action_id,
        "status": status,
        "result_detail": result_detail,
        "recorded": True,
    })


@tool
def transfer_to_measure_agent(query: str) -> str:
    """Transfer to the Measure Agent to track outcomes of actions taken.

    Args:
        query: What should be measured.
    """
    return f"Transferring to measure_agent: {query}"


@tool
def transfer_to_insight_agent(query: str) -> str:
    """Transfer to the Insight Agent when more data is needed before acting.

    Args:
        query: The data question.
    """
    return f"Transferring to insight_agent: {query}"
