"""Tools for the strategy agent — Knowledge Graph read and decision recording."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from knowledge_graph.reader import KGReader
from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


def _get_kg_reader() -> KGReader:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    return KGReader(store)


def _get_kg_writer() -> KGWriter:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    return KGWriter(store)


@tool
def search_knowledge_patterns(query: str, limit: int = 5) -> str:
    """Search the Knowledge Graph for historically proven patterns.

    Finds patterns that match the given query based on description,
    conditions, and recommended actions. Returns patterns with their
    confidence scores and observation counts.

    Args:
        query: Natural language description of the situation or condition.
        limit: Maximum number of patterns to return.
    """
    reader = _get_kg_reader()
    patterns = reader.search_patterns(query, limit=limit)
    if not patterns:
        return json.dumps({
            "patterns": [],
            "message": "No matching patterns found in the Knowledge Graph.",
        })
    return json.dumps({
        "patterns": [
            {
                "id": p.id,
                "type": p.pattern_type,
                "description": p.description,
                "conditions": p.conditions,
                "recommended_actions": p.recommended_actions,
                "confidence": p.confidence,
                "times_observed": p.times_observed,
            }
            for p in patterns
        ]
    })


@tool
def get_outcome_history(action_id: str) -> str:
    """Get historical outcomes for a specific action from the Knowledge Graph.

    Use this to understand how similar actions performed in the past.

    Args:
        action_id: The ID of the action to look up outcomes for.
    """
    reader = _get_kg_reader()
    outcomes = reader.get_outcomes_for_action(action_id)
    if not outcomes:
        return json.dumps({"outcomes": [], "message": "No outcomes found for this action."})
    return json.dumps({
        "outcomes": [
            {
                "id": o.id,
                "metric_name": o.metric_name,
                "metric_value": o.metric_value,
                "metric_unit": o.metric_unit,
                "baseline_value": o.baseline_value,
                "delta": o.delta,
                "attribution_confidence": o.attribution_confidence,
            }
            for o in outcomes
        ]
    })


@tool
def record_recommendation(
    recommendation: str,
    reasoning: str,
    confidence: float = 0.0,
    context: str = "",
    pattern_ids: list[str] | None = None,
) -> str:
    """Record a strategic recommendation as a Decision in the Knowledge Graph.

    Call this when you make a recommendation worth tracking. This closes
    the flywheel loop: patterns inform decisions which drive actions.

    Args:
        recommendation: The recommendation text.
        reasoning: Why this recommendation was made.
        confidence: Confidence level 0.0-1.0.
        context: Business context that led to this recommendation.
        pattern_ids: IDs of KG patterns that informed this decision.
    """
    writer = _get_kg_writer()
    decision = writer.record_decision(
        agent="strategy_agent",
        pillar="STRATEGY",
        context=context,
        decision_text=recommendation,
        confidence=confidence,
        reasoning=reasoning,
    )
    for pid in pattern_ids or []:
        writer.link_pattern_to_decision(pid, decision.id)
    return json.dumps({
        "decision_id": decision.id,
        "message": "Recommendation recorded in Knowledge Graph.",
        "linked_patterns": len(pattern_ids or []),
    })


@tool
def transfer_to_insight_agent(query: str) -> str:
    """Transfer to the Insight Agent when more data analysis is needed.

    Args:
        query: The data question that needs analysis.
    """
    return f"Transferring to insight_agent: {query}"


@tool
def transfer_to_act_agent(query: str) -> str:
    """Transfer to the Act Agent to execute a recommended action.

    Args:
        query: The action to execute.
    """
    return f"Transferring to act_agent: {query}"
