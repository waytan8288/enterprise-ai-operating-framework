"""Tools for the learn agent — pattern detection and KG enrichment."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from knowledge_graph.reader import KGReader
from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


def _get_store() -> SQLiteKnowledgeGraphStore:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    return SQLiteKnowledgeGraphStore(db_path=db_path)


@tool
def get_recent_outcomes(limit: int = 50) -> str:
    """Get recent outcomes from the Knowledge Graph for pattern analysis.

    Args:
        limit: Number of recent outcomes to retrieve.
    """
    store = _get_store()
    reader = KGReader(store)
    outcomes = reader.get_recent_outcomes(limit=limit)
    return json.dumps({
        "outcomes": [
            {
                "id": o.id,
                "action_id": o.action_id,
                "metric_name": o.metric_name,
                "metric_value": o.metric_value,
                "delta": o.delta,
                "attribution_confidence": o.attribution_confidence,
                "timestamp": o.timestamp.isoformat(),
                "metadata": o.metadata,
            }
            for o in outcomes
        ]
    })


@tool
def search_existing_patterns(query: str, limit: int = 10) -> str:
    """Search for existing patterns in the Knowledge Graph.

    Use this to check if a pattern already exists before creating a new one.

    Args:
        query: Description of the pattern to search for.
        limit: Maximum patterns to return.
    """
    store = _get_store()
    reader = KGReader(store)
    patterns = reader.search_patterns(query, limit=limit)
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
def record_pattern(
    pattern_type: str,
    description: str,
    conditions: list[str],
    recommended_actions: list[str],
    confidence: float,
    outcome_ids: list[str],
) -> str:
    """Record a newly detected pattern in the Knowledge Graph.

    Creates a Pattern node and connects it to the outcomes that revealed it.

    Args:
        pattern_type: Type of pattern (correlation, causal, temporal).
        description: Human-readable description of the pattern.
        conditions: When this pattern applies.
        recommended_actions: What to do when this pattern is detected.
        confidence: Confidence score (0-1).
        outcome_ids: IDs of outcomes that support this pattern.
    """
    store = _get_store()
    writer = KGWriter(store)
    pattern = writer.record_pattern(
        pattern_type=pattern_type,
        description=description,
        conditions=conditions,
        recommended_actions=recommended_actions,
        confidence=confidence,
        outcome_ids=outcome_ids,
    )
    return json.dumps({
        "pattern_id": pattern.id,
        "description": description,
        "confidence": confidence,
        "outcome_count": len(outcome_ids),
        "recorded": True,
    })


@tool
def update_pattern_confidence(
    pattern_id: str,
    new_outcome_ids: list[str],
    adjustment: float = 0.1,
) -> str:
    """Update an existing pattern's confidence with new confirming evidence.

    Increments times_observed, adjusts confidence, and resets the decay
    clock (last_confirmed). Also links new outcomes via edges.

    Args:
        pattern_id: ID of the existing pattern.
        new_outcome_ids: IDs of new outcomes that confirm this pattern.
        adjustment: Amount to increase confidence (capped at 1.0).
    """
    from datetime import datetime, timezone

    from knowledge_graph.models import EdgeType, KGEdge

    store = _get_store()
    pattern = store.get_pattern(pattern_id)
    if not pattern:
        return json.dumps({"error": f"Pattern {pattern_id} not found"})

    new_confidence = min(1.0, pattern.confidence + adjustment)
    new_observed = pattern.times_observed + len(new_outcome_ids)
    now = datetime.now(timezone.utc).isoformat()

    store.update_pattern(
        pattern_id,
        confidence=new_confidence,
        times_observed=new_observed,
        last_confirmed=now,
    )

    for oid in new_outcome_ids:
        store.write_edge(
            KGEdge(
                source_id=oid,
                target_id=pattern_id,
                edge_type=EdgeType.REVEALED_PATTERN,
            )
        )

    return json.dumps({
        "pattern_id": pattern_id,
        "previous_confidence": pattern.confidence,
        "new_confidence": new_confidence,
        "times_observed": new_observed,
        "new_outcomes_linked": len(new_outcome_ids),
        "last_confirmed": now,
    })


@tool
def incorporate_feedback(
    pattern_id: str,
    feedback_type: str,
    feedback_detail: str,
) -> str:
    """Incorporate human feedback on a pattern.

    Adjusts pattern confidence and resets the decay clock. Confirmation
    boosts confidence, rejection drops it, refinement keeps it unchanged.

    Args:
        pattern_id: ID of the pattern receiving feedback.
        feedback_type: Type of feedback (confirm, reject, refine).
        feedback_detail: Details of the feedback.
    """
    from datetime import datetime, timezone

    confidence_adjustments = {
        "confirm": 0.15,
        "reject": -0.3,
        "refine": 0.0,
    }
    adjustment = confidence_adjustments.get(feedback_type, 0.0)

    store = _get_store()
    pattern = store.get_pattern(pattern_id)
    if not pattern:
        return json.dumps({"error": f"Pattern {pattern_id} not found"})

    new_confidence = max(0.0, min(1.0, pattern.confidence + adjustment))
    now = datetime.now(timezone.utc).isoformat()

    store.update_pattern(
        pattern_id,
        confidence=new_confidence,
        last_confirmed=now,
    )

    return json.dumps({
        "pattern_id": pattern_id,
        "feedback_type": feedback_type,
        "previous_confidence": pattern.confidence,
        "new_confidence": new_confidence,
        "feedback_recorded": True,
        "detail": feedback_detail,
    })


@tool
def transfer_to_measure_agent(query: str) -> str:
    """Transfer to the Measure Agent when outcomes need to be measured first.

    Args:
        query: What needs to be measured.
    """
    return f"Transferring to measure_agent: {query}"


@tool
def transfer_to_strategy_agent(query: str) -> str:
    """Transfer to the Strategy Agent when patterns should inform recommendations.

    Args:
        query: The strategy context.
    """
    return f"Transferring to strategy_agent: {query}"
