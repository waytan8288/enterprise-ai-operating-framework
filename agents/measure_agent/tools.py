"""Tools for the measure agent — outcome recording and measurement."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from connectors.factory import create_connector
from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


def _get_kg_writer() -> KGWriter:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    return KGWriter(store)


@tool
def measure_metric(sql: str, metric_name: str) -> str:
    """Query a data source to measure a specific metric.

    Args:
        sql: SQL query to compute the metric.
        metric_name: Name of the metric being measured.
    """
    connector = create_connector()
    try:
        result = connector.execute(sql)
        if result.error:
            return json.dumps({"error": result.error})
        return json.dumps({
            "metric_name": metric_name,
            "columns": result.columns,
            "rows": result.rows[:50],
            "row_count": result.row_count,
        })
    finally:
        connector.close()


@tool
def record_outcome(
    action_id: str,
    metric_name: str,
    metric_value: float,
    metric_unit: str = "",
    baseline_value: float | None = None,
    attribution_confidence: float = 0.0,
) -> str:
    """Record a measured outcome in the Knowledge Graph.

    Creates an Outcome node linked to the Action that produced it,
    with edges forming the flywheel.

    Args:
        action_id: ID of the action that produced this outcome.
        metric_name: Name of the metric (e.g., conversion_rate, revenue).
        metric_value: Measured value of the metric.
        metric_unit: Unit of measurement.
        baseline_value: Previous/baseline value for comparison.
        attribution_confidence: How confident we are in the attribution (0-1).
    """
    writer = _get_kg_writer()
    delta = None
    if baseline_value is not None:
        delta = metric_value - baseline_value

    outcome = writer.record_outcome(
        action_id=action_id,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        baseline_value=baseline_value,
        delta=delta,
        attribution_confidence=attribution_confidence,
    )
    return json.dumps({
        "outcome_id": outcome.id,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "baseline_value": baseline_value,
        "delta": delta,
        "recorded": True,
    })


@tool
def get_recent_outcomes(limit: int = 20) -> str:
    """Get recent outcomes from the Knowledge Graph for reporting.

    Args:
        limit: Number of recent outcomes to retrieve.
    """
    from knowledge_graph.reader import KGReader

    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    reader = KGReader(store)
    outcomes = reader.get_recent_outcomes(limit=limit)
    return json.dumps({
        "outcomes": [
            {
                "id": o.id,
                "action_id": o.action_id,
                "metric_name": o.metric_name,
                "metric_value": o.metric_value,
                "metric_unit": o.metric_unit,
                "delta": o.delta,
                "attribution_confidence": o.attribution_confidence,
                "timestamp": o.timestamp.isoformat(),
            }
            for o in outcomes
        ]
    })


@tool
def transfer_to_learn_agent(query: str) -> str:
    """Transfer to the Learn Agent when patterns should be detected from outcomes.

    Args:
        query: What patterns to look for.
    """
    return f"Transferring to learn_agent: {query}"


@tool
def transfer_to_insight_agent(query: str) -> str:
    """Transfer to the Insight Agent when more data is needed for measurement.

    Args:
        query: The data question.
    """
    return f"Transferring to insight_agent: {query}"
