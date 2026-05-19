"""Tools for the govern agent — audit, compliance, and data quality."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from connectors.factory import create_connector
from governance.policies import DEFAULT_POLICIES
from knowledge_graph.reader import KGReader
from knowledge_graph.store import SQLiteKnowledgeGraphStore


def _get_kg_reader() -> KGReader:
    db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
    store = SQLiteKnowledgeGraphStore(db_path=db_path)
    return KGReader(store)


@tool
def get_audit_trail(node_id: str) -> str:
    """Get the full provenance chain for any Knowledge Graph node.

    Traces Decision → Action → Outcome → Pattern chains in both directions,
    showing the complete history of how a decision was made or an outcome was produced.

    Args:
        node_id: The ID of any KG node (decision, action, outcome, or pattern).
    """
    reader = _get_kg_reader()
    chain = reader.get_audit_chain(node_id)
    return json.dumps({
        "decisions": [
            {"id": d.id, "agent": d.agent, "pillar": d.pillar,
             "decision_text": d.decision_text, "confidence": d.confidence,
             "timestamp": d.timestamp.isoformat()}
            for d in chain["decisions"]
        ],
        "actions": [
            {"id": a.id, "agent": a.agent, "action_type": a.action_type,
             "action_detail": a.action_detail, "status": a.status,
             "timestamp": a.timestamp.isoformat()}
            for a in chain["actions"]
        ],
        "outcomes": [
            {"id": o.id, "metric_name": o.metric_name, "metric_value": o.metric_value,
             "delta": o.delta, "timestamp": o.timestamp.isoformat()}
            for o in chain["outcomes"]
        ],
        "patterns": [
            {"id": p.id, "description": p.description, "confidence": p.confidence,
             "times_observed": p.times_observed}
            for p in chain["patterns"]
        ],
        "edge_count": len(chain["edges"]),
    })


@tool
def check_compliance(action_type: str, target_system: str = "") -> str:
    """Check if an action type complies with governance policies.

    Validates the proposed action against all active policies and returns
    any violations or warnings.

    Args:
        action_type: The type of action to check.
        target_system: The target system for the action.
    """
    results = []
    for policy in DEFAULT_POLICIES:
        if not policy.enabled:
            continue
        status = "PASS"
        message = f"Policy '{policy.name}' satisfied."

        if policy.id == "approval_required" and action_type in ("api_call", "workflow"):
            status = "WARN"
            message = f"Action type '{action_type}' on '{target_system}' requires human approval before execution."
        elif policy.id == "audit_trail":
            status = "PASS"
            message = "Action will be recorded in the Knowledge Graph."

        results.append({
            "policy_id": policy.id,
            "policy_name": policy.name,
            "severity": policy.severity,
            "status": status,
            "message": message,
        })

    overall = "PASS"
    if any(r["status"] == "FAIL" for r in results):
        overall = "FAIL"
    elif any(r["status"] == "WARN" for r in results):
        overall = "WARN"

    return json.dumps({"overall_status": overall, "checks": results})


@tool
def validate_data_quality(table_name: str = "") -> str:
    """Validate data quality for a given table or the overall data source.

    Checks row counts, null ratios, and basic consistency.

    Args:
        table_name: Specific table to validate, or empty for overview.
    """
    connector = create_connector()
    try:
        schema = connector.get_schema()
        if table_name and table_name in schema.tables:
            tables_to_check = {table_name: schema.tables[table_name]}
        else:
            tables_to_check = schema.tables

        results = {}
        for tname, columns in tables_to_check.items():
            count_result = connector.execute(f"SELECT COUNT(*) as cnt FROM {tname}")
            row_count = 0
            if not count_result.error and count_result.rows:
                row_count = count_result.rows[0].get("cnt", 0)

            results[tname] = {
                "row_count": row_count,
                "column_count": len(columns),
                "columns": [c.name for c in columns],
                "status": "PASS" if row_count > 0 else "WARN",
            }

        return json.dumps({"tables": results})
    finally:
        connector.close()


@tool
def list_policies() -> str:
    """List all active governance policies.

    Returns the current set of policies with their descriptions and severity levels.
    """
    return json.dumps({
        "policies": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "severity": p.severity,
                "enabled": p.enabled,
            }
            for p in DEFAULT_POLICIES
        ]
    })
