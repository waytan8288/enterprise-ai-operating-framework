"""Tools for the insight agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from connectors.factory import create_connector


@tool
def execute_query(sql: str) -> str:
    """Execute a SQL query against the enterprise data source and return results.

    Write SQL appropriate for the configured database backend.
    Results are returned as a JSON string with columns and rows.

    Args:
        sql: The SQL query to execute.
    """
    connector = create_connector()
    try:
        result = connector.execute(sql)
        if result.error:
            return json.dumps({"error": result.error})
        return json.dumps({
            "columns": result.columns,
            "rows": result.rows[:100],
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
        })
    finally:
        connector.close()


@tool
def get_data_schema() -> str:
    """Get the schema of available tables and columns in the data source.

    Returns table names with their columns, data types, and nullability.
    """
    connector = create_connector()
    try:
        schema = connector.get_schema()
        result = {}
        for table_name, columns in schema.tables.items():
            result[table_name] = [
                {"name": c.name, "type": c.data_type, "nullable": c.nullable}
                for c in columns
            ]
        return json.dumps(result, indent=2)
    finally:
        connector.close()
