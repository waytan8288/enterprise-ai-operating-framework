"""Tools for the semantic SQL agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from connectors.factory import create_connector


@tool
def execute_query(sql: str) -> str:
    """Execute a SQL query against the enterprise data source.

    Write SQL using the table and column names from the schema reference.
    Results are returned as JSON with columns and rows.

    Args:
        sql: The SQL query to execute.
    """
    connector = create_connector()
    try:
        result = connector.execute(sql)
        if result.error:
            return json.dumps({"error": result.error, "sql": sql})
        return json.dumps({
            "columns": result.columns,
            "rows": result.rows[:100],
            "row_count": result.row_count,
            "execution_time_ms": result.execution_time_ms,
        })
    finally:
        connector.close()


@tool
def summarize_results(question: str, query_results: str) -> str:
    """Summarize SQL query results in natural language.

    After executing a query, call this tool to generate a human-readable
    summary of the results.

    Args:
        question: The user's original analytics question.
        query_results: The JSON results from execute_query.
    """
    try:
        data = json.loads(query_results)
        if "error" in data:
            return f"Query failed: {data['error']}"

        row_count = data.get("row_count", 0)
        columns = data.get("columns", [])
        rows = data.get("rows", [])

        if row_count == 0:
            return "The query returned no results."

        summary_parts = [
            f"Query returned {row_count} row(s) with columns: {', '.join(columns)}.",
        ]

        if rows:
            summary_parts.append(f"First row: {json.dumps(rows[0])}")

        return "\n".join(summary_parts)
    except json.JSONDecodeError:
        return f"Raw results: {query_results}"
