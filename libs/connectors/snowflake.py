"""Snowflake data connector implementation.

Requires the snowflake-sqlalchemy package: pip install snowflake-sqlalchemy
"""

from __future__ import annotations

import os
import time
from typing import Any

from connectors.base import ColumnInfo, QueryResult, SchemaInfo


class SnowflakeConnector:
    def __init__(self) -> None:
        from sqlalchemy import create_engine, text

        account = os.environ["SNOWFLAKE_ACCOUNT"]
        user = os.environ["SNOWFLAKE_USER"]
        password = os.environ["SNOWFLAKE_PASSWORD"]
        database = os.environ["SNOWFLAKE_DATABASE"]
        schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

        url = (
            f"snowflake://{user}:{password}@{account}/{database}/{schema}"
            f"?warehouse={warehouse}"
        )
        self._engine = create_engine(url)
        self._connection = self._engine.connect()

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        from sqlalchemy import text

        start = time.monotonic()
        try:
            result = self._connection.execute(text(sql), params or {})
            if result.returns_rows:
                columns = list(result.keys())
                rows = [dict(row._mapping) for row in result.fetchall()]
                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=(time.monotonic() - start) * 1000,
                )
            return QueryResult(
                row_count=result.rowcount,
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return QueryResult.from_error(str(e))

    def get_schema(self) -> SchemaInfo:
        from sqlalchemy import text

        tables: dict[str, list[ColumnInfo]] = {}
        result = self._connection.execute(text("SHOW TABLES"))
        for row in result.fetchall():
            table_name = row._mapping["name"]
            cols_result = self._connection.execute(
                text(f"DESCRIBE TABLE {table_name}")
            )
            columns = [
                ColumnInfo(
                    name=col._mapping["name"],
                    data_type=col._mapping["type"],
                    nullable=col._mapping.get("null?", "Y") == "Y",
                )
                for col in cols_result.fetchall()
            ]
            tables[table_name] = columns
        return SchemaInfo(tables=tables)

    def test_connection(self) -> bool:
        from sqlalchemy import text

        try:
            self._connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._connection.close()
        self._engine.dispose()
