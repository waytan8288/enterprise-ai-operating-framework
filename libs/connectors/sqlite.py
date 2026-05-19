"""SQLite data connector implementation."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from connectors.base import ColumnInfo, QueryResult, SchemaInfo


class SQLiteConnector:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.getenv(
            "SQLITE_DB_PATH", "./data/enterprise.db"
        )
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        start = time.monotonic()
        try:
            cursor = self._conn.execute(sql, params or {})
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(row) for row in cursor.fetchall()]
                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=(time.monotonic() - start) * 1000,
                )
            self._conn.commit()
            return QueryResult(
                row_count=cursor.rowcount,
                execution_time_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return QueryResult.from_error(str(e))

    def get_schema(self) -> SchemaInfo:
        tables: dict[str, list[ColumnInfo]] = {}
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        for row in cursor.fetchall():
            table_name = row["name"]
            col_cursor = self._conn.execute(
                f"PRAGMA table_info('{table_name}')"
            )
            columns = [
                ColumnInfo(
                    name=col["name"],
                    data_type=col["type"] or "TEXT",
                    nullable=not col["notnull"],
                )
                for col in col_cursor.fetchall()
            ]
            tables[table_name] = columns
        return SchemaInfo(tables=tables)

    def test_connection(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._conn.close()
