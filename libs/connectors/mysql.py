"""MySQL data connector implementation.

Requires the pymysql package: pip install pymysql
"""

from __future__ import annotations

import os
import time
from typing import Any

from connectors.base import ColumnInfo, QueryResult, SchemaInfo


class MySQLConnector:
    def __init__(self) -> None:
        import pymysql

        self._conn = pymysql.connect(
            host=os.environ["MYSQL_HOST"],
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.environ["MYSQL_USER"],
            password=os.environ["MYSQL_PASSWORD"],
            database=os.environ["MYSQL_DATABASE"],
            cursorclass=pymysql.cursors.DictCursor,
        )

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        start = time.monotonic()
        try:
            with self._conn.cursor() as cursor:
                cursor.execute(sql, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return QueryResult(
                        columns=columns,
                        rows=list(rows),
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
        with self._conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            table_names = [list(row.values())[0] for row in cursor.fetchall()]
            for table_name in table_names:
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = [
                    ColumnInfo(
                        name=col["Field"],
                        data_type=col["Type"],
                        nullable=col["Null"] == "YES",
                    )
                    for col in cursor.fetchall()
                ]
                tables[table_name] = columns
        return SchemaInfo(tables=tables)

    def test_connection(self) -> bool:
        try:
            with self._conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._conn.close()
