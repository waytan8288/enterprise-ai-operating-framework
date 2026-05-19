"""Data connector protocol and shared models.

All connectors implement the DataConnector protocol, providing a uniform
interface for SQL execution and schema introspection across database backends.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    description: str = ""


class SchemaInfo(BaseModel):
    tables: dict[str, list[ColumnInfo]] = Field(default_factory=dict)


class QueryResult(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: str | None = None

    @classmethod
    def from_error(cls, error: str) -> QueryResult:
        return cls(error=error)


@runtime_checkable
class DataConnector(Protocol):
    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> QueryResult: ...

    def get_schema(self) -> SchemaInfo: ...

    def test_connection(self) -> bool: ...

    def close(self) -> None: ...
