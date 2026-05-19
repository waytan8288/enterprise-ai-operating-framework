"""Connector factory — creates the appropriate DataConnector from env config."""

from __future__ import annotations

import os

from connectors.base import DataConnector


def create_connector(connector_type: str | None = None) -> DataConnector:
    """Create a data connector based on CONNECTOR_TYPE env var or argument."""
    ctype = (connector_type or os.getenv("CONNECTOR_TYPE", "sqlite")).lower()

    if ctype == "sqlite":
        from connectors.sqlite import SQLiteConnector

        return SQLiteConnector()
    elif ctype == "mysql":
        from connectors.mysql import MySQLConnector

        return MySQLConnector()
    elif ctype == "snowflake":
        from connectors.snowflake import SnowflakeConnector

        return SnowflakeConnector()
    else:
        raise ValueError(
            f"Unknown connector type: {ctype}. "
            f"Supported: sqlite, mysql, snowflake"
        )
