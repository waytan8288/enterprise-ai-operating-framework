"""Tests for data connectors."""

import tempfile
import os

import pytest

from connectors.sqlite import SQLiteConnector
from connectors.factory import create_connector


@pytest.fixture
def connector():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        c = SQLiteConnector(db_path=db_path)
        c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        c.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
        c.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
        yield c
        c.close()


class TestSQLiteConnector:
    def test_execute_select(self, connector):
        result = connector.execute("SELECT * FROM users")
        assert result.error is None
        assert result.row_count == 2
        assert result.columns == ["id", "name", "email"]
        assert result.rows[0]["name"] == "Alice"

    def test_execute_with_error(self, connector):
        result = connector.execute("SELECT * FROM nonexistent")
        assert result.error is not None

    def test_get_schema(self, connector):
        schema = connector.get_schema()
        assert "users" in schema.tables
        columns = schema.tables["users"]
        col_names = [c.name for c in columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names

    def test_test_connection(self, connector):
        assert connector.test_connection() is True


class TestFactory:
    def test_create_sqlite_connector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SQLITE_DB_PATH"] = os.path.join(tmpdir, "test.db")
            c = create_connector("sqlite")
            assert c.test_connection()
            c.close()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown connector type"):
            create_connector("postgres")
