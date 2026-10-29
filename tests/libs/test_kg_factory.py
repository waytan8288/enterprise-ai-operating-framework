"""Tests for the KG store factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph.factory import create_kg_store
from knowledge_graph.store import SQLiteKnowledgeGraphStore


@pytest.fixture(autouse=True)
def _clear_cache():
    create_kg_store.cache_clear()
    yield
    create_kg_store.cache_clear()


def test_sqlite_default(tmp_path):
    db_path = str(tmp_path / "test.db")
    with patch.dict("os.environ", {"KG_SQLITE_PATH": db_path}):
        store = create_kg_store("sqlite")
    assert isinstance(store, SQLiteKnowledgeGraphStore)
    store.close()


def test_neo4j_store_created():
    mock_driver = MagicMock()
    mock_driver.verify_connectivity = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock()
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch("knowledge_graph.neo4j_store.GraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        store = create_kg_store("neo4j")

    from knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore
    assert isinstance(store, Neo4jKnowledgeGraphStore)


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown KG store type"):
        create_kg_store("postgres")
