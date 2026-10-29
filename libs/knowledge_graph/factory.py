"""Factory for creating KnowledgeGraphStore instances.

Reads ``KG_STORE_TYPE`` to decide which backend to use and caches the
resulting store so that repeated calls within the same process reuse the
same connection / driver.
"""

from __future__ import annotations

import functools
import os

from knowledge_graph.store import KnowledgeGraphStore


@functools.lru_cache(maxsize=4)
def create_kg_store(store_type: str | None = None) -> KnowledgeGraphStore:
    stype = (store_type or os.getenv("KG_STORE_TYPE", "sqlite")).lower()

    if stype == "sqlite":
        from knowledge_graph.store import SQLiteKnowledgeGraphStore

        db_path = os.getenv("KG_SQLITE_PATH", "./data/knowledge_graph.db")
        return SQLiteKnowledgeGraphStore(db_path=db_path)

    if stype == "neo4j":
        from knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

        return Neo4jKnowledgeGraphStore()

    raise ValueError(f"Unknown KG store type: {stype!r}. Supported: sqlite, neo4j")
