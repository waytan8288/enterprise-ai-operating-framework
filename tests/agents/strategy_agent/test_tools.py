"""Tests for strategy agent tools — record_recommendation KG write."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


@pytest.fixture()
def kg_db(tmp_path):
    db_path = str(tmp_path / "test_kg.db")
    os.environ["KG_SQLITE_PATH"] = db_path
    yield db_path
    os.environ.pop("KG_SQLITE_PATH", None)


class TestRecordRecommendation:
    def test_records_decision_in_kg(self, kg_db):
        from strategy_agent.tools import record_recommendation

        result = record_recommendation.invoke({
            "recommendation": "Increase email frequency for high-value segments",
            "reasoning": "Pattern shows 2x conversion for weekly emails",
            "confidence": 0.85,
            "context": "Q2 revenue decline in segment A",
        })
        data = json.loads(result)
        assert "decision_id" in data
        assert data["linked_patterns"] == 0

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        decision = store.get_decision(data["decision_id"])
        assert decision is not None
        assert decision.agent == "strategy_agent"
        assert decision.pillar == "STRATEGY"
        assert decision.confidence == 0.85

    def test_links_patterns_to_decision(self, kg_db):
        from strategy_agent.tools import record_recommendation

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="correlation",
            description="Email frequency correlates with conversions",
            confidence=0.9,
        )

        result = record_recommendation.invoke({
            "recommendation": "Increase emails",
            "reasoning": "Based on pattern",
            "confidence": 0.8,
            "context": "test",
            "pattern_ids": [pattern.id],
        })
        data = json.loads(result)
        assert data["linked_patterns"] == 1

        edges = store.get_edges(pattern.id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].edge_type.value == "informed_decision"
        assert edges[0].target_id == data["decision_id"]
