"""Tests for learn agent tools — pattern update persistence and confidence decay."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from knowledge_graph.models import Pattern
from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter


@pytest.fixture()
def kg_db(tmp_path):
    db_path = str(tmp_path / "test_kg.db")
    os.environ["KG_SQLITE_PATH"] = db_path
    yield db_path
    os.environ.pop("KG_SQLITE_PATH", None)


class TestUpdatePatternConfidence:
    def test_persists_confidence_update(self, kg_db):
        from learn_agent.tools import update_pattern_confidence

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="causal",
            description="Price drops boost sales",
            confidence=0.5,
        )

        result = update_pattern_confidence.invoke({
            "pattern_id": pattern.id,
            "new_outcome_ids": ["outcome-1", "outcome-2"],
            "adjustment": 0.2,
        })
        data = json.loads(result)
        assert data["new_confidence"] == 0.7
        assert data["times_observed"] == 3

        store2 = SQLiteKnowledgeGraphStore(db_path=kg_db)
        updated = store2.get_pattern(pattern.id)
        assert updated.confidence == 0.7
        assert updated.times_observed == 3

    def test_creates_edges_for_new_outcomes(self, kg_db):
        from learn_agent.tools import update_pattern_confidence

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="temporal",
            description="Monday spikes",
            confidence=0.3,
        )

        update_pattern_confidence.invoke({
            "pattern_id": pattern.id,
            "new_outcome_ids": ["o1"],
        })

        edges = store.get_edges(pattern.id, direction="incoming")
        assert any(e.source_id == "o1" for e in edges)

    def test_returns_error_for_missing_pattern(self, kg_db):
        from learn_agent.tools import update_pattern_confidence

        result = update_pattern_confidence.invoke({
            "pattern_id": "nonexistent",
            "new_outcome_ids": [],
        })
        data = json.loads(result)
        assert "error" in data


class TestIncorporateFeedback:
    def test_confirm_boosts_confidence(self, kg_db):
        from learn_agent.tools import incorporate_feedback

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="causal",
            description="Test pattern",
            confidence=0.5,
        )

        result = incorporate_feedback.invoke({
            "pattern_id": pattern.id,
            "feedback_type": "confirm",
            "feedback_detail": "Verified by domain expert",
        })
        data = json.loads(result)
        assert data["new_confidence"] == 0.65

        store2 = SQLiteKnowledgeGraphStore(db_path=kg_db)
        updated = store2.get_pattern(pattern.id)
        assert updated.confidence == 0.65

    def test_reject_drops_confidence(self, kg_db):
        from learn_agent.tools import incorporate_feedback

        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="causal",
            description="Bad pattern",
            confidence=0.5,
        )

        result = incorporate_feedback.invoke({
            "pattern_id": pattern.id,
            "feedback_type": "reject",
            "feedback_detail": "Not valid",
        })
        data = json.loads(result)
        assert data["new_confidence"] == 0.2


class TestConfidenceDecay:
    def test_recent_pattern_no_decay(self, kg_db):
        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="causal",
            description="Fresh pattern for decay test",
            confidence=0.8,
        )
        results = store.search_patterns("Fresh pattern decay", limit=5)
        matched = [p for p in results if p.id == pattern.id]
        assert len(matched) == 1
        assert matched[0].confidence >= 0.75

    def test_old_pattern_decays(self, kg_db):
        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="causal",
            description="Stale pattern for decay test",
            confidence=0.8,
        )
        sixty_days_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        store.update_pattern(pattern.id, last_confirmed=sixty_days_ago)

        results = store.search_patterns("Stale pattern decay", limit=5)
        matched = [p for p in results if p.id == pattern.id]
        assert len(matched) == 1
        assert matched[0].confidence < 0.25

    def test_ninety_day_pattern_near_zero(self, kg_db):
        store = SQLiteKnowledgeGraphStore(db_path=kg_db)
        writer = KGWriter(store)
        pattern = writer.record_pattern(
            pattern_type="temporal",
            description="Ancient pattern for decay test",
            confidence=1.0,
        )
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        store.update_pattern(pattern.id, last_confirmed=ninety_days_ago)

        results = store.search_patterns("Ancient pattern decay", limit=5)
        matched = [p for p in results if p.id == pattern.id]
        assert len(matched) == 1
        assert matched[0].confidence < 0.15
