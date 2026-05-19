"""Tests for the Knowledge Graph store, writer, and reader."""

import tempfile
import os

import pytest

from knowledge_graph.models import (
    Action,
    Decision,
    EdgeType,
    KGEdge,
    Outcome,
    Pattern,
)
from knowledge_graph.store import SQLiteKnowledgeGraphStore
from knowledge_graph.writer import KGWriter
from knowledge_graph.reader import KGReader


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kg.db")
        s = SQLiteKnowledgeGraphStore(db_path=db_path)
        yield s
        s.close()


@pytest.fixture
def writer(store):
    return KGWriter(store)


@pytest.fixture
def reader(store):
    return KGReader(store)


class TestSQLiteKnowledgeGraphStore:
    def test_write_and_read_decision(self, store):
        d = Decision(
            agent="insight_agent",
            pillar="INSIGHT",
            context="User asked about revenue trends",
            decision_text="Analyze quarterly revenue data",
            confidence=0.85,
        )
        store.write_decision(d)
        result = store.get_decision(d.id)
        assert result is not None
        assert result.agent == "insight_agent"
        assert result.decision_text == "Analyze quarterly revenue data"
        assert result.confidence == 0.85

    def test_write_and_read_action(self, store):
        d = Decision(
            agent="act_agent", pillar="ACT",
            context="test", decision_text="test",
        )
        store.write_decision(d)
        a = Action(
            decision_id=d.id,
            agent="act_agent",
            action_type="api_call",
            action_detail="Called CRM endpoint",
            target_system="salesforce",
        )
        store.write_action(a)
        result = store.get_action(a.id)
        assert result is not None
        assert result.action_type == "api_call"
        assert result.target_system == "salesforce"

    def test_write_and_read_outcome(self, store):
        d = Decision(agent="a", pillar="p", context="c", decision_text="d")
        store.write_decision(d)
        a = Action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        store.write_action(a)
        o = Outcome(
            action_id=a.id,
            metric_name="conversion_rate",
            metric_value=0.045,
            metric_unit="ratio",
            baseline_value=0.03,
            delta=0.015,
        )
        store.write_outcome(o)
        result = store.get_outcome(o.id)
        assert result is not None
        assert result.metric_name == "conversion_rate"
        assert result.delta == 0.015

    def test_write_and_search_pattern(self, store):
        p = Pattern(
            pattern_type="correlation",
            description="Email campaigns with personalized subject lines increase open rates by 25%",
            conditions=["email campaign", "personalized subject"],
            recommended_actions=["Use personalization in subject lines"],
            confidence=0.9,
            times_observed=5,
        )
        store.write_pattern(p)
        results = store.search_patterns("email personalized")
        assert len(results) >= 1
        assert results[0].id == p.id

    def test_get_outcomes_for_action(self, store):
        d = Decision(agent="a", pillar="p", context="c", decision_text="d")
        store.write_decision(d)
        a = Action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        store.write_action(a)
        for i in range(3):
            store.write_outcome(Outcome(
                action_id=a.id,
                metric_name=f"metric_{i}",
                metric_value=float(i),
            ))
        outcomes = store.get_outcomes_for_action(a.id)
        assert len(outcomes) == 3

    def test_edges(self, store):
        edge = KGEdge(
            source_id="src-1",
            target_id="tgt-1",
            edge_type=EdgeType.DECIDED_ACTION,
        )
        store.write_edge(edge)
        outgoing = store.get_edges("src-1", direction="outgoing")
        assert len(outgoing) == 1
        incoming = store.get_edges("tgt-1", direction="incoming")
        assert len(incoming) == 1

    def test_get_recent_outcomes(self, store):
        d = Decision(agent="a", pillar="p", context="c", decision_text="d")
        store.write_decision(d)
        a = Action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        store.write_action(a)
        for i in range(5):
            store.write_outcome(Outcome(
                action_id=a.id,
                metric_name=f"m{i}",
                metric_value=float(i),
            ))
        recent = store.get_recent_outcomes(limit=3)
        assert len(recent) == 3


class TestKGWriter:
    def test_record_decision(self, writer, store):
        d = writer.record_decision(
            agent="insight_agent",
            pillar="INSIGHT",
            context="test context",
            decision_text="test decision",
        )
        assert store.get_decision(d.id) is not None

    def test_record_action_creates_edge(self, writer, store):
        d = writer.record_decision(
            agent="a", pillar="p", context="c", decision_text="d",
        )
        a = writer.record_action(
            decision_id=d.id,
            agent="act_agent",
            action_type="api_call",
            action_detail="test",
        )
        assert store.get_action(a.id) is not None
        edges = store.get_edges(d.id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.DECIDED_ACTION

    def test_record_outcome_creates_edge(self, writer, store):
        d = writer.record_decision(
            agent="a", pillar="p", context="c", decision_text="d",
        )
        a = writer.record_action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        o = writer.record_outcome(
            action_id=a.id,
            metric_name="revenue",
            metric_value=1000.0,
        )
        edges = store.get_edges(a.id, direction="outgoing")
        assert any(e.edge_type == EdgeType.PRODUCED_OUTCOME for e in edges)

    def test_record_pattern_creates_edges(self, writer, store):
        d = writer.record_decision(
            agent="a", pillar="p", context="c", decision_text="d",
        )
        a = writer.record_action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        o1 = writer.record_outcome(
            action_id=a.id, metric_name="m1", metric_value=1.0,
        )
        o2 = writer.record_outcome(
            action_id=a.id, metric_name="m2", metric_value=2.0,
        )
        p = writer.record_pattern(
            pattern_type="correlation",
            description="Test pattern",
            outcome_ids=[o1.id, o2.id],
        )
        edges = store.get_edges(p.id, direction="incoming")
        assert len(edges) == 2

    def test_flywheel_loop(self, writer, store):
        """Test the full flywheel: Decision -> Action -> Outcome -> Pattern -> Decision."""
        d1 = writer.record_decision(
            agent="insight", pillar="INSIGHT",
            context="revenue drop", decision_text="investigate Q3",
        )
        a = writer.record_action(
            decision_id=d1.id, agent="act",
            action_type="query", action_detail="ran analysis",
        )
        o = writer.record_outcome(
            action_id=a.id, metric_name="revenue_delta",
            metric_value=-0.15,
        )
        p = writer.record_pattern(
            pattern_type="causal",
            description="Q3 revenue drops correlate with seasonal demand",
            outcome_ids=[o.id],
        )
        d2 = writer.record_decision(
            agent="strategy", pillar="STRATEGY",
            context="seasonal planning",
            decision_text="adjust Q3 pricing",
        )
        edge = writer.link_pattern_to_decision(p.id, d2.id)
        assert edge.edge_type == EdgeType.INFORMED_DECISION


class TestKGReader:
    def test_search_patterns(self, writer, reader):
        writer.record_pattern(
            pattern_type="correlation",
            description="Higher engagement during morning hours",
            conditions=["time of day", "morning"],
        )
        results = reader.search_patterns("morning engagement")
        assert len(results) >= 1

    def test_audit_chain(self, writer, reader, store):
        d = writer.record_decision(
            agent="insight", pillar="INSIGHT",
            context="test", decision_text="test",
        )
        a = writer.record_action(
            decision_id=d.id, agent="act",
            action_type="t", action_detail="d",
        )
        o = writer.record_outcome(
            action_id=a.id, metric_name="m", metric_value=1.0,
        )
        chain = reader.get_audit_chain(d.id)
        assert len(chain["decisions"]) == 1
        assert len(chain["actions"]) == 1
        assert len(chain["outcomes"]) == 1
