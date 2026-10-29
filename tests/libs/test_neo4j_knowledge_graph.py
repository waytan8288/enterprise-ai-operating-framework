"""Tests for the Neo4j Knowledge Graph store.

Requires a running Neo4j instance (docker compose up -d).
Skipped automatically when Neo4j is unreachable.
"""

from __future__ import annotations

import pytest

from knowledge_graph.models import (
    Action,
    Decision,
    EdgeType,
    KGEdge,
    Outcome,
    Pattern,
)

neo4j = pytest.importorskip("neo4j")


def _neo4j_available() -> bool:
    try:
        driver = neo4j.GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "changeme"),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not reachable"),
]


@pytest.fixture
def store():
    from knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore

    s = Neo4jKnowledgeGraphStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="changeme",
    )
    # Clean slate for each test
    with s._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CALL db.awaitIndexes()")
    yield s
    # Clean up after test
    with s._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    s.close()


class TestNeo4jKnowledgeGraphStore:
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
        # Wait for fulltext index to catch up
        with store._driver.session() as session:
            session.run("CALL db.awaitIndexes()")
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

    def test_edges_as_relationships(self, store):
        d = Decision(agent="a", pillar="p", context="c", decision_text="d")
        store.write_decision(d)
        a = Action(
            decision_id=d.id, agent="a",
            action_type="t", action_detail="d",
        )
        store.write_action(a)
        edge = KGEdge(
            source_id=d.id,
            target_id=a.id,
            edge_type=EdgeType.DECIDED_ACTION,
        )
        store.write_edge(edge)
        outgoing = store.get_edges(d.id, direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0].edge_type == EdgeType.DECIDED_ACTION
        incoming = store.get_edges(a.id, direction="incoming")
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

    def test_search_with_special_characters(self, store):
        p = Pattern(
            pattern_type="test",
            description="Revenue increase of +25% after Q3 campaign",
            conditions=["Q3"],
            recommended_actions=["repeat campaign"],
            confidence=0.8,
        )
        store.write_pattern(p)
        with store._driver.session() as session:
            session.run("CALL db.awaitIndexes()")
        results = store.search_patterns("+25% revenue Q3")
        assert len(results) >= 1
