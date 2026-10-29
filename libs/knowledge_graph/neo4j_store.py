"""Neo4j-backed Knowledge Graph store.

Uses native Neo4j relationships for edges and fulltext indexes
for pattern search.  Requires the ``neo4j`` Python driver.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from neo4j import GraphDatabase

from knowledge_graph._decay import apply_confidence_decay
from knowledge_graph.models import (
    Action,
    Decision,
    EdgeType,
    KGEdge,
    Outcome,
    Pattern,
)

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}[\]^"~*?:\\/])')

_CONSTRAINTS = [
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT outcome_id IF NOT EXISTS FOR (o:Outcome) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
]

_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX pattern_search IF NOT EXISTS "
    "FOR (p:Pattern) ON EACH [p.description, p.conditions_text, p.recommended_actions_text]"
)

_EDGE_TYPE_TO_REL = {
    EdgeType.DECIDED_ACTION: "DECIDED_ACTION",
    EdgeType.PRODUCED_OUTCOME: "PRODUCED_OUTCOME",
    EdgeType.REVEALED_PATTERN: "REVEALED_PATTERN",
    EdgeType.INFORMED_DECISION: "INFORMED_DECISION",
}


class Neo4jKnowledgeGraphStore:
    """Neo4j Community Edition backend for the Enterprise Knowledge Graph."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "changeme")
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password),
        )
        self._driver.verify_connectivity()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._driver.session() as session:
            for stmt in _CONSTRAINTS:
                session.run(stmt)
            session.run(_FULLTEXT_INDEX)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_decision(self, decision: Decision) -> str:
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "CREATE (d:Decision {id: $id, timestamp: $ts, agent: $agent, "
                    "pillar: $pillar, context: $context, decision_text: $dt, "
                    "confidence: $conf, reasoning: $reasoning, metadata: $meta})",
                    id=decision.id,
                    ts=decision.timestamp.isoformat(),
                    agent=decision.agent,
                    pillar=decision.pillar,
                    context=decision.context,
                    dt=decision.decision_text,
                    conf=decision.confidence,
                    reasoning=decision.reasoning,
                    meta=json.dumps(decision.metadata),
                ),
            )
        return decision.id

    def write_action(self, action: Action) -> str:
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "CREATE (a:Action {id: $id, timestamp: $ts, decision_id: $did, "
                    "agent: $agent, action_type: $atype, action_detail: $adetail, "
                    "target_system: $tsys, status: $status, metadata: $meta})",
                    id=action.id,
                    ts=action.timestamp.isoformat(),
                    did=action.decision_id,
                    agent=action.agent,
                    atype=action.action_type,
                    adetail=action.action_detail,
                    tsys=action.target_system,
                    status=action.status,
                    meta=json.dumps(action.metadata),
                ),
            )
        return action.id

    def write_outcome(self, outcome: Outcome) -> str:
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "CREATE (o:Outcome {id: $id, timestamp: $ts, action_id: $aid, "
                    "metric_name: $mn, metric_value: $mv, metric_unit: $mu, "
                    "baseline_value: $bv, delta: $delta, "
                    "attribution_confidence: $ac, metadata: $meta})",
                    id=outcome.id,
                    ts=outcome.timestamp.isoformat(),
                    aid=outcome.action_id,
                    mn=outcome.metric_name,
                    mv=outcome.metric_value,
                    mu=outcome.metric_unit,
                    bv=outcome.baseline_value,
                    delta=outcome.delta,
                    ac=outcome.attribution_confidence,
                    meta=json.dumps(outcome.metadata),
                ),
            )
        return outcome.id

    def write_pattern(self, pattern: Pattern) -> str:
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    "CREATE (p:Pattern {id: $id, timestamp: $ts, "
                    "pattern_type: $pt, description: $desc, "
                    "conditions: $conds, conditions_text: $conds_text, "
                    "recommended_actions: $ra, recommended_actions_text: $ra_text, "
                    "confidence: $conf, times_observed: $to, "
                    "last_confirmed: $lc, outcome_ids: $oids, metadata: $meta})",
                    id=pattern.id,
                    ts=pattern.timestamp.isoformat(),
                    pt=pattern.pattern_type,
                    desc=pattern.description,
                    conds=pattern.conditions,
                    conds_text=", ".join(pattern.conditions),
                    ra=pattern.recommended_actions,
                    ra_text=", ".join(pattern.recommended_actions),
                    conf=pattern.confidence,
                    to=pattern.times_observed,
                    lc=pattern.last_confirmed.isoformat(),
                    oids=pattern.outcome_ids,
                    meta=json.dumps(pattern.metadata),
                ),
            )
        return pattern.id

    def write_edge(self, edge: KGEdge) -> str:
        rel_type = _EDGE_TYPE_TO_REL[edge.edge_type]
        cypher = (
            "MATCH (s) WHERE s.id = $src "
            "MATCH (t) WHERE t.id = $tgt "
            f"CREATE (s)-[r:{rel_type} {{id: $id, timestamp: $ts, "
            "weight: $w, metadata: $meta}]->(t)"
        )
        with self._driver.session() as s:
            s.execute_write(
                lambda tx: tx.run(
                    cypher,
                    src=edge.source_id,
                    tgt=edge.target_id,
                    id=edge.id,
                    ts=edge.timestamp.isoformat(),
                    w=edge.weight,
                    meta=json.dumps(edge.metadata),
                ),
            )
        return edge.id

    _UPDATABLE_PATTERN_FIELDS = {
        "confidence", "times_observed", "last_confirmed",
    }

    def update_pattern(self, pattern_id: str, **fields: object) -> None:
        valid = {
            k: v for k, v in fields.items()
            if k in self._UPDATABLE_PATTERN_FIELDS
        }
        if not valid:
            return
        set_parts = [f"p.{k} = ${k}" for k in valid]
        cypher = f"MATCH (p:Pattern {{id: $pid}}) SET {', '.join(set_parts)}"
        params: dict[str, Any] = {"pid": pattern_id, **valid}
        with self._driver.session() as s:
            s.execute_write(lambda tx: tx.run(cypher, **params))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @staticmethod
    def _props(record: Any, key: str = "n") -> dict[str, Any]:
        return dict(record[key])

    def _to_decision(self, props: dict[str, Any]) -> Decision:
        return Decision(
            id=props["id"],
            timestamp=props["timestamp"],
            agent=props["agent"],
            pillar=props["pillar"],
            context=props["context"],
            decision_text=props["decision_text"],
            confidence=props["confidence"],
            reasoning=props["reasoning"],
            metadata=json.loads(props["metadata"]),
        )

    def _to_action(self, props: dict[str, Any]) -> Action:
        return Action(
            id=props["id"],
            timestamp=props["timestamp"],
            decision_id=props["decision_id"],
            agent=props["agent"],
            action_type=props["action_type"],
            action_detail=props["action_detail"],
            target_system=props["target_system"],
            status=props["status"],
            metadata=json.loads(props["metadata"]),
        )

    def _to_outcome(self, props: dict[str, Any]) -> Outcome:
        return Outcome(
            id=props["id"],
            timestamp=props["timestamp"],
            action_id=props["action_id"],
            metric_name=props["metric_name"],
            metric_value=props["metric_value"],
            metric_unit=props.get("metric_unit", ""),
            baseline_value=props.get("baseline_value"),
            delta=props.get("delta"),
            attribution_confidence=props.get("attribution_confidence", 0.0),
            metadata=json.loads(props.get("metadata", "{}")),
        )

    def _to_pattern(self, props: dict[str, Any]) -> Pattern:
        return Pattern(
            id=props["id"],
            timestamp=props["timestamp"],
            pattern_type=props["pattern_type"],
            description=props["description"],
            conditions=list(props["conditions"]),
            recommended_actions=list(props["recommended_actions"]),
            confidence=props["confidence"],
            times_observed=props["times_observed"],
            last_confirmed=props["last_confirmed"] or props["timestamp"],
            outcome_ids=list(props["outcome_ids"]),
            metadata=json.loads(props["metadata"]),
        )

    def _to_edge(self, record: Any) -> KGEdge:
        r = record["r"]
        rel_type = r.type
        edge_type = next(
            (et for et, rt in _EDGE_TYPE_TO_REL.items() if rt == rel_type),
            EdgeType.DECIDED_ACTION,
        )
        props = dict(r)
        return KGEdge(
            id=props["id"],
            source_id=record["src"],
            target_id=record["tgt"],
            edge_type=edge_type,
            timestamp=props["timestamp"],
            weight=props["weight"],
            metadata=json.loads(props["metadata"]),
        )

    def get_decision(self, decision_id: str) -> Decision | None:
        with self._driver.session() as s:
            result = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (n:Decision {id: $id}) RETURN n", id=decision_id,
                ).single(),
            )
        if result is None:
            return None
        return self._to_decision(self._props(result))

    def get_action(self, action_id: str) -> Action | None:
        with self._driver.session() as s:
            result = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (n:Action {id: $id}) RETURN n", id=action_id,
                ).single(),
            )
        if result is None:
            return None
        return self._to_action(self._props(result))

    def get_outcome(self, outcome_id: str) -> Outcome | None:
        with self._driver.session() as s:
            result = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (n:Outcome {id: $id}) RETURN n", id=outcome_id,
                ).single(),
            )
        if result is None:
            return None
        return self._to_outcome(self._props(result))

    def get_pattern(self, pattern_id: str) -> Pattern | None:
        with self._driver.session() as s:
            result = s.execute_read(
                lambda tx: tx.run(
                    "MATCH (n:Pattern {id: $id}) RETURN n", id=pattern_id,
                ).single(),
            )
        if result is None:
            return None
        return self._to_pattern(self._props(result))

    def get_edges(
        self, node_id: str, direction: str = "outgoing",
    ) -> list[KGEdge]:
        if direction == "outgoing":
            cypher = (
                "MATCH (s {id: $nid})-[r]->(t) "
                "RETURN r, s.id AS src, t.id AS tgt"
            )
        elif direction == "incoming":
            cypher = (
                "MATCH (s)-[r]->(t {id: $nid}) "
                "RETURN r, s.id AS src, t.id AS tgt"
            )
        else:
            cypher = (
                "MATCH (s)-[r]->(t) "
                "WHERE s.id = $nid OR t.id = $nid "
                "RETURN r, s.id AS src, t.id AS tgt"
            )
        with self._driver.session() as s:
            records = s.execute_read(
                lambda tx: list(tx.run(cypher, nid=node_id)),
            )
        return [self._to_edge(r) for r in records]

    @staticmethod
    def _sanitize_lucene(query: str) -> str:
        escaped = _LUCENE_SPECIAL.sub(r"\\\1", query)
        terms = escaped.split()
        return " OR ".join(terms) if terms else "*"

    def search_patterns(self, query: str, limit: int = 5) -> list[Pattern]:
        safe = self._sanitize_lucene(query)
        with self._driver.session() as s:
            records = s.execute_read(
                lambda tx: list(
                    tx.run(
                        "CALL db.index.fulltext.queryNodes('pattern_search', $q) "
                        "YIELD node, score "
                        "RETURN node AS n "
                        "ORDER BY score DESC LIMIT $limit",
                        q=safe,
                        limit=limit,
                    ),
                ),
            )
        patterns = [self._to_pattern(self._props(r)) for r in records]
        return [apply_confidence_decay(p) for p in patterns]

    def get_outcomes_for_action(self, action_id: str) -> list[Outcome]:
        with self._driver.session() as s:
            records = s.execute_read(
                lambda tx: list(
                    tx.run(
                        "MATCH (n:Outcome {action_id: $aid}) "
                        "RETURN n ORDER BY n.timestamp",
                        aid=action_id,
                    ),
                ),
            )
        return [self._to_outcome(self._props(r)) for r in records]

    def get_recent_outcomes(self, limit: int = 50) -> list[Outcome]:
        with self._driver.session() as s:
            records = s.execute_read(
                lambda tx: list(
                    tx.run(
                        "MATCH (n:Outcome) RETURN n "
                        "ORDER BY n.timestamp DESC LIMIT $limit",
                        limit=limit,
                    ),
                ),
            )
        return [self._to_outcome(self._props(r)) for r in records]

    def close(self) -> None:
        self._driver.close()
