"""Knowledge Graph storage layer.

Provides a Protocol for pluggable backends and a default SQLite implementation
with FTS5 full-text search for pattern and decision lookups.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from knowledge_graph.models import (
    Action,
    Decision,
    EdgeType,
    KGEdge,
    Outcome,
    Pattern,
)


@runtime_checkable
class KnowledgeGraphStore(Protocol):
    def write_decision(self, decision: Decision) -> str: ...
    def write_action(self, action: Action) -> str: ...
    def write_outcome(self, outcome: Outcome) -> str: ...
    def write_pattern(self, pattern: Pattern) -> str: ...
    def write_edge(self, edge: KGEdge) -> str: ...

    def get_decision(self, decision_id: str) -> Decision | None: ...
    def get_action(self, action_id: str) -> Action | None: ...
    def get_outcome(self, outcome_id: str) -> Outcome | None: ...
    def get_pattern(self, pattern_id: str) -> Pattern | None: ...

    def get_edges(
        self, node_id: str, direction: str = "outgoing"
    ) -> list[KGEdge]: ...

    def update_pattern(
        self, pattern_id: str, **fields: object
    ) -> None: ...

    def search_patterns(self, query: str, limit: int = 5) -> list[Pattern]: ...
    def get_outcomes_for_action(self, action_id: str) -> list[Outcome]: ...
    def get_recent_outcomes(self, limit: int = 50) -> list[Outcome]: ...

    def close(self) -> None: ...


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent TEXT NOT NULL,
    pillar TEXT NOT NULL,
    context TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    reasoning TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_detail TEXT NOT NULL,
    target_system TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT DEFAULT '',
    baseline_value REAL,
    delta REAL,
    attribution_confidence REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (action_id) REFERENCES actions(id)
);

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    conditions TEXT DEFAULT '[]',
    recommended_actions TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.0,
    times_observed INTEGER DEFAULT 1,
    last_confirmed TEXT,
    outcome_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_actions_decision ON actions(decision_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_action ON outcomes(action_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON outcomes(timestamp DESC);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS patterns_fts USING fts5(
    id UNINDEXED, description, conditions, recommended_actions,
    content='patterns',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    id UNINDEXED, context, decision_text,
    content='decisions',
    content_rowid='rowid'
);
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS patterns_ai AFTER INSERT ON patterns BEGIN
    INSERT INTO patterns_fts(id, description, conditions, recommended_actions)
    VALUES (new.id, new.description, new.conditions, new.recommended_actions);
END;

CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
    INSERT INTO decisions_fts(id, context, decision_text)
    VALUES (new.id, new.context, new.decision_text);
END;
"""


class SQLiteKnowledgeGraphStore:
    """SQLite-backed knowledge graph with FTS5 full-text search."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.getenv(
            "KG_SQLITE_PATH", "./data/knowledge_graph.db"
        )
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.executescript(_FTS_SQL)
        self._conn.executescript(_FTS_TRIGGERS)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        try:
            self._conn.execute(
                "ALTER TABLE patterns ADD COLUMN last_confirmed TEXT"
            )
        except sqlite3.OperationalError:
            pass

    # -- Write operations --

    def write_decision(self, decision: Decision) -> str:
        self._conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                decision.id,
                decision.timestamp.isoformat(),
                decision.agent,
                decision.pillar,
                decision.context,
                decision.decision_text,
                decision.confidence,
                decision.reasoning,
                json.dumps(decision.metadata),
            ),
        )
        self._conn.commit()
        return decision.id

    def write_action(self, action: Action) -> str:
        self._conn.execute(
            "INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                action.id,
                action.timestamp.isoformat(),
                action.decision_id,
                action.agent,
                action.action_type,
                action.action_detail,
                action.target_system,
                action.status,
                json.dumps(action.metadata),
            ),
        )
        self._conn.commit()
        return action.id

    def write_outcome(self, outcome: Outcome) -> str:
        self._conn.execute(
            "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                outcome.id,
                outcome.timestamp.isoformat(),
                outcome.action_id,
                outcome.metric_name,
                outcome.metric_value,
                outcome.metric_unit,
                outcome.baseline_value,
                outcome.delta,
                outcome.attribution_confidence,
                json.dumps(outcome.metadata),
            ),
        )
        self._conn.commit()
        return outcome.id

    def write_pattern(self, pattern: Pattern) -> str:
        self._conn.execute(
            "INSERT INTO patterns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                pattern.id,
                pattern.timestamp.isoformat(),
                pattern.pattern_type,
                pattern.description,
                json.dumps(pattern.conditions),
                json.dumps(pattern.recommended_actions),
                pattern.confidence,
                pattern.times_observed,
                pattern.last_confirmed.isoformat(),
                json.dumps(pattern.outcome_ids),
                json.dumps(pattern.metadata),
            ),
        )
        self._conn.commit()
        return pattern.id

    def write_edge(self, edge: KGEdge) -> str:
        self._conn.execute(
            "INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
            (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                edge.timestamp.isoformat(),
                edge.weight,
                json.dumps(edge.metadata),
            ),
        )
        self._conn.commit()
        return edge.id

    _UPDATABLE_PATTERN_FIELDS = {
        "confidence", "times_observed", "last_confirmed",
    }

    def update_pattern(self, pattern_id: str, **fields: object) -> None:
        valid = {k: v for k, v in fields.items() if k in self._UPDATABLE_PATTERN_FIELDS}
        if not valid:
            return
        set_clause = ", ".join(f"{k} = ?" for k in valid)
        values = list(valid.values())
        values.append(pattern_id)
        self._conn.execute(
            f"UPDATE patterns SET {set_clause} WHERE id = ?", values,
        )
        self._conn.commit()

    # -- Read operations --

    def _row_to_decision(self, row: sqlite3.Row) -> Decision:
        return Decision(
            id=row["id"],
            timestamp=row["timestamp"],
            agent=row["agent"],
            pillar=row["pillar"],
            context=row["context"],
            decision_text=row["decision_text"],
            confidence=row["confidence"],
            reasoning=row["reasoning"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_action(self, row: sqlite3.Row) -> Action:
        return Action(
            id=row["id"],
            timestamp=row["timestamp"],
            decision_id=row["decision_id"],
            agent=row["agent"],
            action_type=row["action_type"],
            action_detail=row["action_detail"],
            target_system=row["target_system"],
            status=row["status"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_outcome(self, row: sqlite3.Row) -> Outcome:
        return Outcome(
            id=row["id"],
            timestamp=row["timestamp"],
            action_id=row["action_id"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            metric_unit=row["metric_unit"],
            baseline_value=row["baseline_value"],
            delta=row["delta"],
            attribution_confidence=row["attribution_confidence"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_pattern(self, row: sqlite3.Row) -> Pattern:
        return Pattern(
            id=row["id"],
            timestamp=row["timestamp"],
            pattern_type=row["pattern_type"],
            description=row["description"],
            conditions=json.loads(row["conditions"]),
            recommended_actions=json.loads(row["recommended_actions"]),
            confidence=row["confidence"],
            times_observed=row["times_observed"],
            last_confirmed=row["last_confirmed"] or row["timestamp"],
            outcome_ids=json.loads(row["outcome_ids"]),
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> KGEdge:
        return KGEdge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            timestamp=row["timestamp"],
            weight=row["weight"],
            metadata=json.loads(row["metadata"]),
        )

    def get_decision(self, decision_id: str) -> Decision | None:
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        return self._row_to_decision(row) if row else None

    def get_action(self, action_id: str) -> Action | None:
        row = self._conn.execute(
            "SELECT * FROM actions WHERE id = ?", (action_id,)
        ).fetchone()
        return self._row_to_action(row) if row else None

    def get_outcome(self, outcome_id: str) -> Outcome | None:
        row = self._conn.execute(
            "SELECT * FROM outcomes WHERE id = ?", (outcome_id,)
        ).fetchone()
        return self._row_to_outcome(row) if row else None

    def get_pattern(self, pattern_id: str) -> Pattern | None:
        row = self._conn.execute(
            "SELECT * FROM patterns WHERE id = ?", (pattern_id,)
        ).fetchone()
        return self._row_to_pattern(row) if row else None

    def get_edges(
        self, node_id: str, direction: str = "outgoing"
    ) -> list[KGEdge]:
        if direction == "outgoing":
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (node_id,)
            ).fetchall()
        elif direction == "incoming":
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ?", (node_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape special FTS5 characters and wrap each word as a term."""
        words = []
        for word in query.split():
            cleaned = "".join(c for c in word if c.isalnum() or c == "_")
            if cleaned:
                words.append(f'"{cleaned}"')
        return " OR ".join(words) if words else '""'

    _DECAY_HALF_LIFE_DAYS = 30.0

    def _apply_confidence_decay(self, pattern: Pattern) -> Pattern:
        """Apply time-based exponential decay to pattern confidence.

        Uses a 30-day half-life so patterns not confirmed in ~90 days
        drop to ~12.5% of their stored confidence.
        """
        now = datetime.now(timezone.utc)
        confirmed = pattern.last_confirmed
        if confirmed.tzinfo is None:
            confirmed = confirmed.replace(tzinfo=timezone.utc)
        age_days = (now - confirmed).total_seconds() / 86400.0
        if age_days <= 0:
            return pattern
        decay_factor = math.exp(-0.693 * age_days / self._DECAY_HALF_LIFE_DAYS)
        effective = pattern.confidence * decay_factor
        return pattern.model_copy(update={"confidence": round(effective, 4)})

    def search_patterns(self, query: str, limit: int = 5) -> list[Pattern]:
        safe_query = self._sanitize_fts_query(query)
        try:
            rows = self._conn.execute(
                """
                SELECT p.* FROM patterns p
                JOIN patterns_fts fts ON p.id = fts.id
                WHERE patterns_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()
        except Exception:
            rows = self._conn.execute(
                "SELECT * FROM patterns LIMIT ?", (limit,)
            ).fetchall()
        patterns = [self._row_to_pattern(r) for r in rows]
        return [self._apply_confidence_decay(p) for p in patterns]

    def get_outcomes_for_action(self, action_id: str) -> list[Outcome]:
        rows = self._conn.execute(
            "SELECT * FROM outcomes WHERE action_id = ? ORDER BY timestamp",
            (action_id,),
        ).fetchall()
        return [self._row_to_outcome(r) for r in rows]

    def get_recent_outcomes(self, limit: int = 50) -> list[Outcome]:
        rows = self._conn.execute(
            "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_outcome(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
