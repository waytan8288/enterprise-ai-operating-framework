"""Write-path helpers for agents to record nodes and edges in the KG.

Each agent type has a dedicated helper that creates the appropriate node
and connecting edge in a single call.
"""

from __future__ import annotations

from knowledge_graph.models import (
    Action,
    Decision,
    EdgeType,
    KGEdge,
    Outcome,
    Pattern,
)
from knowledge_graph.store import KnowledgeGraphStore


class KGWriter:
    def __init__(self, store: KnowledgeGraphStore) -> None:
        self._store = store

    def record_decision(
        self,
        *,
        agent: str,
        pillar: str,
        context: str,
        decision_text: str,
        confidence: float = 0.0,
        reasoning: str = "",
        metadata: dict | None = None,
    ) -> Decision:
        decision = Decision(
            agent=agent,
            pillar=pillar,
            context=context,
            decision_text=decision_text,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata or {},
        )
        self._store.write_decision(decision)
        return decision

    def record_action(
        self,
        *,
        decision_id: str,
        agent: str,
        action_type: str,
        action_detail: str,
        target_system: str = "",
        status: str = "pending",
        metadata: dict | None = None,
    ) -> Action:
        action = Action(
            decision_id=decision_id,
            agent=agent,
            action_type=action_type,
            action_detail=action_detail,
            target_system=target_system,
            status=status,
            metadata=metadata or {},
        )
        self._store.write_action(action)
        self._store.write_edge(
            KGEdge(
                source_id=decision_id,
                target_id=action.id,
                edge_type=EdgeType.DECIDED_ACTION,
            )
        )
        return action

    def record_outcome(
        self,
        *,
        action_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str = "",
        baseline_value: float | None = None,
        delta: float | None = None,
        attribution_confidence: float = 0.0,
        metadata: dict | None = None,
    ) -> Outcome:
        outcome = Outcome(
            action_id=action_id,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            baseline_value=baseline_value,
            delta=delta,
            attribution_confidence=attribution_confidence,
            metadata=metadata or {},
        )
        self._store.write_outcome(outcome)
        self._store.write_edge(
            KGEdge(
                source_id=action_id,
                target_id=outcome.id,
                edge_type=EdgeType.PRODUCED_OUTCOME,
            )
        )
        return outcome

    def record_pattern(
        self,
        *,
        pattern_type: str,
        description: str,
        conditions: list[str] | None = None,
        recommended_actions: list[str] | None = None,
        confidence: float = 0.0,
        outcome_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Pattern:
        outcome_ids = outcome_ids or []
        pattern = Pattern(
            pattern_type=pattern_type,
            description=description,
            conditions=conditions or [],
            recommended_actions=recommended_actions or [],
            confidence=confidence,
            outcome_ids=outcome_ids,
            metadata=metadata or {},
        )
        self._store.write_pattern(pattern)
        for oid in outcome_ids:
            self._store.write_edge(
                KGEdge(
                    source_id=oid,
                    target_id=pattern.id,
                    edge_type=EdgeType.REVEALED_PATTERN,
                )
            )
        return pattern

    def link_pattern_to_decision(
        self, pattern_id: str, decision_id: str
    ) -> KGEdge:
        edge = KGEdge(
            source_id=pattern_id,
            target_id=decision_id,
            edge_type=EdgeType.INFORMED_DECISION,
        )
        self._store.write_edge(edge)
        return edge
