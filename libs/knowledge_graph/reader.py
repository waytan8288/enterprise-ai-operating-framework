"""Read-path helpers for querying the Knowledge Graph.

Primary consumers:
- strategy_agent: pattern similarity search
- insight_agent: historical outcome baselines
- govern_agent: full audit trail traversal
"""

from __future__ import annotations

from knowledge_graph.models import Action, Decision, KGEdge, Outcome, Pattern
from knowledge_graph.store import KnowledgeGraphStore


class KGReader:
    def __init__(self, store: KnowledgeGraphStore) -> None:
        self._store = store

    def search_patterns(self, query: str, limit: int = 5) -> list[Pattern]:
        return self._store.search_patterns(query, limit)

    def get_outcomes_for_action(self, action_id: str) -> list[Outcome]:
        return self._store.get_outcomes_for_action(action_id)

    def get_recent_outcomes(self, limit: int = 50) -> list[Outcome]:
        return self._store.get_recent_outcomes(limit)

    def get_audit_chain(self, node_id: str) -> dict:
        """Traverse the full provenance chain from any node.

        Returns a dict with the complete Decision -> Action -> Outcome -> Pattern
        chain reachable from the given node.
        """
        visited: set[str] = set()
        chain: dict[str, list] = {
            "decisions": [],
            "actions": [],
            "outcomes": [],
            "patterns": [],
            "edges": [],
        }
        self._traverse(node_id, visited, chain)
        return chain

    def _traverse(
        self, node_id: str, visited: set[str], chain: dict[str, list]
    ) -> None:
        if node_id in visited:
            return
        visited.add(node_id)

        for getter, key in [
            (self._store.get_decision, "decisions"),
            (self._store.get_action, "actions"),
            (self._store.get_outcome, "outcomes"),
            (self._store.get_pattern, "patterns"),
        ]:
            node = getter(node_id)
            if node is not None:
                chain[key].append(node)
                break

        edges = self._store.get_edges(node_id, direction="both")
        for edge in edges:
            chain["edges"].append(edge)
            next_id = (
                edge.target_id if edge.source_id == node_id else edge.source_id
            )
            self._traverse(next_id, visited, chain)
