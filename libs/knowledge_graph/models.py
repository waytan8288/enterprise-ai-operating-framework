"""Pydantic models for the Enterprise Knowledge Graph.

Four node types form the flywheel:
  Decision --decided_action--> Action --produced_outcome--> Outcome
  --revealed_pattern--> Pattern --informed_decision--> Decision
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class NodeType(str, Enum):
    DECISION = "decision"
    ACTION = "action"
    OUTCOME = "outcome"
    PATTERN = "pattern"


class EdgeType(str, Enum):
    DECIDED_ACTION = "decided_action"
    PRODUCED_OUTCOME = "produced_outcome"
    REVEALED_PATTERN = "revealed_pattern"
    INFORMED_DECISION = "informed_decision"


class Decision(BaseModel):
    """A decision made by any agent in the system."""

    id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_utcnow)
    agent: str
    pillar: str
    context: str
    decision_text: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict = Field(default_factory=dict)


class Action(BaseModel):
    """An action executed through the system."""

    id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_utcnow)
    decision_id: str
    agent: str
    action_type: str
    action_detail: str
    target_system: str = ""
    status: str = "pending"
    metadata: dict = Field(default_factory=dict)


class Outcome(BaseModel):
    """A measured result of an action."""

    id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_utcnow)
    action_id: str
    metric_name: str
    metric_value: float
    metric_unit: str = ""
    baseline_value: float | None = None
    delta: float | None = None
    attribution_confidence: float = 0.0
    metadata: dict = Field(default_factory=dict)


class Pattern(BaseModel):
    """A reusable pattern detected from multiple outcomes."""

    id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_utcnow)
    pattern_type: str
    description: str
    conditions: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    times_observed: int = 1
    last_confirmed: datetime = Field(default_factory=_utcnow)
    outcome_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class KGEdge(BaseModel):
    """An edge connecting two nodes in the knowledge graph."""

    id: str = Field(default_factory=_uuid)
    source_id: str
    target_id: str
    edge_type: EdgeType
    timestamp: datetime = Field(default_factory=_utcnow)
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)
