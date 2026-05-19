"""Policy definitions for the GOVERN pillar.

Policies are evaluated by the govern_gate node in the orchestrator
and by the GovernanceMiddleware on every LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Policy(BaseModel):
    """A governance policy that can be checked against actions."""

    id: str
    name: str
    description: str
    severity: str = "warning"
    enabled: bool = True
    conditions: dict = Field(default_factory=dict)


DEFAULT_POLICIES: list[Policy] = [
    Policy(
        id="data_privacy",
        name="Data Privacy",
        description="Ensure PII is not exposed in agent responses",
        severity="critical",
    ),
    Policy(
        id="approval_required",
        name="Action Approval Required",
        description="Actions affecting external systems require human approval",
        severity="critical",
    ),
    Policy(
        id="audit_trail",
        name="Audit Trail",
        description="All decisions and actions must be recorded in the Knowledge Graph",
        severity="warning",
    ),
    Policy(
        id="data_quality",
        name="Data Quality",
        description="Validate data freshness and completeness before analysis",
        severity="warning",
    ),
]
