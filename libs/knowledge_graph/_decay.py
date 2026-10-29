"""Confidence decay for knowledge graph patterns.

Shared by all KG store backends so decay behaviour is consistent
regardless of the underlying database.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from knowledge_graph.models import Pattern

_DECAY_HALF_LIFE_DAYS = 30.0


def apply_confidence_decay(pattern: Pattern) -> Pattern:
    """Apply time-based exponential decay to pattern confidence.

    Uses a 30-day half-life so patterns not confirmed in ~90 days
    drop to ~12.5 % of their stored confidence.
    """
    now = datetime.now(timezone.utc)
    confirmed = pattern.last_confirmed
    if confirmed.tzinfo is None:
        confirmed = confirmed.replace(tzinfo=timezone.utc)
    age_days = (now - confirmed).total_seconds() / 86400.0
    if age_days <= 0:
        return pattern
    decay_factor = math.exp(-0.693 * age_days / _DECAY_HALF_LIFE_DAYS)
    effective = pattern.confidence * decay_factor
    return pattern.model_copy(update={"confidence": round(effective, 4)})
