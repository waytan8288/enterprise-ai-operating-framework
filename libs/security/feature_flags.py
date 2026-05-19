"""Feature flag constants and authorization helpers.

All enforcement points MUST import constants and ``has_feature`` from this
module — never reimplement string comparisons locally.

This module is intentionally a leaf: no imports from auth.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

INSIGHT_AGENT_FLAG = "enable_insight_agent"
STRATEGY_AGENT_FLAG = "enable_strategy_agent"
ACT_AGENT_FLAG = "enable_act_agent"
MEASURE_AGENT_FLAG = "enable_measure_agent"
LEARN_AGENT_FLAG = "enable_learn_agent"
GOVERN_AGENT_FLAG = "enable_govern_agent"

ALL_AGENT_FLAGS = [
    INSIGHT_AGENT_FLAG,
    STRATEGY_AGENT_FLAG,
    ACT_AGENT_FLAG,
    MEASURE_AGENT_FLAG,
    LEARN_AGENT_FLAG,
    GOVERN_AGENT_FLAG,
]

AGENT_FLAG_MAP: dict[str, str] = {
    "insight_agent": INSIGHT_AGENT_FLAG,
    "strategy_agent": STRATEGY_AGENT_FLAG,
    "act_agent": ACT_AGENT_FLAG,
    "measure_agent": MEASURE_AGENT_FLAG,
    "learn_agent": LEARN_AGENT_FLAG,
    "govern_agent": GOVERN_AGENT_FLAG,
}


def has_feature(source: Mapping[str, Any] | None, flag: str) -> bool:
    """Return whether ``flag`` is present in ``source['feature_flags']``.

    Missing or non-list values fail closed (return False).
    """
    if not source:
        return False
    flags = source.get("feature_flags")
    if not isinstance(flags, list):
        return False
    return any(isinstance(f, str) and f == flag for f in flags)


def _valid_flag_list(raw: Any) -> list[str] | None:
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    return None


def resolve_feature_flags(config: Any) -> list[str]:
    """Return the effective feature_flags list for a LangGraph request.

    Priority:
    1. config.configurable.langgraph_auth_user.feature_flags (JWT path)
    2. config.configurable.feature_flags (local dev path)
    3. Empty list (fail closed)
    """
    configurable: Mapping[str, Any]
    if isinstance(config, Mapping):
        raw_configurable = config.get("configurable")
        configurable = (
            raw_configurable if isinstance(raw_configurable, Mapping) else {}
        )
    else:
        configurable = {}

    auth_user = configurable.get("langgraph_auth_user")
    if isinstance(auth_user, Mapping):
        return _valid_flag_list(auth_user.get("feature_flags")) or []

    return _valid_flag_list(configurable.get("feature_flags")) or []
