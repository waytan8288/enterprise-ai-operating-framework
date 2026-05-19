"""Dynamic system prompt builder for the orchestrator agent."""

from __future__ import annotations

from security.feature_flags import (
    AGENT_FLAG_MAP,
    has_feature,
)

_BASE_PROMPT = """You are the Enterprise AI Operating Framework Orchestrator — the COORDINATE pillar.

Your role is to:
1. Understand user intent
2. Route requests to the appropriate specialist agent
3. Validate response quality
4. Synthesize multi-agent responses into coherent answers

# Available Agents
{agent_section}

# Routing Guidelines
- Data analysis, trends, anomalies, root causes → insight_agent
- Recommendations, what to do next, prioritization → strategy_agent
- Execute actions, run workflows, make API calls → act_agent
- Measure outcomes, track KPIs, generate reports → measure_agent
- Capture learnings, detect patterns, update knowledge → learn_agent
- Audit trails, compliance, data quality, policy checks → govern_agent

# Instructions
1. Analyze the user's request to determine which agent should handle it.
2. Use the appropriate transfer tool to hand off to that agent.
3. If the request spans multiple pillars, start with the most relevant agent.
4. If unclear, ask the user for clarification before routing.
"""

_AGENT_DESCRIPTIONS: dict[str, str] = {
    "insight_agent": "INSIGHT — Analyze enterprise data, detect patterns and anomalies, explain root causes, surface risks and opportunities",
    "strategy_agent": "STRATEGY — Convert insights into recommendations, match conditions to proven patterns, prioritize by value/risk/cost",
    "act_agent": "ACT — Execute through approved workflows, connect to tools and APIs, respect approval gates",
    "measure_agent": "MEASURE — Quantify business outcomes, attribute impact to actions, produce portfolio reporting",
    "learn_agent": "LEARN — Capture outcomes, enrich the Knowledge Graph, detect reusable patterns",
    "govern_agent": "GOVERN — Audit trails, compliance checks, data quality validation, policy enforcement",
}


def build_orchestrator_prompt(
    feature_flags: list[str] | None = None,
    user_display_name: str | None = None,
    user_timezone: str | None = None,
) -> str:
    flags = feature_flags or []
    flag_source = {"feature_flags": flags}

    available = []
    for agent_name, description in _AGENT_DESCRIPTIONS.items():
        flag = AGENT_FLAG_MAP.get(agent_name)
        if flag is None or has_feature(flag_source, flag):
            available.append(f"- **{agent_name}**: {description}")

    agent_section = "\n".join(available) if available else "- No agents currently available."

    prompt = _BASE_PROMPT.format(agent_section=agent_section)

    context_parts = []
    if user_display_name:
        context_parts.append(f"- User: {user_display_name}")
    if user_timezone:
        context_parts.append(f"- Timezone: {user_timezone}")
    if context_parts:
        prompt += "\n# User Context\n" + "\n".join(context_parts)

    return prompt
