"""Routing accuracy evaluation for the orchestrator.

Evaluates whether the LLM router correctly classifies user intent and
routes to the expected specialist agent. Can be run standalone or
integrated with LangSmith for tracked evaluation runs.

Usage:
    # Standalone (prints results to stdout)
    python -m evals.routing_eval

    # With LangSmith tracking (requires LANGCHAIN_API_KEY)
    LANGCHAIN_TRACING_V2=true python -m evals.routing_eval
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass

# Add source paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ["agents", "libs"]:
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestrator_agent.prompts import build_orchestrator_prompt


@dataclass
class RoutingTestCase:
    input: str
    expected_agent: str
    category: str = ""


ROUTING_DATASET: list[RoutingTestCase] = [
    # Insight agent — data analysis
    RoutingTestCase(
        input="What were our total sales last quarter?",
        expected_agent="insight_agent",
        category="data_query",
    ),
    RoutingTestCase(
        input="Show me the trend in customer churn over the past 6 months",
        expected_agent="insight_agent",
        category="trend_analysis",
    ),
    RoutingTestCase(
        input="Why did conversion rates drop last week?",
        expected_agent="insight_agent",
        category="root_cause",
    ),
    RoutingTestCase(
        input="Are there any anomalies in our revenue data?",
        expected_agent="insight_agent",
        category="anomaly_detection",
    ),
    # Strategy agent — recommendations
    RoutingTestCase(
        input="What should we do about declining customer retention?",
        expected_agent="strategy_agent",
        category="recommendation",
    ),
    RoutingTestCase(
        input="Prioritize our top 3 growth opportunities",
        expected_agent="strategy_agent",
        category="prioritization",
    ),
    RoutingTestCase(
        input="Based on past patterns, what's the best pricing strategy?",
        expected_agent="strategy_agent",
        category="pattern_advice",
    ),
    # Act agent — execution
    RoutingTestCase(
        input="Execute the email campaign we planned for segment A",
        expected_agent="act_agent",
        category="execute_action",
    ),
    RoutingTestCase(
        input="Run the data pipeline for the nightly ETL job",
        expected_agent="act_agent",
        category="run_workflow",
    ),
    # Measure agent — metrics and outcomes
    RoutingTestCase(
        input="What was the impact of the pricing change we made last month?",
        expected_agent="measure_agent",
        category="measure_impact",
    ),
    RoutingTestCase(
        input="Generate a KPI report for Q1",
        expected_agent="measure_agent",
        category="kpi_report",
    ),
    RoutingTestCase(
        input="Track the ROI of our latest marketing campaign",
        expected_agent="measure_agent",
        category="track_outcome",
    ),
    # Learn agent — pattern and knowledge
    RoutingTestCase(
        input="What patterns have we learned from our past pricing decisions?",
        expected_agent="learn_agent",
        category="pattern_review",
    ),
    RoutingTestCase(
        input="Update the knowledge graph with our latest campaign results",
        expected_agent="learn_agent",
        category="kg_enrichment",
    ),
    # Govern agent — audit and compliance
    RoutingTestCase(
        input="Show me the audit trail for decision dec-123",
        expected_agent="govern_agent",
        category="audit",
    ),
    RoutingTestCase(
        input="Check if our data handling complies with privacy policies",
        expected_agent="govern_agent",
        category="compliance",
    ),
    RoutingTestCase(
        input="Validate data quality in our customer table",
        expected_agent="govern_agent",
        category="data_quality",
    ),
    RoutingTestCase(
        input="What governance policies are currently active?",
        expected_agent="govern_agent",
        category="policy_query",
    ),
]


_ALL_FLAGS = [
    "enable_insight_agent",
    "enable_strategy_agent",
    "enable_act_agent",
    "enable_measure_agent",
    "enable_learn_agent",
    "enable_govern_agent",
]


def _extract_routed_agent(response: AIMessage) -> str | None:
    if not response.tool_calls:
        return None
    tool_name = response.tool_calls[0]["name"]
    return tool_name.replace("transfer_to_", "") if tool_name.startswith("transfer_to_") else None


async def evaluate_routing(
    model_name: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the routing evaluation and return results."""
    from orchestrator_agent.graph import _TRANSFER_TOOLS

    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    model = ChatOpenAI(model=model_name, temperature=0)

    tools = [tool_fn for _, tool_fn in _TRANSFER_TOOLS.values()]
    model_with_tools = model.bind_tools(tools)

    prompt = build_orchestrator_prompt(feature_flags=_ALL_FLAGS)

    results = []
    correct = 0
    total = len(ROUTING_DATASET)

    for tc in ROUTING_DATASET:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=tc.input),
        ]
        response = await model_with_tools.ainvoke(messages)
        routed = _extract_routed_agent(response)
        is_correct = routed == tc.expected_agent

        if is_correct:
            correct += 1

        result = {
            "input": tc.input,
            "expected": tc.expected_agent,
            "actual": routed,
            "correct": is_correct,
            "category": tc.category,
        }
        results.append(result)

        if verbose:
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] {tc.category}: expected={tc.expected_agent}, got={routed}")

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    if verbose:
        print(f"\n  Routing accuracy: {correct}/{total} ({accuracy:.1%})")

        by_agent: dict[str, list[bool]] = {}
        for r in results:
            by_agent.setdefault(r["expected"], []).append(r["correct"])
        print("\n  Per-agent accuracy:")
        for agent, scores in sorted(by_agent.items()):
            agent_acc = sum(scores) / len(scores)
            print(f"    {agent}: {sum(scores)}/{len(scores)} ({agent_acc:.0%})")

    return summary


async def evaluate_sql_generation(verbose: bool = True) -> dict:
    """Evaluate semantic SQL agent quality on sample questions."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    sql_test_cases = [
        {
            "question": "What are total sales by region?",
            "must_contain": ["SELECT", "SUM", "GROUP BY"],
            "must_not_contain": ["DROP", "DELETE", "UPDATE"],
        },
        {
            "question": "Show me the top 10 customers by revenue",
            "must_contain": ["SELECT", "ORDER BY", "LIMIT"],
            "must_not_contain": ["DROP", "DELETE"],
        },
        {
            "question": "What is the average order value this month?",
            "must_contain": ["SELECT", "AVG"],
            "must_not_contain": ["DROP", "DELETE"],
        },
    ]

    from insight_agent.semantic_sql_agent.prompts import build_semantic_sql_prompt

    model = ChatOpenAI(model=model_name, temperature=0)
    prompt = build_semantic_sql_prompt()

    results = []
    correct = 0

    for tc in sql_test_cases:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=tc["question"]),
        ]
        response = await model.ainvoke(messages)
        content = response.content.upper()

        has_required = all(kw.upper() in content for kw in tc["must_contain"])
        no_forbidden = all(kw.upper() not in content for kw in tc["must_not_contain"])
        passed = has_required and no_forbidden

        if passed:
            correct += 1

        result = {
            "question": tc["question"],
            "passed": passed,
            "has_required": has_required,
            "no_forbidden": no_forbidden,
        }
        results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {tc['question'][:60]}")

    accuracy = correct / len(sql_test_cases) if sql_test_cases else 0.0
    summary = {
        "total": len(sql_test_cases),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    if verbose:
        print(f"\n  SQL generation accuracy: {correct}/{len(sql_test_cases)} ({accuracy:.1%})")

    return summary


async def main():
    print("=" * 60)
    print("Enterprise AI Operating Framework — Evaluation Suite")
    print("=" * 60)

    print("\n1. Routing Accuracy Evaluation")
    print("-" * 40)
    routing_results = await evaluate_routing()

    print("\n2. SQL Generation Evaluation")
    print("-" * 40)
    sql_results = await evaluate_sql_generation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Routing:        {routing_results['accuracy']:.1%}")
    print(f"  SQL Generation: {sql_results['accuracy']:.1%}")

    output = {
        "routing": routing_results,
        "sql_generation": sql_results,
    }
    output_path = os.path.join(ROOT, "evals", "latest_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
