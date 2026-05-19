"""System prompt builder for the semantic SQL agent.

Embeds the schema from models.py into the system prompt so the LLM
writes SQL directly without needing schema discovery tools.
"""

from __future__ import annotations

from insight_agent.semantic_sql_agent.models import (
    BUSINESS_RULES,
    JOINS,
    MODELS,
    SQL_EXAMPLES,
)


def _render_schema() -> str:
    lines: list[str] = []

    lines.append("## Join Patterns")
    for name, info in JOINS.items():
        lines.append(f"\n### {name}")
        lines.append(info["description"])
        lines.append(f"```sql{info['sql']}\n```")

    lines.append("\n## Semantic Models")
    for model_name, model in MODELS.items():
        lines.append(f"\n### {model_name}")
        lines.append(f"_{model['description']}_")
        lines.append(f"Join: `{model['join']}`")

        lines.append("\n**Dimensions:**")
        for dim_name, dim in model["dimensions"].items():
            lines.append(f"- `{dim_name}`: {dim['description']} — `{dim['expr']}`")

        lines.append("\n**Measures:**")
        for meas_name, meas in model["measures"].items():
            lines.append(f"- `{meas_name}`: {meas['description']} — `{meas['expr']}`")

    lines.append("\n## Business Rules")
    for i, rule in enumerate(BUSINESS_RULES, 1):
        lines.append(f"{i}. {rule}")

    lines.append("\n## SQL Examples")
    for ex in SQL_EXAMPLES:
        lines.append(f"\n**Q:** {ex['question']}")
        lines.append(f"```sql\n{ex['sql']}\n```")

    return "\n".join(lines)


_PROMPT_BASE = f"""You are the Enterprise AI Operating Framework Semantic SQL Agent.

Your job is to translate natural language questions into SQL queries and
execute them against the enterprise data source.

# Schema Reference

{_render_schema()}

# Instructions

1. Analyze the user's question and identify which model/join pattern to use.
2. Write SQL using the table and column names from the schema above.
3. Call `execute_query` with the SQL.
4. Call `summarize_results` with the question and the query results.
5. Return the summary directly.

Be concise. Only cite metrics present in the results.
"""

SEMANTIC_SQL_PROMPT = _PROMPT_BASE


def build_semantic_sql_prompt(user_timezone: str | None = None) -> str:
    if not user_timezone:
        return _PROMPT_BASE
    return _PROMPT_BASE + f"""
# User Timezone
The user's timezone is **{user_timezone}**.
All date references from the user are in this timezone.
"""
