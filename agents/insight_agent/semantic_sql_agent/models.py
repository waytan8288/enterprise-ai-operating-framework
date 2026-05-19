"""Semantic model definitions for NL-to-SQL generation.

These dictionaries define the data schema, join patterns, business rules,
and SQL examples that are embedded in the semantic SQL agent's system prompt.

Populate these per-deployment based on your data source schema. The skeleton
below provides the structure — replace with your actual table/column definitions.
"""

from __future__ import annotations

MODELS: dict[str, dict] = {
    "sales_summary": {
        "description": "Aggregated sales metrics by date and channel",
        "join": "sales_summary",
        "dimensions": {
            "date": {"description": "Transaction date", "expr": "s.DATE"},
            "channel": {"description": "Sales channel", "expr": "s.CHANNEL"},
            "region": {"description": "Geographic region", "expr": "s.REGION"},
            "product_category": {"description": "Product category", "expr": "s.PRODUCT_CATEGORY"},
        },
        "measures": {
            "total_revenue": {"description": "Total revenue", "expr": "SUM(s.REVENUE)"},
            "transaction_count": {"description": "Number of transactions", "expr": "COUNT(*)"},
            "avg_order_value": {"description": "Average order value", "expr": "AVG(s.ORDER_VALUE)"},
            "conversion_rate": {"description": "Conversion rate", "expr": "SUM(s.CONVERSIONS) * 1.0 / NULLIF(SUM(s.SESSIONS), 0)"},
        },
    },
    "customer_metrics": {
        "description": "Customer-level engagement and lifetime metrics",
        "join": "customer_metrics",
        "dimensions": {
            "customer_segment": {"description": "Customer segment", "expr": "c.SEGMENT"},
            "acquisition_channel": {"description": "How customer was acquired", "expr": "c.ACQUISITION_CHANNEL"},
            "tenure_months": {"description": "Months since first purchase", "expr": "c.TENURE_MONTHS"},
        },
        "measures": {
            "customer_count": {"description": "Number of customers", "expr": "COUNT(DISTINCT c.CUSTOMER_ID)"},
            "avg_lifetime_value": {"description": "Average customer lifetime value", "expr": "AVG(c.LIFETIME_VALUE)"},
            "churn_rate": {"description": "Customer churn rate", "expr": "SUM(CASE WHEN c.IS_CHURNED THEN 1 ELSE 0 END) * 1.0 / COUNT(*)"},
        },
    },
}

JOINS: dict[str, dict] = {
    "sales_summary": {
        "description": "Core sales fact table",
        "sql": "\nFROM sales_summary s",
    },
    "customer_metrics": {
        "description": "Customer dimension table",
        "sql": "\nFROM customer_metrics c",
    },
    "sales_customer": {
        "description": "Sales joined with customer data",
        "sql": "\nFROM sales_summary s\nJOIN customer_metrics c ON s.CUSTOMER_ID = c.CUSTOMER_ID",
    },
}

BUSINESS_RULES: list[str] = [
    "Always use table aliases as defined in the join patterns.",
    "Date filters should use the DATE column, not TIMESTAMP.",
    "Revenue metrics should be rounded to 2 decimal places.",
    "Percentages should be expressed as decimals (0.05 not 5%).",
    "Limit results to 1000 rows unless the user specifies otherwise.",
]

SQL_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What was total revenue last month?",
        "sql": "SELECT SUM(s.REVENUE) as total_revenue FROM sales_summary s WHERE s.DATE >= DATE('now', '-1 month')",
    },
    {
        "question": "Which channels have the highest conversion rate?",
        "sql": "SELECT s.CHANNEL, SUM(s.CONVERSIONS) * 1.0 / NULLIF(SUM(s.SESSIONS), 0) as conversion_rate FROM sales_summary s GROUP BY s.CHANNEL ORDER BY conversion_rate DESC",
    },
    {
        "question": "How many customers churned this quarter?",
        "sql": "SELECT COUNT(*) as churned_customers FROM customer_metrics c WHERE c.IS_CHURNED = 1 AND c.CHURN_DATE >= DATE('now', '-3 months')",
    },
]
