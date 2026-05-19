"""Test configuration — set env vars before any imports."""

import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("CONNECTOR_TYPE", "sqlite")
os.environ.setdefault("SQLITE_DB_PATH", ":memory:")
os.environ.setdefault("KG_SQLITE_PATH", ":memory:")
os.environ.setdefault("LANGGRAPH_LOCAL_DEV", "1")
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret")
os.environ.setdefault(
    "LOCAL_DEV_FEATURE_FLAGS",
    "enable_insight_agent,enable_strategy_agent,enable_act_agent,"
    "enable_measure_agent,enable_learn_agent,enable_govern_agent",
)

# Add source paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for subdir in ["agents", "libs"]:
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
