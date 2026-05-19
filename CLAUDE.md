# Enterprise AI Operating Framework — Development Guide

## Project Overview
Enterprise AI Operating Framework, also known as ATRIUM (Agentic Transformation, Recursive Intelligence, Unified Model), is a multi-agent enterprise AI framework with 7 agents operating as a flywheel around an Enterprise Knowledge Graph.

## Architecture
- **Agents** live in `agents/` — each has graph.py, nodes.py, tools.py, state.py, prompts.py
- **Shared libs** live in `libs/` — knowledge_graph, connectors, security, governance
- **Orchestrator** is the single entry point; routes to other agents via supervisor pattern
- **Knowledge Graph** captures all decisions, actions, outcomes, patterns (SQLite default)

## Package Layout
Non-standard: `pyproject.toml` maps `agents/X` → importable as `X` and `libs/X` → importable as `X`.

## Commands
- `make dev` — install with dev dependencies
- `make test` — run pytest
- `make lint` — ruff + mypy
- `make serve-dev` — start langgraph dev server with all agents exposed

## Key Patterns
- State: TypedDict with `Required[Annotated[list[AnyMessage], add_messages]]`
- Handoffs: `Command(graph=Command.PARENT, goto=<target>)` pattern
- Feature flags: always use `has_feature()` from `security.feature_flags`
- Connectors: always use `create_connector()` factory, never import backends directly

## Environment
- Copy `.env.example` to `.env` and fill in API keys
- `CONNECTOR_TYPE` controls which database backend is used
- `LANGGRAPH_LOCAL_DEV=1` enables dev mode with all feature flags
