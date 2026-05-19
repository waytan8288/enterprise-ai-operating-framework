# ATRIUM

**Agentic Transformation, Recursive Intelligence, Unified Model**

ATRIUM is an enterprise AI operating framework built on [LangGraph](https://github.com/langchain-ai/langgraph). It organizes AI capabilities into seven operational pillars running as a continuous flywheel around a compounding Enterprise Knowledge Graph.

> The companies that win with AI will not be the ones with the most pilots. They will be the ones that turn AI into a governed, learning system that improves every time it operates.

For the full framework specification, see [`ATRIUM_Framework_v1.pdf`](ATRIUM_Framework_v1.pdf).

## Architecture

```
COORDINATE → INSIGHT → STRATEGY → ACT → MEASURE → LEARN → GOVERN → repeat
```

| Pillar | Agent | Purpose |
|---|---|---|
| **COORDINATE** | `orchestrator_agent` | Routes user intent to the right agent, manages handoffs, synthesizes responses |
| **INSIGHT** | `insight_agent` | Analyzes enterprise data, detects patterns, explains root causes |
| **STRATEGY** | `strategy_agent` | Converts insights into recommendations grounded in outcomes and constraints |
| **ACT** | `act_agent` | Executes through approved systems, workflows, and APIs |
| **MEASURE** | `measure_agent` | Quantifies business outcomes against a defined value taxonomy |
| **LEARN** | `learn_agent` | Captures outcomes, feedback, and patterns into the Knowledge Graph |
| **GOVERN** | `govern_agent` | Enforces quality, security, privacy, compliance, and audit controls |

At the center is the **Enterprise Knowledge Graph** -- a shared intelligence layer that captures decisions, actions, outcomes, context, and institutional knowledge. Every pillar contributes to it. Every pillar draws from it. The longer the system operates, the more valuable it becomes.

## Project Structure

```
enterprise-ai-operating-framework/
├── agents/
│   ├── orchestrator_agent/   # COORDINATE — single entry point, supervisor routing
│   ├── insight_agent/        # INSIGHT — data analysis and pattern detection
│   ├── strategy_agent/       # STRATEGY — recommendations and decision logic
│   ├── act_agent/            # ACT — workflow execution and system integration
│   ├── measure_agent/        # MEASURE — value attribution and outcome tracking
│   ├── learn_agent/          # LEARN — feedback capture and knowledge enrichment
│   └── govern_agent/         # GOVERN — policy enforcement and compliance
├── libs/
│   ├── knowledge_graph/      # Enterprise Knowledge Graph (SQLite default)
│   ├── connectors/           # Database and system connectors
│   ├── security/             # Auth, RBAC, feature flags
│   └── governance/           # Policy and compliance rules
├── evals/                    # Agent evaluation harness
├── tests/                    # Test suite
├── langgraph.json            # Production LangGraph config
├── langgraph.dev.json        # Development LangGraph config
└── pyproject.toml            # Package and dependency config
```

Each agent follows a consistent module structure: `graph.py`, `nodes.py`, `tools.py`, `state.py`, `prompts.py`.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd enterprise-ai-operating-framework

# Copy environment config and add your API keys
cp .env.example .env

# Install with dev dependencies
make dev
```

### Running

```bash
# Start the LangGraph dev server (all agents exposed)
make serve-dev

# Start the production server
make serve
```

### Development

```bash
# Run tests
make test

# Lint (ruff + mypy)
make lint

# Format code
make format

# Run routing evaluations
make eval
```

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Description |
|---|---|
| `CONNECTOR_TYPE` | Database backend (`sqlite`, `mysql`, `snowflake`) |
| `LANGGRAPH_LOCAL_DEV` | Set to `1` to enable dev mode with all feature flags |

## Key Design Patterns

- **Supervisor routing** -- The orchestrator is the single entry point. Users never need to know which agent to call.
- **Command handoffs** -- Agents hand off via `Command(graph=Command.PARENT, goto=<target>)`.
- **Feature flags** -- Always use `has_feature()` from `security.feature_flags`.
- **Connector factory** -- Always use `create_connector()`, never import backends directly.

## Industry Applicability

ATRIUM is industry-agnostic. The framework applies wherever decisions are made across many systems, actions taken through different workflows, and outcomes measured inconsistently. Target industries include retail, healthcare, financial services, cybersecurity, manufacturing, software, energy, education, and the public sector.

## License

Proprietary. All rights reserved.
