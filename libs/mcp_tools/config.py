"""MCP server configuration — loads servers and per-agent allowlists from JSON.

Config file (path from MCP_SERVERS_CONFIG, default ./mcp_servers.json)::

    {
      "servers": {
        "jira": {
          "transport": "streamable_http",
          "url": "https://mcp.example.com/jira",
          "headers": {"Authorization": "Bearer ${JIRA_MCP_TOKEN}"}
        }
      },
      "agents": {
        "act_agent": {"tools": ["jira.create_issue", "slack.*"]}
      }
    }

Server entries are passed to ``MultiServerMCPClient`` as-is. ``${VAR}``
placeholders are expanded from the environment so secrets stay in ``.env``.
Agents without an entry get no MCP tools (fail closed).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass(frozen=True)
class MCPConfig:
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_tools: dict[str, list[str]] = field(default_factory=dict)

    def servers_for_agent(self, agent_name: str) -> dict[str, dict[str, Any]]:
        """Return only the servers referenced by the agent's allowlist."""
        wanted = {p.split(".", 1)[0] for p in self.agent_tools.get(agent_name, [])}
        return {name: cfg for name, cfg in self.servers.items() if name in wanted}

    def is_allowed(self, agent_name: str, server: str, tool: str) -> bool:
        """Check ``server.tool`` against the agent's allowlist (glob patterns)."""
        qualified = f"{server}.{tool}"
        return any(
            fnmatch.fnmatchcase(qualified, pattern)
            for pattern in self.agent_tools.get(agent_name, [])
        )


def load_mcp_config(path: str | None = None) -> MCPConfig:
    """Load MCP config from disk. Missing or invalid file yields an empty config."""
    config_path = Path(path or os.getenv("MCP_SERVERS_CONFIG", "./mcp_servers.json"))
    if not config_path.is_file():
        logger.debug("No MCP config at %s", config_path)
        return MCPConfig()

    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read MCP config %s", config_path)
        return MCPConfig()

    servers = raw.get("servers") or {}
    agents = raw.get("agents") or {}
    if not isinstance(servers, dict) or not isinstance(agents, dict):
        logger.error("MCP config %s: 'servers' and 'agents' must be objects", config_path)
        return MCPConfig()

    agent_tools = {
        name: [p for p in (entry or {}).get("tools", []) if isinstance(p, str)]
        for name, entry in agents.items()
        if isinstance(entry, dict)
    }
    return MCPConfig(servers=_expand_env(servers), agent_tools=agent_tools)
