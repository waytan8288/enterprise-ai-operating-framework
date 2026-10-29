"""Load MCP server tools as LangChain tools, filtered by per-agent allowlist.

Tools are renamed ``mcp__<server>__<tool>`` so agents can recognise them
(e.g. to route them through approval gates) and so names never collide
with native tools. Requires the optional ``langchain-mcp-adapters`` package.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool

from mcp_tools.config import MCPConfig, load_mcp_config

logger = logging.getLogger(__name__)

MCP_TOOL_PREFIX = "mcp__"

_cache: dict[str, list[BaseTool]] = {}


def is_mcp_tool(tool_name: str) -> bool:
    return tool_name.startswith(MCP_TOOL_PREFIX)


def parse_mcp_tool_name(tool_name: str) -> tuple[str, str]:
    """Split ``mcp__<server>__<tool>`` into ``(server, tool)``."""
    server, _, tool = tool_name.removeprefix(MCP_TOOL_PREFIX).partition("__")
    return server, tool


def _prepare(agent_name: str, config: MCPConfig | None) -> MCPConfig | None:
    """Blocking setup (file read, SDK import); run off the event loop.

    Returns None when servers are configured but the adapter isn't installed.
    """
    cfg = config or load_mcp_config()
    if cfg.servers_for_agent(agent_name):
        try:
            import langchain_mcp_adapters.client  # noqa: F401
        except ImportError:
            logger.warning(
                "MCP servers configured but langchain-mcp-adapters is not "
                "installed; run `uv sync --group mcp`"
            )
            return None
    return cfg


async def _load_server_tools(
    config: MCPConfig, agent_name: str, server: str
) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({server: config.servers[server]})
    tools = await client.get_tools(server_name=server)

    allowed: list[BaseTool] = []
    for tool in tools:
        if not config.is_allowed(agent_name, server, tool.name):
            continue
        # Renaming is safe: the adapter's coroutine closes over the original name.
        tool.name = f"{MCP_TOOL_PREFIX}{server}__{tool.name}"
        allowed.append(tool)
    return allowed


async def get_mcp_tools(
    agent_name: str, config: MCPConfig | None = None
) -> list[BaseTool]:
    """Return allowlisted MCP tools for ``agent_name`` (cached per agent).

    Servers that fail to connect are logged and skipped so one bad server
    never takes an agent down; results are only cached if all servers load.
    """
    if config is None and agent_name in _cache:
        return _cache[agent_name]

    cfg = await asyncio.to_thread(_prepare, agent_name, config)
    if cfg is None:
        return []

    tools: list[BaseTool] = []
    failed = False
    for server in cfg.servers_for_agent(agent_name):
        try:
            tools.extend(await _load_server_tools(cfg, agent_name, server))
        except Exception:
            failed = True
            logger.exception("Failed to load tools from MCP server %r", server)

    logger.info("Loaded %d MCP tools for %s", len(tools), agent_name)
    # Don't cache partial results, so a transient failure is retried next turn.
    if config is None and not failed:
        _cache[agent_name] = tools
    return tools


def clear_mcp_tool_cache() -> None:
    _cache.clear()
