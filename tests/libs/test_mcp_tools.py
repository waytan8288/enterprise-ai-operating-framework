"""Tests for MCP config loading and tool loading."""

import json
import sys
from pathlib import Path

import pytest

from mcp_tools import (
    MCPConfig,
    get_mcp_tools,
    is_mcp_tool,
    load_mcp_config,
    parse_mcp_tool_name,
)

ECHO_SERVER = Path(__file__).parent / "fixtures" / "echo_mcp_server.py"


class TestLoadConfig:
    def test_missing_file_is_empty(self, tmp_path):
        cfg = load_mcp_config(str(tmp_path / "nope.json"))
        assert cfg.servers == {}
        assert cfg.agent_tools == {}

    def test_invalid_json_is_empty(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text("{not json")
        assert load_mcp_config(str(path)).servers == {}

    def test_expands_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCP_TEST_TOKEN", "s3cret")
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({
            "servers": {"jira": {"url": "x", "headers": {"Authorization": "Bearer ${MCP_TEST_TOKEN}"}}},
            "agents": {"act_agent": {"tools": ["jira.*"]}},
        }))
        cfg = load_mcp_config(str(path))
        assert cfg.servers["jira"]["headers"]["Authorization"] == "Bearer s3cret"
        assert cfg.agent_tools == {"act_agent": ["jira.*"]}


class TestAllowlist:
    cfg = MCPConfig(
        servers={"jira": {}, "slack": {}},
        agent_tools={"act_agent": ["jira.create_issue", "slack.*"]},
    )

    def test_exact_and_glob(self):
        assert self.cfg.is_allowed("act_agent", "jira", "create_issue")
        assert self.cfg.is_allowed("act_agent", "slack", "post_message")
        assert not self.cfg.is_allowed("act_agent", "jira", "delete_project")

    def test_unknown_agent_fails_closed(self):
        assert not self.cfg.is_allowed("insight_agent", "jira", "create_issue")
        assert self.cfg.servers_for_agent("insight_agent") == {}

    def test_servers_for_agent(self):
        assert set(self.cfg.servers_for_agent("act_agent")) == {"jira", "slack"}


def test_tool_name_helpers():
    assert is_mcp_tool("mcp__jira__create_issue")
    assert not is_mcp_tool("execute_action")
    assert parse_mcp_tool_name("mcp__jira__create_issue") == ("jira", "create_issue")


async def test_no_servers_returns_empty():
    assert await get_mcp_tools("act_agent", MCPConfig()) == []


async def test_loads_allowlisted_tools_from_stdio_server():
    pytest.importorskip("langchain_mcp_adapters")
    cfg = MCPConfig(
        servers={"echo": {"transport": "stdio", "command": sys.executable, "args": [str(ECHO_SERVER)]}},
        agent_tools={"act_agent": ["echo.echo"]},
    )
    tools = await get_mcp_tools("act_agent", cfg)
    assert [t.name for t in tools] == ["mcp__echo__echo"]

    result = await tools[0].ainvoke({"text": "hi"})
    assert "echo: hi" in str(result)


async def test_unreachable_server_is_skipped():
    pytest.importorskip("langchain_mcp_adapters")
    cfg = MCPConfig(
        servers={"broken": {"transport": "stdio", "command": "/nonexistent/binary", "args": []}},
        agent_tools={"act_agent": ["broken.*"]},
    )
    assert await get_mcp_tools("act_agent", cfg) == []
