"""Tests for MCP tool wiring in the Act agent."""

from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from act_agent import nodes
from act_agent.graph import route_after_planning, route_after_verify
from security.feature_flags import MCP_TOOLS_FLAG


@tool
def fake_mcp(ticket: str) -> str:
    """Fake MCP tool."""
    return f"created {ticket}"


fake_mcp.name = "mcp__jira__create_issue"


def _ai(*names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": {"ticket": "T-1"}, "id": f"call_{i}"} for i, n in enumerate(names)],
    )


class TestRouting:
    def test_mcp_call_requires_approval(self):
        state = {"messages": [_ai("mcp__jira__create_issue")]}
        assert route_after_planning(state) == "verify"

    def test_approved_mcp_call_runs(self):
        state = {"messages": [_ai("mcp__jira__create_issue")], "action_approved": True}
        assert route_after_planning(state) == "tools"

    def test_native_read_tool_skips_approval(self):
        assert route_after_planning({"messages": [_ai("plan_action")]}) == "tools"

    def test_after_verify(self):
        assert route_after_verify({"action_approved": True}) == "tools"
        assert route_after_verify({"action_approved": False}) == "action_planning"


class TestAvailableTools:
    async def test_flag_off_excludes_mcp(self):
        with patch.object(nodes, "get_mcp_tools") as loader:
            tools = await nodes._available_tools({"messages": []}, {})
        loader.assert_not_called()
        assert tools == nodes._TOOLS

    async def test_flag_on_includes_mcp(self):
        fake_loader = AsyncMock(return_value=[fake_mcp])
        config = {"configurable": {"feature_flags": [MCP_TOOLS_FLAG]}}
        with patch.object(nodes, "get_mcp_tools", fake_loader):
            tools = await nodes._available_tools({"messages": []}, config)
        assert fake_mcp in tools


class TestRunTools:
    async def _run(self, msg):
        fake_loader = AsyncMock(return_value=[fake_mcp])
        state = {"messages": [HumanMessage("go"), msg], "feature_flags": [MCP_TOOLS_FLAG]}
        with (
            patch.object(nodes, "get_mcp_tools", fake_loader),
            patch.object(nodes, "_record_mcp_call") as record,
        ):
            return await nodes.run_tools_node(state, {}), record

    async def test_executes_mcp_tool_records_and_resets_approval(self):
        update, record = await self._run(_ai("mcp__jira__create_issue"))
        [result] = update["messages"]
        assert isinstance(result, ToolMessage)
        assert result.content == "created T-1"
        assert update["action_approved"] is False
        record.assert_called_once_with("mcp__jira__create_issue", {"ticket": "T-1"}, "completed")

    async def test_unknown_tool_returns_error(self):
        update, record = await self._run(_ai("mcp__jira__delete_project"))
        [result] = update["messages"]
        assert result.status == "error"
        record.assert_not_called()


def test_rejection_answers_pending_tool_calls():
    state = {"messages": [_ai("mcp__jira__create_issue")]}
    with patch.object(nodes, "interrupt", return_value=False):
        update = nodes.verify_action_node(state, {})
    assert update["action_approved"] is False
    [msg] = update["messages"]
    assert msg.tool_call_id == "call_0"
