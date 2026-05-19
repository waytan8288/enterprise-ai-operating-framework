"""Tests for orchestrator routing logic and govern_gate."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orchestrator_agent.graph import (
    _clean_messages_for_subgraph,
    govern_gate,
    route_after_agent,
)


class TestCleanMessagesForSubgraph:
    def test_keeps_human_messages(self):
        msgs = [HumanMessage(content="hello")]
        result = _clean_messages_for_subgraph(msgs)
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_keeps_clean_ai_messages(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="response"),
        ]
        result = _clean_messages_for_subgraph(msgs)
        assert len(result) == 2

    def test_strips_ai_messages_with_tool_calls(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "transfer_to_insight_agent", "args": {"query": "test"}}],
            ),
        ]
        result = _clean_messages_for_subgraph(msgs)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_strips_tool_messages(self):
        msgs = [
            HumanMessage(content="hi"),
            ToolMessage(content="result", tool_call_id="1"),
        ]
        result = _clean_messages_for_subgraph(msgs)
        assert len(result) == 1

    def test_fallback_to_human_messages_if_empty(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "fn", "args": {}}],
            ),
            ToolMessage(content="ok", tool_call_id="1"),
        ]
        result = _clean_messages_for_subgraph(msgs)
        assert result == []


class TestRouteAfterAgent:
    def test_ends_on_clean_ai_message(self):
        state = {
            "messages": [AIMessage(content="Here is my answer.")],
            "active_agent": "insight_agent",
        }
        assert route_after_agent(state) == "__end__"

    def test_routes_to_active_agent_on_tool_calls(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "1", "name": "search", "args": {}}],
                )
            ],
            "active_agent": "strategy_agent",
            "feature_flags": [
                "enable_strategy_agent",
            ],
        }
        assert route_after_agent(state) == "strategy_agent"

    def test_defaults_to_insight_when_no_active(self):
        state = {"messages": []}
        assert route_after_agent(state) == "insight_agent"

    def test_downgrades_revoked_agent(self):
        state = {
            "messages": [],
            "active_agent": "act_agent",
            "feature_flags": [],
        }
        result = route_after_agent(state)
        assert result == "insight_agent"


class TestGovernGate:
    @pytest.mark.asyncio
    async def test_redacts_pii_from_input(self):
        state = {
            "messages": [HumanMessage(content="Contact john@example.com about this")],
        }
        config = {"configurable": {}}
        result = await govern_gate(state, config)
        assert "active_policies" in result.update
        msgs = result.update.get("messages")
        if msgs:
            assert "[REDACTED_EMAIL]" in msgs[0].content
            assert "john@example.com" not in msgs[0].content

    @pytest.mark.asyncio
    async def test_passes_clean_messages_through(self):
        state = {
            "messages": [HumanMessage(content="What are our sales trends?")],
        }
        config = {"configurable": {}}
        result = await govern_gate(state, config)
        assert "active_policies" in result.update
        assert "messages" not in result.update

    @pytest.mark.asyncio
    async def test_routes_to_auth_router(self):
        state = {"messages": [HumanMessage(content="hello")]}
        config = {"configurable": {}}
        result = await govern_gate(state, config)
        assert result.goto == "auth_router"

    @pytest.mark.asyncio
    async def test_injects_active_policies(self):
        state = {"messages": [HumanMessage(content="test")]}
        config = {"configurable": {}}
        result = await govern_gate(state, config)
        policies = result.update["active_policies"]
        assert "data_privacy" in policies
        assert "approval_required" in policies
