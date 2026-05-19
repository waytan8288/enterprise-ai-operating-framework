"""Tests for GovernanceMiddleware — PII redaction and model wrapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from governance.middleware import GovernedModel, redact_message, redact_pii


class TestRedactPii:
    def test_redacts_email(self):
        assert "[REDACTED_EMAIL]" in redact_pii("user@example.com")

    def test_redacts_ssn(self):
        assert "[REDACTED_SSN]" in redact_pii("SSN: 123-45-6789")

    def test_redacts_phone(self):
        assert "[REDACTED_PHONE]" in redact_pii("Call 555-123-4567")

    def test_redacts_credit_card(self):
        assert "[REDACTED_CREDIT_CARD]" in redact_pii("Card: 4111-1111-1111-1111")

    def test_preserves_clean_text(self):
        text = "Revenue increased by 15% last quarter"
        assert redact_pii(text) == text

    def test_redacts_multiple_patterns(self):
        text = "Email: a@b.com, SSN: 111-22-3333"
        result = redact_pii(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result


class TestRedactMessage:
    def test_redacts_ai_message_content(self):
        msg = AIMessage(content="Contact user@test.com", id="1")
        result = redact_message(msg)
        assert "[REDACTED_EMAIL]" in result.content

    def test_preserves_clean_ai_message(self):
        msg = AIMessage(content="All good", id="1")
        result = redact_message(msg)
        assert result is msg

    def test_ignores_non_ai_messages(self):
        msg = HumanMessage(content="user@test.com")
        result = redact_message(msg)
        assert result is msg

    def test_preserves_tool_calls(self):
        msg = AIMessage(
            content="Here is user@test.com",
            tool_calls=[{"id": "1", "name": "fn", "args": {}}],
            id="1",
        )
        result = redact_message(msg)
        assert result.tool_calls == msg.tool_calls


class TestGovernedModel:
    @pytest.mark.asyncio
    async def test_ainvoke_redacts_output(self):
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AIMessage(
            content="Their SSN is 999-88-7777", id="resp"
        )
        governed = GovernedModel(mock_model, "test_agent")
        result = await governed.ainvoke([HumanMessage(content="hi")])
        assert "[REDACTED_SSN]" in result.content

    def test_bind_tools_returns_governed(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        governed = GovernedModel(mock_model, "test")
        bound = governed.bind_tools([])
        assert isinstance(bound, GovernedModel)

    def test_delegates_unknown_attrs(self):
        mock_model = MagicMock()
        mock_model.temperature = 0.5
        governed = GovernedModel(mock_model, "test")
        assert governed.temperature == 0.5
