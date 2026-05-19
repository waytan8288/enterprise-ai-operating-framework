"""GovernanceMiddleware — cross-cutting governance applied to all agents.

Wraps LLM invocations with PII redaction on outputs and audit logging.
Use `governed_model()` to wrap any ChatOpenAI instance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


def redact_pii(text: str) -> str:
    """Redact common PII patterns from text."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text


def redact_message(msg: BaseMessage) -> BaseMessage:
    """Return a copy of the message with PII redacted from content."""
    if not isinstance(msg, AIMessage) or not isinstance(msg.content, str):
        return msg
    redacted = redact_pii(msg.content)
    if redacted == msg.content:
        return msg
    logger.info("PII redacted from AI response")
    return AIMessage(
        content=redacted,
        tool_calls=msg.tool_calls,
        response_metadata=msg.response_metadata,
        id=msg.id,
    )


class GovernedModel:
    """Thin wrapper around a chat model that applies governance on invoke.

    Applies PII redaction to outputs and logs all calls for audit.
    Delegates bind_tools and all other attributes to the wrapped model.
    """

    def __init__(self, model: Any, agent_name: str = "unknown") -> None:
        self._model = model
        self._agent_name = agent_name

    def bind_tools(self, tools: list, **kwargs: Any) -> GovernedModel:
        bound = self._model.bind_tools(tools, **kwargs)
        return GovernedModel(bound, self._agent_name)

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        logger.debug("GovernedModel call: agent=%s, messages=%d", self._agent_name, len(messages))
        response = await self._model.ainvoke(messages, **kwargs)
        return redact_message(response)

    def invoke(self, messages: list, **kwargs: Any) -> AIMessage:
        logger.debug("GovernedModel call: agent=%s, messages=%d", self._agent_name, len(messages))
        response = self._model.invoke(messages, **kwargs)
        return redact_message(response)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def governed_model(model: Any, agent_name: str = "unknown") -> GovernedModel:
    """Wrap a chat model with governance (PII redaction + audit logging)."""
    return GovernedModel(model, agent_name)
