"""Authentication handler for LangGraph agents.

Supports two modes:
- Local dev: LANGGRAPH_LOCAL_DEV=1, uses LOCAL_DEV_FEATURE_FLAGS
- Production: JWT validation with feature flags from token claims
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser

logger = logging.getLogger(__name__)

auth = Auth()


def _get_local_dev_user_dict() -> dict[str, Any]:
    flags_raw = os.getenv("LOCAL_DEV_FEATURE_FLAGS", "")
    flags = [f.strip() for f in flags_raw.split(",") if f.strip()]
    return {
        "identity": "local-dev-user",
        "display_name": "Local Developer",
        "is_authenticated": True,
        "feature_flags": flags,
    }


@auth.authenticate
async def authenticate(headers: dict[str, Any]) -> dict[str, Any]:
    """Authenticate incoming requests.

    In local dev mode (LANGGRAPH_LOCAL_DEV=1), returns a dev user with
    feature flags from LOCAL_DEV_FEATURE_FLAGS env var.

    In production, validates JWT from Authorization header.
    """
    if os.getenv("LANGGRAPH_LOCAL_DEV") == "1":
        return _get_local_dev_user_dict()

    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing authorization header"
        )

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        import jwt

        secret = os.environ["AUTH_SECRET_KEY"]
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        flags = payload.get("feature_flags", [])
        if not isinstance(flags, list):
            flags = []
        return {
            "identity": payload.get("sub", "unknown"),
            "display_name": payload.get("name", ""),
            "is_authenticated": True,
            "feature_flags": flags,
        }
    except Exception as e:
        logger.warning("JWT validation failed: %s", e)
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid token"
        )


@auth.on
async def set_request_configurable(
    ctx: Auth.types.AuthContext,
    value: dict[str, Any],
) -> None:
    """Inject auth context into RunnableConfig.configurable."""
    configurable = value.setdefault("configurable", {})

    if isinstance(ctx.user, StudioUser):
        dev_user = _get_local_dev_user_dict()
        configurable["feature_flags"] = dev_user["feature_flags"]
        configurable["user_display_name"] = dev_user["display_name"]
    elif isinstance(ctx.user, dict):
        configurable["langgraph_auth_user"] = ctx.user
        configurable["user_display_name"] = ctx.user.get("display_name", "")
        configurable["feature_flags"] = ctx.user.get("feature_flags", [])
