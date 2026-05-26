"""Gemini-flavoured auth.

Accepts either ``x-goog-api-key`` (Gemini SDK default) or the standard
``Authorization: Bearer`` header. Either way, the token is forwarded to
Replicate as-is.
"""

from __future__ import annotations

from fastapi import Request

from app.core.auth import enforce_api_secret
from app.vendors.gemini.errors import GeminiAPIError


def extract_token(request: Request) -> str:
    token = request.headers.get("x-goog-api-key", "").strip()
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ").strip()
    return ""


async def verify_auth(request: Request) -> None:
    token = extract_token(request)
    if not token:
        raise GeminiAPIError(
            401,
            "Missing API key. Provide x-goog-api-key or Authorization header.",
            "UNAUTHENTICATED",
        )
    enforce_api_secret(token, GeminiAPIError)


def require_token(request: Request) -> str:
    """For use *inside* handlers — re-extracts the validated token."""
    token = extract_token(request)
    if not token:
        raise GeminiAPIError(401, "Missing API key", "UNAUTHENTICATED")
    return token
