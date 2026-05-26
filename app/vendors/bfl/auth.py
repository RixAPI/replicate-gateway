"""BFL auth: ``x-key: <token>`` header (BFL convention) or ``Authorization: Bearer``.

The token is your Replicate API token; it's forwarded as-is to Replicate.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.auth import enforce_api_secret


def extract_token(request: Request) -> str:
    """Pull the caller's Replicate token from ``x-key`` or ``Authorization: Bearer``."""
    token = request.headers.get("x-key", "").strip()
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ").strip()
    return ""


async def verify_auth(request: Request) -> None:
    token = extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token. Provide x-key or Authorization: Bearer.",
        )
    enforce_api_secret(token)
