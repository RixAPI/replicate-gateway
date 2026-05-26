"""RunwayML-flavoured auth: ``Authorization: Bearer <token>``."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.auth import enforce_api_secret


def extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )
    token = auth.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty Bearer token")
    return token


async def verify_auth(request: Request) -> None:
    token = extract_token(request)
    enforce_api_secret(token)
