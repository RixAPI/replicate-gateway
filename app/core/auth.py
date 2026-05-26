"""Shared authentication helper.

The Bearer/API-key value the caller sends is their **Replicate API token**
and will be forwarded as-is to Replicate. We only enforce two things here:

1. The header is present (this lives in each vendor's own ``auth.py`` —
   header conventions differ between vendors).
2. If ``API_SECRET`` is configured, the token matches it (constant-time).

To keep error responses in the vendor's native format, callers pass the
vendor-specific exception class via ``error_class``.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException

from app.core.config import get_settings


def enforce_api_secret(
    token: str,
    error_class: type[HTTPException] = HTTPException,
) -> None:
    """Raise ``error_class(401)`` if API_SECRET is set and ``token`` doesn't match."""
    settings = get_settings()
    if settings.api_secret and not hmac.compare_digest(token, settings.api_secret):
        raise error_class(status_code=401, detail="Invalid API token")
