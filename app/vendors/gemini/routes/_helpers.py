"""Shared helpers used by Gemini route modules."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEBUG_DIR = "/tmp/replicate-debug"


def dump_debug(inp: dict, model: str, error: str, kind: str = "request") -> str | None:
    """Save full request payload (including base64 blobs) to a debug file."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{kind}_{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(DEBUG_DIR, fname)
        with open(path, "w") as f:
            json.dump(
                {"model": model, "error": error, "input": inp},
                f,
                ensure_ascii=False,
            )
        return path
    except Exception:
        logger.warning("Failed to dump debug payload", exc_info=True)
        return None


def safe_params(inp: dict) -> dict:
    """Redact bulky base64 fields for logging."""
    redacted = {}
    for k, v in inp.items():
        if k in ("image_input", "image", "last_frame", "reference_images", "video"):
            if isinstance(v, list):
                redacted[k] = f"[{len(v)} item(s)]"
            else:
                redacted[k] = "<base64 blob>"
        else:
            redacted[k] = v
    return redacted
