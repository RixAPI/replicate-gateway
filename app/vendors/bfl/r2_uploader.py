"""Cloudflare R2 / S3-compatible image uploader.

BFL accepts base64 input images, but Replicate prefers URLs (large data
URIs occasionally trip 400/413 errors on the upstream). We sniff the
content type, upload the bytes to R2, and hand back a public URL the
upstream can fetch.

`boto3` is synchronous; we wrap the put_object call with
:func:`asyncio.to_thread` so it doesn't block the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:[^;,]*(?:;[^,]*)?,", re.IGNORECASE)
_VALID_B64 = re.compile(r"[^A-Za-z0-9+/=]")


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def clean_base64(data: str) -> str:
    """Strip data URI prefix, whitespace, illegal chars; pad to mod-4."""
    if not data:
        return ""
    data = data.strip()
    data = _DATA_URI_RE.sub("", data)
    data = _VALID_B64.sub("", data)
    pad = (-len(data)) % 4
    if pad:
        data = data + ("=" * pad)
    return data


def sniff_mime(blob: bytes) -> str:
    """Best-effort image format sniffing — matches Go's http.DetectContentType
    for the formats BFL realistically receives."""
    if blob.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a"):
        return "image/gif"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _ext_for(mime: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(mime, ".jpg")


class R2NotConfiguredError(RuntimeError):
    pass


class R2Uploader:
    """Lazy boto3 S3 client bound to R2 (or any S3-compatible endpoint)."""

    def __init__(self):
        self._client = None
        self._configured: Optional[bool] = None

    @property
    def is_configured(self) -> bool:
        if self._configured is None:
            s = get_settings()
            self._configured = bool(
                s.storage_endpoint
                and s.storage_access_key_id
                and s.storage_access_key_secret
                and s.storage_bucket
            )
        return self._configured

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise R2NotConfiguredError(
                "boto3 is required for R2 uploads; install boto3."
            ) from exc

        s = get_settings()
        endpoint = s.storage_endpoint
        if not endpoint.startswith("http"):
            endpoint = "https://" + endpoint

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=s.storage_access_key_id,
            aws_secret_access_key=s.storage_access_key_secret,
            region_name=s.storage_region or "auto",
            config=BotoConfig(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
            ),
        )
        return self._client

    def _public_url(self, filename: str) -> str:
        s = get_settings()
        if s.storage_custom_domain:
            domain = s.storage_custom_domain
            if not domain.startswith(("http://", "https://")):
                domain = "https://" + domain
            return f"{domain.rstrip('/')}/{filename}"
        if "r2.cloudflarestorage.com" in s.storage_endpoint:
            # R2 public dev domain: https://{account-id}.r2.dev/{key}
            tail = s.storage_endpoint.split("//", 1)[-1]
            account_id = tail.split(".", 1)[0]
            return f"https://{account_id}.r2.dev/{filename}"
        return f"{s.storage_endpoint.rstrip('/')}/{s.storage_bucket}/{filename}"

    async def upload_bytes(self, blob: bytes, mime: Optional[str] = None) -> str:
        if not self.is_configured:
            raise R2NotConfiguredError(
                "R2 storage is not configured. Set STORAGE_ENDPOINT, "
                "STORAGE_ACCESS_KEY_ID, STORAGE_ACCESS_KEY_SECRET, "
                "and STORAGE_BUCKET to enable base64 input_image uploads."
            )
        mime = mime or sniff_mime(blob)
        filename = f"input_image_{time.time_ns()}_{uuid.uuid4().hex[:6]}{_ext_for(mime)}"
        client = self._client_lazy()
        bucket = get_settings().storage_bucket

        def _put():
            client.put_object(
                Bucket=bucket, Key=filename, Body=blob, ContentType=mime,
            )

        await asyncio.to_thread(_put)
        url = self._public_url(filename)
        logger.info("Uploaded base64 image to R2: %s (%d bytes, %s)", url, len(blob), mime)
        return url

    async def upload_base64(self, b64_data: str) -> str:
        cleaned = clean_base64(b64_data)
        if not cleaned:
            raise ValueError("invalid base64 data")
        try:
            blob = base64.b64decode(cleaned)
        except Exception as exc:
            raise ValueError(f"failed to decode base64: {exc}") from exc
        return await self.upload_bytes(blob)


_global_uploader: Optional[R2Uploader] = None


def get_uploader() -> R2Uploader:
    """Process-wide uploader singleton."""
    global _global_uploader
    if _global_uploader is None:
        _global_uploader = R2Uploader()
    return _global_uploader
