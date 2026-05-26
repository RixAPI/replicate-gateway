"""Unified settings.

A single Settings model holds all knobs used by any vendor module. Each
vendor only reads the keys it needs; unused keys simply sit at their
defaults.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Optional: restrict access to a single known key. When set, every
    # request must present a Bearer token matching this exact value.
    api_secret: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Async vendors (runwayml-style): persistent task store
    db_path: str = "tasks.db"

    # Sync vendors (gemini-style): poll-until-done tuning
    predict_timeout: float = 300.0
    poll_interval: float = 2.0

    # Replicate client concurrency
    max_concurrent: int = 2000
    api_pool_size: int = 500
    cdn_pool_size: int = 200

    # Cloudflare R2 (or S3-compatible) storage for BFL base64 image uploads.
    # When not configured, BFL will reject base64 input_image with a clear error.
    storage_endpoint: str = ""
    storage_access_key_id: str = ""
    storage_access_key_secret: str = ""
    storage_bucket: str = ""
    storage_region: str = "auto"
    storage_custom_domain: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
