"""Unified Replicate HTTP client.

Two usage patterns coexist:

* **Async / fire-and-forget** (RunwayML video generation):
  ``create_prediction`` → return immediately with the prediction record.
  The caller stores the prediction id and polls later via
  ``get_prediction`` / ``cancel_prediction``.

* **Sync / wait-until-done** (Gemini image generation):
  ``run_prediction`` polls in-process until the prediction reaches a
  terminal state, then returns ``(output_url, prediction_id)``. The image
  payload is then fetched via ``download_image_b64``.

The client maintains a shared connection pool but does NOT bind a fixed
API token — every call receives the caller's token so that each end-user
authenticates with their own Replicate account.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

REPLICATE_BASE = "https://api.replicate.com/v1"
_TERMINAL_STATUSES = frozenset(("succeeded", "failed", "canceled"))


class UpstreamError(Exception):
    """Non-2xx response from Replicate, with status and parsed detail."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Replicate {status_code}: {detail}")


class ReplicateClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._api: httpx.AsyncClient | None = None
        self._cdn: httpx.AsyncClient | None = None
        self._sem = asyncio.Semaphore(settings.max_concurrent)
        self._api_pool_size = settings.api_pool_size
        self._cdn_pool_size = settings.cdn_pool_size

    @property
    def api(self) -> httpx.AsyncClient:
        if self._api is None or self._api.is_closed:
            self._api = httpx.AsyncClient(
                base_url=REPLICATE_BASE,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=self._api_pool_size,
                    max_keepalive_connections=self._api_pool_size // 2,
                ),
            )
        return self._api

    @property
    def cdn(self) -> httpx.AsyncClient:
        if self._cdn is None or self._cdn.is_closed:
            self._cdn = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=self._cdn_pool_size,
                    max_keepalive_connections=self._cdn_pool_size // 2,
                ),
            )
        return self._cdn

    async def close(self) -> None:
        for c in (self._api, self._cdn):
            if c and not c.is_closed:
                await c.aclose()

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _check(resp: httpx.Response, url: str) -> None:
        if resp.status_code >= 400:
            body = resp.text
            logger.error(
                "Replicate %s %s → %s body=%s",
                resp.status_code, url, resp.reason_phrase, body,
            )
            detail = body
            try:
                data = resp.json()
                detail = data.get("detail") or data.get("title") or body
            except Exception:
                pass
            raise UpstreamError(resp.status_code, detail)

    async def create_prediction(
        self, token: str, model: str, inp: dict[str, Any]
    ) -> dict:
        url = f"/models/{model}/predictions"
        logger.info("POST %s model=%s", url, model)
        resp = await self.api.post(
            url, json={"input": inp}, headers=self._auth(token)
        )
        self._check(resp, url)
        return resp.json()

    async def get_prediction(self, token: str, prediction_id: str) -> dict:
        url = f"/predictions/{prediction_id}"
        resp = await self.api.get(url, headers=self._auth(token))
        self._check(resp, url)
        return resp.json()

    async def cancel_prediction(self, token: str, prediction_id: str) -> dict:
        url = f"/predictions/{prediction_id}/cancel"
        resp = await self.api.post(url, headers=self._auth(token))
        self._check(resp, url)
        return resp.json()

    async def run_prediction(
        self,
        token: str,
        model: str,
        inp: dict[str, Any],
    ) -> tuple[str, str]:
        """Create → poll → return ``(output_url, prediction_id)``.

        The global semaphore caps in-flight predictions per worker to
        protect the connection pool under burst load.
        """
        settings = get_settings()

        async with self._sem:
            pred = await self.create_prediction(token, model, inp)
            pred_id = pred.get("id", "unknown")

            elapsed = 0.0
            while pred["status"] not in _TERMINAL_STATUSES:
                if elapsed >= settings.predict_timeout:
                    raise TimeoutError(
                        f"Prediction {pred['id']} did not complete "
                        f"within {settings.predict_timeout}s"
                    )
                await asyncio.sleep(settings.poll_interval)
                elapsed += settings.poll_interval
                pred = await self.get_prediction(token, pred["id"])
                logger.debug(
                    "pred %s status=%s %.1fs",
                    pred["id"], pred["status"], elapsed,
                )

        if pred["status"] != "succeeded":
            error_msg = pred.get("error") or "Prediction failed"
            logger.error(
                "Prediction %s failed (status=%s): %s | logs=%s",
                pred.get("id"), pred["status"], error_msg,
                (pred.get("logs") or "")[-500:],
            )
            raise RuntimeError(str(error_msg))

        output = pred.get("output")
        if not output:
            logger.error(
                "Prediction %s succeeded but output is empty: %s",
                pred.get("id"), pred,
            )
            raise RuntimeError("Prediction succeeded but returned no output")
        url = output[0] if isinstance(output, list) else output
        return url, pred_id

    async def download_image_b64(self, url: str) -> tuple[str, str]:
        """Download ``url`` and return ``(base64_data, mime_type)``."""
        resp = await self.cdn.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/png").split(";")[0].strip()
        return base64.b64encode(resp.content).decode("ascii"), mime
