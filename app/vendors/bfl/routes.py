"""BFL routes: ``POST /bfl/v1/{model}`` and ``GET /bfl/v1/get_result``.

Translates BFL native requests/responses into a Replicate prediction
call. Mirrors the Go gateway in :file:`replicate->bfl/main.go`
function-for-function, but uses our shared ReplicateClient instead of
talking to a third-party aggregator.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.replicate_client import UpstreamError
from app.vendors.bfl.auth import extract_token, verify_auth
from app.vendors.bfl.models import (
    BFLTaskCreated,
    BFLTaskResult,
    BFLTaskResultPayload,
)
from app.vendors.bfl.param_converter import (
    MODEL_PRICING,
    build_flux2_replicate_input,
    calculate_image_cost,
    is_flux2_model,
    normalize_legacy_input,
    replicate_model_path,
)
from app.vendors.bfl.r2_uploader import (
    R2NotConfiguredError,
    clean_base64,
    get_uploader,
    is_url,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bfl/v1", tags=["BFL"], dependencies=[Depends(verify_auth)])

# Marker suffix appended to task ids exposed to the caller. Lets us recognise
# that an id was created by *this* proxy without breaking BFL clients that
# treat the id as opaque.
_ID_SUFFIX = "-od9su3"


def _expose_id(prediction_id: str) -> str:
    """Append the proxy marker to an upstream prediction id."""
    return prediction_id + _ID_SUFFIX


def _unwrap_id(task_id: str) -> str:
    """Strip the proxy marker from a caller-provided task id, if present."""
    if task_id.endswith(_ID_SUFFIX):
        return task_id[: -len(_ID_SUFFIX)]
    return task_id

_PIXELS_PER_MP = 1024.0 * 1024.0
_MAX_FETCH_BYTES = 1024 * 1024  # 1 MB, same as the Go gateway


def _detect_dimensions(blob: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` from an image blob, or ``(0, 0)`` if we can't read it.

    Uses Pillow's lazy header read (no full decode).
    """
    try:
        from PIL import Image  # imported lazily so Pillow stays a soft dep
    except ImportError:
        logger.warning("Pillow not installed; cannot read image dimensions for pricing")
        return 0, 0
    try:
        with Image.open(io.BytesIO(blob)) as img:
            return img.size  # (w, h)
    except Exception:
        return 0, 0


async def _read_url_header(request: Request, url: str) -> bytes:
    """Fetch up to 1 MB from ``url`` so we can parse the image header."""
    client = request.app.state.replicate_client.cdn
    try:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            buf = b""
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                buf += chunk
                if len(buf) >= _MAX_FETCH_BYTES:
                    break
            return buf[:_MAX_FETCH_BYTES]
    except Exception:
        return b""


async def _process_image(request: Request, image_str: str) -> tuple[str, float, int, int]:
    """Return ``(public_url, mp, width, height)`` for one input image.

    URL inputs are returned as-is (we still try to read dims for pricing).
    Base64 inputs are uploaded to R2 and replaced with the public URL.
    On any decode/dim error we fall back to ``(_, 1.0, 0, 0)``.
    """
    if not image_str:
        return "", 0.0, 0, 0

    if is_url(image_str):
        header = await _read_url_header(request, image_str)
        w, h = _detect_dimensions(header) if header else (0, 0)
        mp = (w * h) / _PIXELS_PER_MP if (w and h) else 1.0
        return image_str, mp, w, h

    # base64 branch
    cleaned = clean_base64(image_str)
    if not cleaned:
        raise ValueError("invalid base64 data")
    try:
        blob = base64.b64decode(cleaned)
    except Exception as exc:
        raise ValueError(f"failed to decode base64: {exc}") from exc

    public_url = await get_uploader().upload_bytes(blob)
    w, h = _detect_dimensions(blob)
    mp = (w * h) / _PIXELS_PER_MP if (w and h) else 1.0
    return public_url, mp, w, h


async def _collect_input_images(
    request: Request, raw_input: dict[str, Any],
) -> tuple[list[str], float, int, int, int]:
    """Walk ``input_image``, ``input_image_2`` ... ``input_image_8`` in parallel.

    Returns ``(urls, total_input_mp, ref_w, ref_h, num_images)``.

    Single-image mode bills the actual MP; multi-image mode bills 1.0 MP/image
    — same logic as the Go gateway.
    """
    keys = ["input_image"] + [f"input_image_{i}" for i in range(2, 9)]
    tasks = [
        (k, raw_input[k]) for k in keys if isinstance(raw_input.get(k), str) and raw_input[k]
    ]
    if not tasks:
        return [], 0.0, 0, 0, 0

    results = await asyncio.gather(
        *[_process_image(request, value) for _, value in tasks]
    )

    num = len(results)
    urls = [r[0] for r in results]
    total_mp = 0.0
    for _, mp, _w, _h in results:
        total_mp += mp if num == 1 else 1.0

    first_w, first_h = results[0][2], results[0][3]
    return urls, total_mp, first_w, first_h, num


async def _process_legacy_input_image(
    request: Request, raw_input: dict[str, Any],
) -> dict[str, Any]:
    """Handle the single ``input_image`` field for non-flux-2 models."""
    img = raw_input.get("input_image")
    if not isinstance(img, str) or not img:
        return raw_input

    out = dict(raw_input)
    if is_url(img):
        return out  # already a URL — pass through

    cleaned = clean_base64(img)
    if not cleaned:
        raise ValueError("invalid base64 data in input_image")
    try:
        blob = base64.b64decode(cleaned)
    except Exception as exc:
        raise ValueError(f"failed to decode base64: {exc}") from exc

    public_url = await get_uploader().upload_bytes(blob)
    out["input_image"] = public_url
    logger.info("Replaced legacy input_image with R2 URL: %s", public_url)
    return out


# ── Routes ──────────────────────────────────────────────────────────


@router.post("/{model}")
async def create_task(model: str, request: Request):
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail=f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(raw_body, dict):
        raise HTTPException(400, detail="Request body must be a JSON object")

    raw_input: dict[str, Any] = raw_body

    replicate_input: dict[str, Any]
    total_input_mp = 0.0
    ref_w = ref_h = 0
    num_images = 0

    if is_flux2_model(model):
        try:
            input_images, total_input_mp, ref_w, ref_h, num_images = (
                await _collect_input_images(request, raw_input)
            )
        except R2NotConfiguredError as exc:
            raise HTTPException(500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        replicate_input = build_flux2_replicate_input(model, raw_input, input_images)
    else:
        try:
            raw_input = await _process_legacy_input_image(request, raw_input)
        except R2NotConfiguredError as exc:
            raise HTTPException(500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # Strip top-level BFL-only sentinels we don't forward
        replicate_input = normalize_legacy_input(raw_input)
        # input_image (singular) is the only file-bearing field for legacy
        if "input_image" in replicate_input and not isinstance(
            replicate_input["input_image"], str
        ):
            replicate_input.pop("input_image", None)

    token = extract_token(request)
    client = request.app.state.replicate_client
    replicate_model = replicate_model_path(model)

    try:
        prediction = await client.create_prediction(
            token, replicate_model, replicate_input,
        )
    except UpstreamError as exc:
        status = exc.status_code if 400 <= exc.status_code < 500 else 502
        raise HTTPException(status, detail=exc.detail) from exc
    except Exception as exc:
        logger.exception("Unexpected error calling Replicate for BFL model %s", model)
        raise HTTPException(502, detail="Upstream model provider returned an error") from exc

    resp = BFLTaskCreated(id=_expose_id(prediction["id"]))
    if model in MODEL_PRICING:
        cost, in_mp, out_mp = calculate_image_cost(
            model, raw_input, total_input_mp, ref_w, ref_h, num_images,
        )
        resp.cost = cost
        resp.input_mp = round(in_mp * 100) / 100
        resp.output_mp = round(out_mp * 100) / 100

    return resp.model_dump(exclude_none=True)


@router.get("/get_result")
async def get_result(request: Request):
    task_id = request.query_params.get("id", "").strip()
    if not task_id:
        raise HTTPException(400, detail="Missing task ID parameter")

    upstream_id = _unwrap_id(task_id)
    token = extract_token(request)
    client = request.app.state.replicate_client

    try:
        prediction = await client.get_prediction(token, upstream_id)
    except UpstreamError as exc:
        if exc.status_code == 404:
            raise HTTPException(404, detail="Task not found") from exc
        raise HTTPException(
            exc.status_code if 400 <= exc.status_code < 500 else 502,
            detail=exc.detail,
        ) from exc
    except Exception:
        logger.exception("Failed to fetch prediction %s", upstream_id)
        raise HTTPException(502, detail="Upstream model provider returned an error")

    status = prediction.get("status")

    if status == "succeeded":
        sample = _extract_sample(prediction.get("output"))
        prompt = (prediction.get("input") or {}).get("prompt", "")
        duration = (prediction.get("metrics") or {}).get("predict_time", 0)
        start_time = _parse_rfc3339_to_unix(prediction.get("started_at"))
        end_time = _parse_rfc3339_to_unix(prediction.get("completed_at"))

        result = BFLTaskResult(
            id=task_id,
            status="Ready",
            result=BFLTaskResultPayload(
                prompt=prompt,
                sample=sample,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
            ),
        )
        return result.model_dump(exclude_none=False)

    if status == "failed":
        return BFLTaskResult(id=task_id, status="Error").model_dump(exclude_none=True)

    return BFLTaskResult(id=task_id, status="Pending").model_dump(exclude_none=True)


def _extract_sample(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, str):
            return first
        return json.dumps(first, ensure_ascii=False)
    return json.dumps(output, ensure_ascii=False)


def _parse_rfc3339_to_unix(s: Any) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    try:
        # fromisoformat handles trailing 'Z' from Py 3.11+; pad just in case.
        v = s.replace("Z", "+00:00")
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return 0.0


def register(app):
    app.include_router(router)
