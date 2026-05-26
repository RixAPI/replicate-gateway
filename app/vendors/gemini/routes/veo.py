""":predictLongRunning — Veo asynchronous video generation.

Lifecycle mirrors Google's official long-running operation flow:

* POST :predictLongRunning  → create Replicate prediction, persist mapping,
                              return Operation{name, done: false}
* GET  /v1beta/{operation}  → look up prediction by name, return refreshed
                              Operation with done/response/error fields
"""

from __future__ import annotations

import logging

from fastapi import Request

from app.core.replicate_client import UpstreamError
from app.vendors.gemini.auth import require_token
from app.vendors.gemini.errors import gemini_error_response
from app.vendors.gemini.models import (
    GenerateVideoResponseBody,
    Operation,
    OperationError,
    PredictLongRunningRequest,
    PredictLongRunningResponseBody,
    VeoFileRef,
    VeoGeneratedSample,
)
from app.vendors.gemini.param_converter import (
    VEO_MODEL_MAP,
    build_veo_input,
    get_veo_replicate_model,
)

from ._helpers import dump_debug, safe_params

logger = logging.getLogger(__name__)

VENDOR = "gemini-veo"
SUPPORTED = ", ".join(VEO_MODEL_MAP.keys())


def _operation_name(model: str, task_id: str) -> str:
    return f"models/{model}/operations/{task_id}"


def _parse_operation_name(name: str) -> tuple[str | None, str | None]:
    """Parse ``models/{model}/operations/{task_id}``.

    Tolerates a leading slash and ``/v1beta`` prefix that some clients send.
    """
    s = name.lstrip("/")
    if s.startswith("v1beta/"):
        s = s[len("v1beta/"):]
    parts = s.split("/")
    if len(parts) >= 4 and parts[0] == "models" and parts[2] == "operations":
        model = parts[1]
        task_id = parts[3]
        return model, task_id
    return None, None


async def handle_predict_long_running(
    gemini_model: str,
    body: PredictLongRunningRequest,
    request: Request,
):
    replicate_model = get_veo_replicate_model(gemini_model)
    if replicate_model is None:
        return gemini_error_response(
            404,
            f"Model '{gemini_model}' is not supported. Supported: {SUPPORTED}",
            "NOT_FOUND",
        )

    if not body.instances:
        return gemini_error_response(
            400, "instances must contain at least one item", "INVALID_ARGUMENT",
        )

    if body.parameters and body.parameters.number_of_videos and body.parameters.number_of_videos > 1:
        return gemini_error_response(
            400,
            "numberOfVideos > 1 is not supported by this proxy; call once per video.",
            "INVALID_ARGUMENT",
        )

    try:
        inp = build_veo_input(body)
    except ValueError as exc:
        return gemini_error_response(400, str(exc), "INVALID_ARGUMENT")

    token = require_token(request)
    client = request.app.state.replicate_client
    store = request.app.state.task_store

    try:
        prediction = await client.create_prediction(token, replicate_model, inp)
    except UpstreamError as exc:
        resp_body = exc.detail[:1000] if isinstance(exc.detail, str) else str(exc.detail)
        dump = dump_debug(inp, replicate_model, f"HTTP {exc.status_code}: {resp_body}")
        logger.error(
            "Replicate API error on Veo create: %s | model=%s params=%s | body=%s | dump=%s",
            exc.status_code, replicate_model, safe_params(inp), resp_body, dump,
        )
        sc = exc.status_code
        if sc in (401, 403):
            return gemini_error_response(401, "Invalid API key", "UNAUTHENTICATED")
        if sc == 422:
            return gemini_error_response(
                400, f"Invalid request parameters: {resp_body}", "INVALID_ARGUMENT",
            )
        if sc == 429:
            return gemini_error_response(429, "Rate limit exceeded", "RESOURCE_EXHAUSTED")
        return gemini_error_response(502, "Upstream model provider returned an error", "INTERNAL")
    except Exception:
        dump = dump_debug(inp, replicate_model, "UnexpectedError")
        logger.exception(
            "Unexpected error on Veo create model=%s params=%s dump=%s",
            replicate_model, safe_params(inp), dump,
        )
        return gemini_error_response(500, "Internal server error", "INTERNAL")

    task_id = await store.create(
        prediction["id"], VENDOR, gemini_model, "predictLongRunning",
    )

    op = Operation(
        name=_operation_name(gemini_model, task_id),
        metadata={
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningMetadata",
        },
        done=False,
    )
    return op.model_dump(by_alias=True, exclude_none=True)


async def handle_get_operation(operation_name: str, request: Request):
    """Refresh a Veo operation by querying Replicate for the underlying prediction."""

    model, task_id = _parse_operation_name(operation_name)
    if not model or not task_id:
        return gemini_error_response(
            404, f"Operation '{operation_name}' not found", "NOT_FOUND",
        )

    store = request.app.state.task_store
    task = await store.get(task_id)
    if not task or task["vendor"] != VENDOR or task["model"] != model:
        return gemini_error_response(
            404, f"Operation '{operation_name}' not found", "NOT_FOUND",
        )

    token = require_token(request)
    client = request.app.state.replicate_client

    try:
        prediction = await client.get_prediction(token, task["prediction_id"])
    except UpstreamError as exc:
        sc = exc.status_code
        if sc in (401, 403):
            return gemini_error_response(401, "Invalid API key", "UNAUTHENTICATED")
        if sc == 404:
            return gemini_error_response(
                404, f"Operation '{operation_name}' not found", "NOT_FOUND",
            )
        return gemini_error_response(502, "Upstream model provider returned an error", "INTERNAL")
    except Exception:
        logger.exception("Failed to fetch Veo prediction %s", task["prediction_id"])
        return gemini_error_response(502, "Upstream model provider returned an error", "INTERNAL")

    status = prediction.get("status", "starting")
    op_name = _operation_name(model, task_id)

    if status in ("starting", "processing"):
        return Operation(name=op_name, done=False).model_dump(by_alias=True, exclude_none=True)

    if status == "succeeded":
        output = prediction.get("output")
        if isinstance(output, list):
            uris = [str(u) for u in output if u]
        elif isinstance(output, str):
            uris = [output]
        else:
            uris = []
        samples = [
            VeoGeneratedSample(video=VeoFileRef(uri=u, mime_type="video/mp4"))
            for u in uris
        ]
        op = Operation(
            name=op_name,
            done=True,
            response=PredictLongRunningResponseBody(
                generate_video_response=GenerateVideoResponseBody(
                    generated_samples=samples,
                ),
            ),
        )
        return op.model_dump(by_alias=True, exclude_none=True)

    if status in ("failed", "canceled"):
        msg = prediction.get("error") or (
            "Prediction was canceled" if status == "canceled" else "Prediction failed"
        )
        op = Operation(
            name=op_name,
            done=True,
            error=OperationError(code=13 if status == "failed" else 1, message=str(msg)),
        )
        return op.model_dump(by_alias=True, exclude_none=True)

    # Unknown / future status — treat as still running.
    return Operation(name=op_name, done=False).model_dump(by_alias=True, exclude_none=True)
