"""Gemini routes package.

Two public endpoint families share the same ``/v1beta/models/...`` prefix,
so they're dispatched from a single FastAPI catch-all handler that
inspects the ``:action`` suffix:

* ``:generateContent`` → :mod:`generate_content` (sync, image)
* ``:predictLongRunning`` → :mod:`veo` (async LRO, video)

Plus a separate path for polling Veo operations.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ValidationError

from app.vendors.gemini.auth import verify_auth
from app.vendors.gemini.errors import gemini_error_response, install_handlers
from app.vendors.gemini.models import (
    GenerateContentRequest,
    PredictLongRunningRequest,
)

from .generate_content import handle_generate_content
from .veo import handle_get_operation, handle_predict_long_running

router = APIRouter(prefix="/gemini", dependencies=[Depends(verify_auth)])


async def _parse_body(request: Request, model: type[BaseModel]):
    """Return either a validated model instance or a JSONResponse error."""
    try:
        raw = await request.json()
    except json.JSONDecodeError as exc:
        return gemini_error_response(400, f"Invalid JSON: {exc.msg}", "INVALID_ARGUMENT")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        return gemini_error_response(
            400, f"Invalid request body: {exc.errors(include_url=False)}", "INVALID_ARGUMENT",
        )


@router.post("/v1beta/models/{model_action}")
async def dispatch_model_action(model_action: str, request: Request):
    """Dispatch ``{model}:{action}`` to the right handler."""
    if ":" not in model_action:
        return gemini_error_response(404, "Not found", "NOT_FOUND")

    model, _, action = model_action.partition(":")

    if action == "generateContent":
        parsed = await _parse_body(request, GenerateContentRequest)
        if not isinstance(parsed, GenerateContentRequest):
            return parsed  # error JSONResponse
        return await handle_generate_content(model, parsed, request)

    if action == "predictLongRunning":
        parsed = await _parse_body(request, PredictLongRunningRequest)
        if not isinstance(parsed, PredictLongRunningRequest):
            return parsed
        return await handle_predict_long_running(model, parsed, request)

    return gemini_error_response(404, f"Unsupported action ':{action}'", "NOT_FOUND")


@router.get("/v1beta/models/{model}/operations/{operation_id}")
async def get_operation(model: str, operation_id: str, request: Request):
    return await handle_get_operation(
        f"models/{model}/operations/{operation_id}", request
    )


def register(app):
    install_handlers(app)
    app.include_router(router)
