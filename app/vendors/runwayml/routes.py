"""Generation and task-management routes mirroring the RunwayML API."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.replicate_client import UpstreamError
from app.vendors.runwayml.auth import extract_token, verify_auth
from app.vendors.runwayml.models import (
    CharacterPerformanceRequest,
    ErrorResponse,
    ImageToVideoRequest,
    TaskCreatedResponse,
    TaskDetailResponse,
    TextToVideoRequest,
    VideoToVideoRequest,
)
from app.vendors.runwayml.param_converter import (
    build_image_to_video_input,
    build_text_to_video_input,
    build_video_to_video_input,
    convert_status,
    resolve_replicate_model,
)

logger = logging.getLogger(__name__)

VENDOR = "runwayml"


def _stringify_error(err: object) -> str:
    """Coerce an upstream `error` field (str / dict / None / other) into a string."""
    if err is None:
        return ""
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return err.get("message") or err.get("detail") or json.dumps(err, ensure_ascii=False)
    return str(err)

router = APIRouter(prefix="/runwayml/v1", tags=["RunwayML"], dependencies=[Depends(verify_auth)])


@router.post(
    "/image_to_video",
    response_model=TaskCreatedResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def image_to_video(body: ImageToVideoRequest, request: Request):
    replicate_model = resolve_replicate_model(body.model)
    if not replicate_model:
        raise HTTPException(400, detail=f"Unsupported model: {body.model}")

    inp = build_image_to_video_input(body)
    return await _create_task(request, replicate_model, inp, body.model, "image_to_video")


@router.post(
    "/text_to_video",
    response_model=TaskCreatedResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def text_to_video(body: TextToVideoRequest, request: Request):
    replicate_model = resolve_replicate_model(body.model)
    if not replicate_model:
        raise HTTPException(400, detail=f"Unsupported model: {body.model}")

    inp = build_text_to_video_input(body)
    return await _create_task(request, replicate_model, inp, body.model, "text_to_video")


@router.post(
    "/video_to_video",
    response_model=TaskCreatedResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def video_to_video(body: VideoToVideoRequest, request: Request):
    replicate_model = resolve_replicate_model(body.model)
    if not replicate_model:
        raise HTTPException(400, detail=f"Unsupported model: {body.model}")

    inp = build_video_to_video_input(body)
    return await _create_task(request, replicate_model, inp, body.model, "video_to_video")


@router.post("/character_performance")
async def character_performance(body: CharacterPerformanceRequest):
    raise HTTPException(
        501,
        detail="character_performance (act_two) is not yet available on Replicate. "
        "This endpoint will be implemented once the model is supported.",
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_task(task_id: str, request: Request):
    token = extract_token(request)
    store = request.app.state.task_store
    task = await store.get(task_id)
    if not task or task["vendor"] != VENDOR:
        raise HTTPException(404, detail="Task not found")

    client = request.app.state.replicate_client
    try:
        prediction = await client.get_prediction(token, task["prediction_id"])
    except UpstreamError as exc:
        detail = _stringify_error(exc.detail)
        logger.error("Replicate API error for prediction %s: %s",
                     task["prediction_id"], detail)
        sc = exc.status_code
        if sc in (401, 403):
            raise HTTPException(401, detail="Invalid or unauthorized API token") from exc
        if sc == 404:
            # Upstream lost the prediction — surface as 404 since that's
            # genuinely "task not found" at this point.
            raise HTTPException(404, detail="Task not found") from exc
        if 400 <= sc < 500:
            raise HTTPException(sc, detail=detail) from exc
        raise HTTPException(502, detail="Upstream model provider returned an error") from exc
    except Exception as exc:
        logger.exception("Failed to fetch prediction %s", task["prediction_id"])
        raise HTTPException(502, detail="Upstream model provider returned an error") from exc

    status = convert_status(prediction.get("status", "starting"))

    output = None
    if status == "SUCCEEDED":
        raw_output = prediction.get("output")
        if isinstance(raw_output, str):
            output = [raw_output]
        elif isinstance(raw_output, list):
            output = raw_output
        elif raw_output is not None:
            output = [str(raw_output)]

    failure = None
    failure_code = None
    if status == "FAILED":
        failure = _stringify_error(prediction.get("error")) or "Unknown error"
        failure_code = "INTERNAL_ERROR"
    elif status == "CANCELLED":
        # Surface a friendly message so the client can distinguish cancellation
        # from a generic failure without inspecting upstream details.
        failure = _stringify_error(prediction.get("error")) or "Task was cancelled"
        failure_code = "CANCELLED"

    return TaskDetailResponse(
        id=task_id,
        status=status,
        createdAt=task["created_at"],
        output=output,
        failure=failure,
        failureCode=failure_code,
    )


@router.delete(
    "/tasks/{task_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_task(task_id: str, request: Request):
    token = extract_token(request)
    store = request.app.state.task_store
    task = await store.get(task_id)
    if not task or task["vendor"] != VENDOR:
        raise HTTPException(404, detail="Task not found")

    client = request.app.state.replicate_client
    try:
        await client.cancel_prediction(token, task["prediction_id"])
    except Exception:
        logger.warning(
            "Could not cancel prediction %s (may already be finished)",
            task["prediction_id"],
        )

    await store.delete(task_id)
    return {"detail": "Task cancelled / deleted"}


async def _create_task(
    request: Request,
    replicate_model: str,
    inp: dict,
    runway_model: str,
    endpoint: str,
) -> TaskCreatedResponse:
    token = extract_token(request)
    client = request.app.state.replicate_client
    store = request.app.state.task_store

    try:
        prediction = await client.create_prediction(token, replicate_model, inp)
    except UpstreamError as exc:
        detail = _stringify_error(exc.detail)
        logger.error("Replicate API error for model %s: %s", replicate_model, detail)
        sc = exc.status_code
        if sc in (401, 403):
            raise HTTPException(401, detail="Invalid or unauthorized API token") from exc
        if sc == 404:
            # 404 on create means the (model, version) wasn't reachable upstream.
            # Surface as 502 rather than a literal 404 so clients don't mistake
            # it for "task not found" on a yet-to-be-created task.
            raise HTTPException(502, detail=f"Upstream model not reachable: {detail}") from exc
        if sc == 422:
            raise HTTPException(400, detail=f"Invalid request parameters: {detail}") from exc
        if sc == 429:
            raise HTTPException(429, detail="Rate limit exceeded") from exc
        if 400 <= sc < 500:
            raise HTTPException(sc, detail=detail) from exc
        raise HTTPException(502, detail="Upstream model provider returned an error") from exc
    except Exception as exc:
        logger.exception("Unexpected error calling Replicate for model %s", replicate_model)
        raise HTTPException(502, detail="Upstream model provider returned an error") from exc

    task_id = await store.create(
        prediction["id"], VENDOR, runway_model, endpoint,
    )
    return TaskCreatedResponse(id=task_id)


def register(app):
    app.include_router(router)
