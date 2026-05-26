""":generateContent — synchronous image generation (Nano Banana family)."""

from __future__ import annotations

import logging
import uuid

from fastapi import Request

from app.core.replicate_client import UpstreamError
from app.vendors.gemini.auth import require_token
from app.vendors.gemini.errors import gemini_error_response
from app.vendors.gemini.models import (
    Candidate,
    Content,
    GenerateContentRequest,
    GenerateContentResponse,
    InlineData,
    ModalityTokenCount,
    Part,
    UsageMetadata,
)
from app.vendors.gemini.param_converter import (
    MODEL_MAP,
    convert_request,
    get_replicate_model,
)

from ._helpers import dump_debug, safe_params

logger = logging.getLogger(__name__)

SUPPORTED = ", ".join(MODEL_MAP.keys())

# Official image output token counts from Google pricing docs
# https://ai.google.dev/gemini-api/docs/pricing
_OUTPUT_IMAGE_TOKENS: dict[str, dict[str, int]] = {
    "google/nano-banana-2": {
        "512": 747, "0.5K": 747,
        "1K": 1120,
        "2K": 1680,
        "4K": 2520,
    },
    "google/nano-banana-pro": {
        "1K": 1120, "2K": 1120,
        "4K": 2000,
    },
    "google/nano-banana": {
        "1K": 1290,
    },
}
_OUTPUT_IMAGE_DEFAULTS: dict[str, int] = {
    "google/nano-banana-2": 1120,
    "google/nano-banana-pro": 1120,
    "google/nano-banana": 1290,
}
_INPUT_IMAGE_TOKENS: dict[str, int] = {
    "google/nano-banana-2": 560,
    "google/nano-banana-pro": 560,
    "google/nano-banana": 560,
}


def _collect_texts(body: GenerateContentRequest) -> list[str]:
    return [p.text for c in body.contents for p in c.parts if p.text]


def _count_images(body: GenerateContentRequest) -> int:
    return sum(1 for c in body.contents for p in c.parts if p.inline_data)


def _get_resolution(body: GenerateContentRequest) -> str | None:
    gc = body.generation_config
    if gc and gc.image_config and gc.image_config.image_size:
        return gc.image_config.image_size
    return None


async def handle_generate_content(
    gemini_model: str,
    body: GenerateContentRequest,
    request: Request,
):
    replicate_model = get_replicate_model(gemini_model)
    if replicate_model is None:
        return gemini_error_response(
            404,
            f"Model '{gemini_model}' is not supported. Supported: {SUPPORTED}",
            "NOT_FOUND",
        )

    token = require_token(request)
    client = request.app.state.replicate_client

    inp = convert_request(gemini_model, body)
    params_for_log = safe_params(inp)

    try:
        output_url, pred_id = await client.run_prediction(token, replicate_model, inp)
        logger.info(
            "Prediction %s OK model=%s params=%s",
            pred_id, replicate_model, params_for_log,
        )
        b64_data, mime_type = await client.download_image_b64(output_url)
    except UpstreamError as exc:
        resp_body = exc.detail[:1000] if isinstance(exc.detail, str) else str(exc.detail)
        dump = dump_debug(inp, replicate_model, f"HTTP {exc.status_code}: {resp_body}")
        logger.error(
            "Replicate API error: %s | model=%s params=%s | body=%s | dump=%s",
            exc.status_code, replicate_model, params_for_log, resp_body, dump,
        )
        sc = exc.status_code
        if sc in (401, 403):
            return gemini_error_response(401, "Invalid API key", "UNAUTHENTICATED")
        if sc == 422:
            return gemini_error_response(
                422, f"Invalid request parameters: {resp_body}", "INVALID_ARGUMENT",
            )
        if sc == 429:
            return gemini_error_response(429, "Rate limit exceeded", "RESOURCE_EXHAUSTED")
        return gemini_error_response(
            502, "Upstream model provider returned an error", "INTERNAL",
        )
    except TimeoutError:
        dump = dump_debug(inp, replicate_model, "TimeoutError")
        logger.exception(
            "Prediction timeout model=%s params=%s dump=%s",
            replicate_model, params_for_log, dump,
        )
        return gemini_error_response(504, "Image generation timed out", "DEADLINE_EXCEEDED")
    except RuntimeError as exc:
        error_str = str(exc)
        dump = dump_debug(inp, replicate_model, error_str)
        logger.error(
            "Prediction failed: %s model=%s params=%s dump=%s",
            error_str, replicate_model, params_for_log, dump,
        )
        lowered = error_str.lower()
        if "rate limit" in lowered or "high demand" in lowered or "(E003)" in error_str:
            return gemini_error_response(
                429, "Model is at capacity, please retry later", "RESOURCE_EXHAUSTED",
            )
        return gemini_error_response(502, f"Prediction failed: {exc}", "INTERNAL")
    except Exception:
        dump = dump_debug(inp, replicate_model, "UnexpectedError")
        logger.exception(
            "Unexpected error model=%s params=%s dump=%s",
            replicate_model, params_for_log, dump,
        )
        return gemini_error_response(500, "Internal server error", "INTERNAL")

    # ── token counts (official Google pricing data) ──────────────
    prompt_text_tokens = max(1, sum(len(t) for t in _collect_texts(body)) // 4)
    prompt_image_count = _count_images(body)
    input_img_tok = _INPUT_IMAGE_TOKENS.get(replicate_model, 560)
    prompt_image_tokens = prompt_image_count * input_img_tok
    prompt_tokens = prompt_text_tokens + prompt_image_tokens

    resolution = _get_resolution(body)
    res_table = _OUTPUT_IMAGE_TOKENS.get(replicate_model, {})
    candidate_image_tokens = (
        res_table.get(resolution, _OUTPUT_IMAGE_DEFAULTS.get(replicate_model, 1120))
        if resolution
        else _OUTPUT_IMAGE_DEFAULTS.get(replicate_model, 1120)
    )
    total_tokens = prompt_tokens + candidate_image_tokens

    prompt_details = [ModalityTokenCount(modality="TEXT", token_count=prompt_text_tokens)]
    if prompt_image_tokens:
        prompt_details.append(
            ModalityTokenCount(modality="IMAGE", token_count=prompt_image_tokens)
        )

    usage = UsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_image_tokens,
        total_token_count=total_tokens,
        prompt_tokens_details=prompt_details,
        candidates_tokens_details=[
            ModalityTokenCount(modality="IMAGE", token_count=candidate_image_tokens),
        ],
    )

    parts = [Part(inline_data=InlineData(mime_type=mime_type, data=b64_data))]
    resp = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=parts, role="model"),
                finish_reason="STOP",
                index=0,
                token_count=candidate_image_tokens,
            )
        ],
        model_version=gemini_model,
        usage_metadata=usage,
        response_id=str(uuid.uuid4()),
    )
    return resp.model_dump(by_alias=True, exclude_none=True)
