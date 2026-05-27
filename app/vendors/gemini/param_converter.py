"""Convert a Gemini generateContent request into Replicate prediction input."""

from __future__ import annotations

from typing import Any

from app.vendors.gemini.models import (
    GenerateContentRequest,
    InlineData,
    PredictLongRunningRequest,
    SafetySetting,
    VeoInlineDataInstance,
)

MODEL_MAP: dict[str, str] = {
    "gemini-3.1-flash-image-preview": "google/nano-banana-2",
    "gemini-3-pro-image-preview": "google/nano-banana-pro",
    "gemini-2.5-flash-image": "google/nano-banana",
}

_RESOLUTION_MODELS = {"google/nano-banana-2", "google/nano-banana-pro"}
_SEARCH_MODELS = {"google/nano-banana-2"}
_SAFETY_MODELS = {"google/nano-banana-pro"}

# Gemini threshold → Replicate safety_filter_level (higher = more restrictive)
_THRESHOLD_RANK: dict[str, int] = {
    "BLOCK_NONE": 0,
    "OFF": 0,
    "BLOCK_ONLY_HIGH": 1,
    "BLOCK_MEDIUM_AND_ABOVE": 2,
    "BLOCK_LOW_AND_ABOVE": 3,
}
_RANK_TO_REPLICATE: dict[int, str] = {
    0: "block_only_high",
    1: "block_only_high",
    2: "block_medium_and_above",
    3: "block_low_and_above",
}


def get_replicate_model(gemini_model: str) -> str | None:
    return MODEL_MAP.get(gemini_model)


def _to_data_uri(d: InlineData) -> str:
    return f"data:{d.mime_type};base64,{d.data}"


def _map_safety(settings: list[SafetySetting]) -> str | None:
    """Pick the most restrictive Gemini threshold and map it to Replicate."""
    max_rank = -1
    for s in settings:
        if s.threshold:
            rank = _THRESHOLD_RANK.get(s.threshold.upper(), -1)
            if rank > max_rank:
                max_rank = rank
    if max_rank < 0:
        return None
    return _RANK_TO_REPLICATE[max_rank]


def convert_request(
    gemini_model: str,
    req: GenerateContentRequest,
) -> dict[str, Any]:
    replicate_model = MODEL_MAP[gemini_model]

    texts: list[str] = []
    images: list[str] = []
    for content in req.contents:
        for part in content.parts:
            if part.text:
                texts.append(part.text)
            if part.inline_data:
                images.append(_to_data_uri(part.inline_data))

    inp: dict[str, Any] = {"prompt": "\n".join(texts)}

    if images:
        inp["image_input"] = images

    gc = req.generation_config
    if gc and gc.image_config:
        ic = gc.image_config
        if ic.aspect_ratio:
            inp["aspect_ratio"] = ic.aspect_ratio
        if ic.image_size and replicate_model in _RESOLUTION_MODELS:
            inp["resolution"] = ic.image_size

    if req.tools and replicate_model in _SEARCH_MODELS:
        for tool in req.tools:
            gs = tool.google_search
            if gs is not None:
                inp["google_search"] = True
                if gs.search_types and gs.search_types.image_search is not None:
                    inp["image_search"] = True

    if req.safety_settings and replicate_model in _SAFETY_MODELS:
        level = _map_safety(req.safety_settings)
        if level:
            inp["safety_filter_level"] = level

    inp["output_format"] = "png"
    return inp


# ── Veo (predictLongRunning) ────────────────────────────────────────


VEO_MODEL_MAP: dict[str, str] = {
    "veo-3.1-generate-preview": "google/veo-3.1",
    "veo-3.1-fast-generate-preview": "google/veo-3.1-fast",
    "veo-3.1-lite-generate-preview": "google/veo-3.1-lite",
    "veo-3.0-generate-001": "google/veo-3",
    "veo-3.0-fast-generate-001": "google/veo-3-fast",
    "veo-2.0-generate-001": "google/veo-2",
}


def get_veo_replicate_model(gemini_model: str) -> str | None:
    return VEO_MODEL_MAP.get(gemini_model)


def _instance_to_data_uri(inst: VeoInlineDataInstance) -> str:
    if inst.gcs_uri and not inst.inline_data.data:
        raise ValueError(
            f"gcsUri inputs ({inst.gcs_uri}) are not supported by this proxy. "
            "Provide image bytes via inlineData / bytesBase64Encoded instead."
        )
    return _to_data_uri(inst.inline_data)


def build_veo_input(req: PredictLongRunningRequest) -> dict[str, Any]:
    """Convert a Gemini predictLongRunning request body to a Replicate Veo input.

    Veo on Replicate accepts a single instance per call. ``numberOfVideos`` > 1
    is rejected at the route layer so callers get a clear 400 instead of
    silently dropping the extra videos.
    """
    if not req.instances:
        raise ValueError("instances must contain at least one item")
    instance = req.instances[0]
    inp: dict[str, Any] = {}

    if instance.prompt:
        inp["prompt"] = instance.prompt

    if instance.image:
        inp["image"] = _instance_to_data_uri(instance.image)

    if instance.last_frame:
        inp["last_frame"] = _instance_to_data_uri(instance.last_frame)

    if instance.reference_images:
        inp["reference_images"] = [
            _instance_to_data_uri(ri.image) for ri in instance.reference_images
        ]

    p = req.parameters
    if p:
        if p.aspect_ratio:
            inp["aspect_ratio"] = p.aspect_ratio
        if p.duration_seconds:
            try:
                inp["duration"] = int(p.duration_seconds)
            except ValueError:
                pass
        if p.person_generation:
            inp["person_generation"] = p.person_generation
        if p.resolution:
            inp["resolution"] = p.resolution
        if p.seed is not None:
            inp["seed"] = p.seed
        if p.negative_prompt:
            inp["negative_prompt"] = p.negative_prompt

    return inp
