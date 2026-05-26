"""Convert RunwayML API parameters to Replicate prediction inputs."""

from __future__ import annotations

from math import gcd

from app.vendors.runwayml.models import (
    ImageToVideoRequest,
    TextToVideoRequest,
    VideoToVideoRequest,
)

RUNWAY_TO_REPLICATE_MODEL: dict[str, str] = {
    "gen4.5": "runwayml/gen-4.5",
    "gen4_turbo": "runwayml/gen4-turbo",
    "gen4_aleph": "runwayml/gen4-aleph",
}

_RATIO_MAP: dict[str, str] = {
    "1280:720": "16:9",
    "720:1280": "9:16",
    "848:480": "16:9",
    "480:848": "9:16",
    "1584:672": "21:9",
    "672:1584": "9:21",
    "1104:832": "4:3",
    "832:1104": "3:4",
    "640:480": "4:3",
    "480:640": "3:4",
    "960:960": "1:1",
    "1280:768": "5:3",
    "768:1280": "3:5",
}


def convert_ratio(runway_ratio: str | None) -> str:
    """Convert a RunwayML pixel-based ratio to the simplified ratio Replicate expects.

    If the ratio is already simplified (e.g. ``"16:9"``) or not in the map,
    compute the simplified form from the pixel values.
    """
    if not runway_ratio:
        return "16:9"

    if runway_ratio in _RATIO_MAP:
        return _RATIO_MAP[runway_ratio]

    parts = runway_ratio.split(":")
    if len(parts) == 2:
        try:
            w, h = int(parts[0]), int(parts[1])
            if w <= 100 and h <= 100:
                return runway_ratio
            g = gcd(w, h)
            return f"{w // g}:{h // g}"
        except ValueError:
            pass

    return "16:9"


_STATUS_MAP: dict[str, str] = {
    "starting": "PENDING",
    "processing": "RUNNING",
    "succeeded": "SUCCEEDED",
    "failed": "FAILED",
    "canceled": "CANCELLED",
}


def convert_status(replicate_status: str) -> str:
    return _STATUS_MAP.get(replicate_status, "RUNNING")


def build_image_to_video_input(req: ImageToVideoRequest) -> dict:
    inp: dict = {}
    if req.promptText:
        inp["prompt"] = req.promptText
    if req.promptImage:
        inp["image"] = req.promptImage
    inp["duration"] = req.duration or 5
    inp["aspect_ratio"] = convert_ratio(req.ratio)
    if req.seed is not None:
        inp["seed"] = req.seed
    return inp


def build_text_to_video_input(req: TextToVideoRequest) -> dict:
    inp: dict = {
        "prompt": req.promptText,
        "duration": req.duration or 5,
        "aspect_ratio": convert_ratio(req.ratio),
    }
    if req.seed is not None:
        inp["seed"] = req.seed
    return inp


def build_video_to_video_input(req: VideoToVideoRequest) -> dict:
    inp: dict = {
        "video": req.videoUri,
        "aspect_ratio": convert_ratio(req.ratio),
    }
    if req.promptText:
        inp["prompt"] = req.promptText
    if req.referenceImages and len(req.referenceImages) > 0:
        inp["reference_image"] = req.referenceImages[0].uri
    if req.seed is not None:
        inp["seed"] = req.seed
    return inp


def resolve_replicate_model(runway_model: str) -> str | None:
    return RUNWAY_TO_REPLICATE_MODEL.get(runway_model)
