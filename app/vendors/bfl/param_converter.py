"""BFL ↔ Replicate parameter conversion + pricing.

Mirrors the original Go gateway (replicate->bfl) one-to-one:

* Aspect-ratio fuzzy matching against per-model allow-lists.
* FLUX.2 family (Max / Pro / Flex / Klein) — separate field shape than
  the legacy flux-pro-1.x family.
* Image cost calculation in cents, with the same caps (4 MP) and same
  pricing rules as the Go version.

Image *uploading* (base64 → R2 URL) and *dimension reading* (URL → MP)
live in routes.py; this module deals only with shape transformation
and arithmetic so it stays pure and easy to unit-test.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Aspect-ratio allow-lists ────────────────────────────────────────


VALID_ASPECT_RATIOS_LEGACY: list[str] = [
    "match_input_image",
    "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "4:5", "5:4", "21:9",
    "9:21", "2:1", "1:2",
]

VALID_ASPECT_RATIOS_MAX_PRO_FLEX: list[str] = [
    "match_input_image",
    "1:1", "16:9", "3:2", "2:3", "4:5", "5:4", "9:16", "3:4", "4:3",
]

VALID_ASPECT_RATIOS_KLEIN: list[str] = [
    "match_input_image",
    "1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "21:9", "9:21",
]


def _parse_ratio(s: str) -> Optional[float]:
    if not s or s in ("match_input_image", "custom"):
        return None
    parts = s.split(":")
    if len(parts) != 2:
        return None
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if h == 0:
        return None
    return w / h


def find_closest_aspect_ratio(ratio: str, valid_list: list[str]) -> str:
    if ratio in valid_list:
        return ratio
    target = _parse_ratio(ratio)
    if target is None:
        logger.info("Invalid aspect_ratio format %r, defaulting to 1:1", ratio)
        return "1:1"
    best_name, best_diff = "1:1", float("inf")
    for v in valid_list:
        vr = _parse_ratio(v)
        if vr is None:
            continue
        diff = abs(target - vr)
        if diff < best_diff:
            best_diff, best_name = diff, v
    return best_name


# ── Model name mapping (BFL → Replicate) ────────────────────────────


_LEGACY_MODEL_MAP: dict[str, str] = {
    "flux-pro-1.1": "flux-1.1-pro",
    "flux-pro-1.1-ultra": "flux-1.1-pro-ultra",
    "flux-pro-1.0-fill": "flux-fill-pro",
    "flux-pro-1.0-expand": "flux-pro",
    "flux-pro-1.0-canny": "flux-canny-pro",
    "flux-pro-1.0-depth": "flux-depth-pro",
}


def map_model_name(bfl_model: str) -> str:
    """Translate a BFL model id to a Replicate model id (without ``black-forest-labs/``)."""
    return _LEGACY_MODEL_MAP.get(bfl_model, bfl_model)


def replicate_model_path(bfl_model: str) -> str:
    """Full Replicate model identifier (``black-forest-labs/<name>``)."""
    return f"black-forest-labs/{map_model_name(bfl_model)}"


def is_flux2_model(model: str) -> bool:
    return model.startswith("flux-2-")


# ── Pricing ──────────────────────────────────────────────────────────


class PricingRule:
    __slots__ = ("base", "subsequent", "reference")

    def __init__(self, base: float, subsequent: float, reference: float):
        self.base = base
        self.subsequent = subsequent
        self.reference = reference


# Cents per MP (matches the Go gateway exactly).
MODEL_PRICING: dict[str, PricingRule] = {
    "flux-2-max":      PricingRule(base=7.0, subsequent=3.0, reference=3.0),
    "flux-2-klein-9b": PricingRule(base=1.5, subsequent=0.2, reference=0.2),
    "flux-2-klein-4b": PricingRule(base=1.4, subsequent=0.1, reference=0.1),
    "flux-2-pro":      PricingRule(base=3.0, subsequent=1.5, reference=1.5),
    "flux-2-flex":     PricingRule(base=5.0, subsequent=5.0, reference=5.0),
}

_PIXELS_PER_MP = 1024.0 * 1024.0


def calculate_image_cost(
    model: str,
    raw_input: dict[str, Any],
    input_mp: float,
    ref_w: int,
    ref_h: int,
    num_images: int,
) -> tuple[float, float, float]:
    """Return ``(cost_cents, raw_input_mp, output_mp)``.

    Mirrors the Go ``calculateImageCost`` exactly so callers get the same
    numbers from either gateway.
    """
    rule = MODEL_PRICING.get(model)
    if rule is None:
        return 0.0, 0.0, 0.0

    # 1. Default dimensions: ref-image size for single-image mode, else 1024².
    default_w, default_h = 1024, 1024
    if num_images == 1 and ref_w > 0 and ref_h > 0:
        default_w, default_h = ref_w, ref_h

    # 2. Honour explicit width/height, clamped to (0, 2048].
    w_req = _num(raw_input.get("width"))
    h_req = _num(raw_input.get("height"))
    final_w = int(w_req) if w_req and w_req > 0 else default_w
    final_h = int(h_req) if h_req and h_req > 0 else default_h
    final_w = min(final_w, 2048)
    final_h = min(final_h, 2048)

    output_mp = (final_w * final_h) / _PIXELS_PER_MP

    pricing_input_mp = math.ceil(min(input_mp, 4.0))
    pricing_output_mp = math.ceil(min(output_mp, 4.0))

    cost = rule.base
    if pricing_output_mp > 1.0:
        cost += (pricing_output_mp - 1.0) * rule.subsequent
    cost += pricing_input_mp * rule.reference

    logger.info(
        "Cost calc: model=%s outputMP=%.2f (w=%d,h=%d,pricing=%.0f) "
        "inputMP=%.2f (pricing=%.0f) cost=%.2f cents",
        model, output_mp, final_w, final_h, pricing_output_mp,
        input_mp, pricing_input_mp, cost,
    )
    return cost, input_mp, output_mp


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ── FLUX.2 input builder ────────────────────────────────────────────


def _round_to_multiple_of_32_clamped(value: int) -> int:
    """Mirror Go's ((v+15)/32)*32 clamp/round used by Max/Pro/Flex."""
    if value < 256:
        value = 256
    if value > 2048:
        value = 2048
    return ((value + 15) // 32) * 32


_VALID_KLEIN_MP_BUCKETS = (0.25, 0.5, 1.0, 2.0, 4.0)


def build_flux2_replicate_input(
    model: str,
    raw_input: dict[str, Any],
    input_images: list[str],
) -> dict[str, Any]:
    """Convert a BFL FLUX.2 request body to the Replicate input shape.

    Mirrors Go's ``buildFlux2ReplicateInput``.
    """
    result: dict[str, Any] = {}

    # ── Generic ───────────────────────────────────────────────────
    prompt = raw_input.get("prompt")
    if isinstance(prompt, str):
        result["prompt"] = prompt

    seed = raw_input.get("seed")
    if seed is not None:
        result["seed"] = seed

    fmt = raw_input.get("output_format")
    if isinstance(fmt, str) and fmt:
        if fmt == "jpeg":
            result["output_format"] = "jpg"
        elif fmt in ("png", "jpg", "webp"):
            result["output_format"] = fmt
        else:
            result["output_format"] = "jpg"

    is_klein = model.startswith("flux-2-klein")

    if is_klein:
        # ── Klein-specific ────────────────────────────────────────
        if input_images:
            result["images"] = input_images

        w_req = _num(raw_input.get("width"))
        h_req = _num(raw_input.get("height"))
        if w_req and h_req and w_req > 0 and h_req > 0:
            mp = math.ceil((w_req * h_req) / _PIXELS_PER_MP)
            mp = max(1, min(4, int(mp)))
            closest = min(_VALID_KLEIN_MP_BUCKETS, key=lambda v: abs(mp - v))
            result["output_megapixels"] = _format_mp(closest)
            logger.info(
                "Klein: width=%d height=%d → output_megapixels=%s",
                int(w_req), int(h_req), _format_mp(closest),
            )
            result["aspect_ratio"] = find_closest_aspect_ratio(
                f"{int(w_req)}:{int(h_req)}", VALID_ASPECT_RATIOS_KLEIN,
            )
        else:
            ar = raw_input.get("aspect_ratio")
            if isinstance(ar, str) and ar:
                result["aspect_ratio"] = find_closest_aspect_ratio(
                    ar, VALID_ASPECT_RATIOS_KLEIN,
                )

        st = _num(raw_input.get("safety_tolerance"))
        if st is not None:
            result["disable_safety_checker"] = int(st) >= 5
    else:
        # ── Max / Pro / Flex ──────────────────────────────────────
        if input_images:
            result["input_images"] = input_images

        w_req = _num(raw_input.get("width"))
        h_req = _num(raw_input.get("height"))
        if w_req and h_req and w_req > 0 and h_req > 0:
            result["aspect_ratio"] = "custom"
            result["width"] = _round_to_multiple_of_32_clamped(int(w_req))
            result["height"] = _round_to_multiple_of_32_clamped(int(h_req))
            logger.info(
                "Max/Pro/Flex: custom dims %dx%d",
                result["width"], result["height"],
            )
        else:
            ar = raw_input.get("aspect_ratio")
            if isinstance(ar, str) and ar:
                result["aspect_ratio"] = find_closest_aspect_ratio(
                    ar, VALID_ASPECT_RATIOS_MAX_PRO_FLEX,
                )

        st = _num(raw_input.get("safety_tolerance"))
        if st is not None:
            clamped = max(1, min(5, int(st)))
            result["safety_tolerance"] = clamped

        if model == "flux-2-flex":
            for k in ("steps", "guidance"):
                if raw_input.get(k) is not None:
                    result[k] = raw_input[k]

    return result


def _format_mp(mp: float) -> str:
    if mp == int(mp):
        return str(int(mp))
    return str(mp)


# ── Legacy (flux-pro-1.x) input post-processing ─────────────────────


def normalize_legacy_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Fix up output_format + aspect_ratio for legacy flux models.

    Returns a shallow copy with normalized fields. The caller still owns
    the ``input_image`` replacement (R2 upload happens elsewhere).
    """
    out = dict(raw_input)
    fmt = out.get("output_format")
    if isinstance(fmt, str) and fmt and fmt not in ("png", "jpg"):
        out["output_format"] = "jpg" if fmt == "jpeg" else "png"

    ar = out.get("aspect_ratio")
    if isinstance(ar, str) and ar:
        adjusted = find_closest_aspect_ratio(ar, VALID_ASPECT_RATIOS_LEGACY)
        if adjusted != ar:
            logger.info("aspect_ratio adjusted: %s → %s", ar, adjusted)
        out["aspect_ratio"] = adjusted
    return out
