"""Category-specific garment keypoint detection via Gemini Flash.

All LLM calls route through fitcore.llm.generate_json (see CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from fitcore.garment_photo.schemas import (
    CATEGORY_KEYPOINTS,
    GarmentCategory,
    GeminiGarmentKeypoints,
)

GenerateJsonFn = Callable[..., GeminiGarmentKeypoints]


def _build_prompt(category: GarmentCategory) -> str:
    names = CATEGORY_KEYPOINTS[category]
    names_list = "\n".join(f"- {n}" for n in names)
    return (
        f"This is a rectified top-down photo of a {category.value} garment laid flat next to "
        "an A4 reference sheet, photographed on a contrasting surface. Locate each of the "
        "following garment construction keypoints and return its pixel position normalized "
        "to a 0-1000 coordinate space over the image width (x) and height (y):\n"
        f"{names_list}\n\n"
        "Also report quality flags: whether the garment is folded or overlapping itself, "
        "not laid fully flat, or partly out of frame. Describe only the garment, never the "
        "person or body. If a keypoint is not visible, omit it from the keypoints map rather "
        "than guessing."
    )


def detect_garment_keypoints(
    rectified_image_bgr: np.ndarray,
    category: GarmentCategory,
    generate_json_fn: GenerateJsonFn | None = None,
    llm_client: Any | None = None,
) -> GeminiGarmentKeypoints:
    """Call Gemini Flash to locate category-specific keypoints on the rectified image.

    Args:
        rectified_image_bgr: Rectified (A4-scaled) canvas in BGR, as produced by rectify.rectify_image.
        category: Garment category, determines the keypoint vocabulary requested.
        generate_json_fn: Optional injected callable (for tests), replacing fitcore.llm.generate_json.
        llm_client: Optional pre-configured/mock LLM client forwarded to generate_json.

    Returns:
        GeminiGarmentKeypoints with normalized (0-1000) keypoints and quality flags.
    """
    ok, encoded = cv2.imencode(".png", rectified_image_bgr)
    if not ok:
        raise ValueError("Failed to encode rectified image for Gemini keypoint detection.")

    parts: list[Any] = [_build_prompt(category), encoded.tobytes()]

    system = (
        "You are a precise garment measurement assistant. You describe only the garment's "
        "construction and dimensions. Never comment on a person's body, weight, or how a "
        "garment would flatter someone."
    )

    if generate_json_fn is not None:
        return generate_json_fn(
            role="flash", parts=parts, schema=GeminiGarmentKeypoints, system=system, client=llm_client
        )

    from fitcore.llm import generate_json

    return generate_json(
        role="flash", parts=parts, schema=GeminiGarmentKeypoints, system=system, client=llm_client
    )
