"""End-to-end garment flat-lay measurement pipeline.

1. Detect the A4 reference sheet in the source photo.
2. Rectify the full photo onto a fixed-scale (2 px/mm) canvas via homography.
3. Detect category-specific keypoints with Gemini Flash, unless user-corrected
   keypoints were supplied, in which case Gemini is skipped entirely.
4. Compute deterministic centimeter measurements from the keypoints.
5. Render a diagnostic overlay PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from fitcore.garment_photo.a4_detection import detect_a4_sheet
from fitcore.garment_photo.keypoints import GenerateJsonFn, detect_garment_keypoints
from fitcore.garment_photo.measure import compute_garment_measurements
from fitcore.garment_photo.overlay import generate_overlay_image
from fitcore.garment_photo.rectify import rectify_image
from fitcore.garment_photo.schemas import (
    A4DetectionResult,
    GarmentCategory,
    GarmentFlatMeasurements,
    GarmentQualityFlags,
    KeypointXY,
)

MIN_A4_CONFIDENCE = 0.5

A4_NOT_FOUND_MESSAGE = "Place a full A4 sheet next to the garment."


class A4NotFoundError(ValueError):
    """Raised when no sufficiently confident A4 sheet is found in the photo.

    The message is UI-safe and intended to be shown directly to the shopper.
    """

    def __init__(self) -> None:
        super().__init__(A4_NOT_FOUND_MESSAGE)


@dataclass
class GarmentPhotoPipelineResult:
    category: GarmentCategory
    a4: A4DetectionResult
    keypoints: dict[str, KeypointXY]
    keypoints_source: str  # "gemini" | "user_corrected"
    quality: GarmentQualityFlags
    measurements: GarmentFlatMeasurements
    rectified_image_png: bytes
    overlay_image_png: bytes


def measure_garment_photo(
    image_bytes: bytes,
    category: GarmentCategory,
    corrected_keypoints: dict[str, KeypointXY] | None = None,
    generate_json_fn: GenerateJsonFn | None = None,
    llm_client: Any | None = None,
) -> GarmentPhotoPipelineResult:
    """Run the full garment flat-lay measurement pipeline on a single photo.

    Args:
        image_bytes: Raw encoded (JPEG/PNG) source photo bytes.
        category: Garment category, selects the keypoint vocabulary.
        corrected_keypoints: Optional user-corrected keypoints (normalized 0-1000,
            in the same coordinate space the overlay from a prior call showed).
            When supplied, Gemini is skipped and measurements are recomputed
            deterministically from these points.
        generate_json_fn: Optional injected replacement for fitcore.llm.generate_json (testing).
        llm_client: Optional pre-configured/mock LLM client forwarded to generate_json.

    Returns:
        GarmentPhotoPipelineResult with measurements, keypoints, and rendered images.

    Raises:
        A4NotFoundError: If no A4 sheet is confidently detected (UI-safe message).
        ValueError: If the image bytes cannot be decoded.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode garment photo. Please upload a valid JPEG or PNG image.")

    a4_result, a4_corners = detect_a4_sheet(image_bgr)
    if not a4_result.a4_found or a4_corners is None or a4_result.a4_confidence < MIN_A4_CONFIDENCE:
        raise A4NotFoundError()

    rectified_bgr = rectify_image(image_bgr, a4_corners)
    canvas_h, canvas_w = rectified_bgr.shape[:2]

    if corrected_keypoints is not None:
        keypoints = corrected_keypoints
        quality = GarmentQualityFlags()
        keypoints_source = "user_corrected"
    else:
        gemini_result = detect_garment_keypoints(
            rectified_bgr, category, generate_json_fn=generate_json_fn, llm_client=llm_client
        )
        keypoints = gemini_result.keypoints
        quality = gemini_result.quality
        keypoints_source = "gemini"

    measurements = compute_garment_measurements(
        keypoints=keypoints,
        category=category,
        canvas_width_px=canvas_w,
        canvas_height_px=canvas_h,
        a4_confidence=a4_result.a4_confidence,
        quality=quality,
    )

    overlay_png = generate_overlay_image(
        rectified_bgr=rectified_bgr,
        keypoints=keypoints,
        category=category,
        measurements=measurements,
        quality=quality,
        a4=a4_result,
    )

    ok, rectified_encoded = cv2.imencode(".png", rectified_bgr)
    if not ok:
        raise ValueError("Failed to encode rectified image.")

    return GarmentPhotoPipelineResult(
        category=category,
        a4=a4_result,
        keypoints=keypoints,
        keypoints_source=keypoints_source,
        quality=quality,
        measurements=measurements,
        rectified_image_png=rectified_encoded.tobytes(),
        overlay_image_png=overlay_png,
    )
