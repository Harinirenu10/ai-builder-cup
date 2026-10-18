from __future__ import annotations

import cv2
import numpy as np
import pytest

from fitcore.garment_photo.a4_detection import detect_a4_sheet
from fitcore.garment_photo.pipeline import A4_NOT_FOUND_MESSAGE, A4NotFoundError, measure_garment_photo
from fitcore.garment_photo.rectify import rectify_image
from fitcore.garment_photo.schemas import GarmentCategory, GarmentQualityFlags, GeminiGarmentKeypoints
from tests.garment_photo.synth_helpers import (
    build_synthetic_scene,
    expected_cm_distance,
    keypoints_to_normalized,
)

TOLERANCE_CM = 0.5


def _run_with_mocked_gemini(category: GarmentCategory, warp_strength: float = 0.03):
    image_bytes, keypoints_px_warped, offsets_mm = build_synthetic_scene(
        category, warp_strength=warp_strength
    )

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    a4_result, a4_corners = detect_a4_sheet(img)
    assert a4_result.a4_found
    rectified = rectify_image(img, a4_corners)
    canvas_h, canvas_w = rectified.shape[:2]

    normalized_keypoints = keypoints_to_normalized(
        keypoints_px_warped, a4_corners, canvas_w, canvas_h, image_shape=img.shape[:2]
    )

    def mock_generate_json(role, parts, schema, system=None, client=None):
        return GeminiGarmentKeypoints(keypoints=normalized_keypoints, quality=GarmentQualityFlags())

    result = measure_garment_photo(
        image_bytes=image_bytes,
        category=category,
        generate_json_fn=mock_generate_json,
    )
    return result, offsets_mm


def test_tops_measurements_within_half_cm_of_ground_truth():
    result, offsets_mm = _run_with_mocked_gemini(GarmentCategory.tops)

    expected_chest_flat = expected_cm_distance(offsets_mm, "left_armpit", "right_armpit")
    expected_shoulder = expected_cm_distance(offsets_mm, "left_shoulder_seam", "right_shoulder_seam")
    expected_garment_length = expected_cm_distance(offsets_mm, "back_neck_point", "hem_center")
    expected_hem_width = expected_cm_distance(offsets_mm, "hem_left", "hem_right")

    assert abs(result.measurements.chest_flat_cm - expected_chest_flat) <= TOLERANCE_CM
    assert abs(result.measurements.chest_circ_cm - expected_chest_flat * 2) <= TOLERANCE_CM
    assert abs(result.measurements.shoulder_width_cm - expected_shoulder) <= TOLERANCE_CM
    assert abs(result.measurements.garment_length_cm - expected_garment_length) <= TOLERANCE_CM
    assert abs(result.measurements.hem_width_cm - expected_hem_width) <= TOLERANCE_CM
    assert result.keypoints_source == "gemini"
    assert result.a4.a4_found is True
    assert len(result.overlay_image_png) > 0
    assert len(result.rectified_image_png) > 0


def test_bottoms_measurements_within_half_cm_of_ground_truth():
    # A milder warp here: this test isolates the deterministic keypoint-to-cm math
    # (A4 corner-localization precision under strong warps is covered separately
    # in test_a4_detection.py), and the inseam span is long enough that a few
    # pixels of corner noise under a strong warp would dominate the 0.5 cm budget.
    result, offsets_mm = _run_with_mocked_gemini(GarmentCategory.bottoms, warp_strength=0.015)

    expected_waist_flat = expected_cm_distance(offsets_mm, "waist_left", "waist_right")
    expected_hip_flat = expected_cm_distance(offsets_mm, "hip_left", "hip_right")
    expected_inseam = expected_cm_distance(offsets_mm, "crotch", "left_hem_inner")
    expected_leg_opening = expected_cm_distance(offsets_mm, "left_hem_inner", "left_hem_outer")

    assert abs(result.measurements.waist_flat_cm - expected_waist_flat) <= TOLERANCE_CM
    assert abs(result.measurements.waist_circ_cm - expected_waist_flat * 2) <= TOLERANCE_CM
    assert abs(result.measurements.hip_flat_cm - expected_hip_flat) <= TOLERANCE_CM
    assert abs(result.measurements.inseam_cm - expected_inseam) <= TOLERANCE_CM
    assert abs(result.measurements.leg_opening_cm - expected_leg_opening) <= TOLERANCE_CM


def test_missing_a4_raises_ui_safe_error():
    img = np.full((600, 800, 3), 50, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok

    with pytest.raises(A4NotFoundError) as exc_info:
        measure_garment_photo(image_bytes=encoded.tobytes(), category=GarmentCategory.tops)

    assert str(exc_info.value) == A4_NOT_FOUND_MESSAGE


def test_corrected_keypoints_skip_gemini_and_recompute_deterministically():
    image_bytes, keypoints_px_warped, offsets_mm = build_synthetic_scene(
        GarmentCategory.tops, warp_strength=0.03
    )
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    a4_result, a4_corners = detect_a4_sheet(img)
    rectified = rectify_image(img, a4_corners)
    canvas_h, canvas_w = rectified.shape[:2]
    normalized_keypoints = keypoints_to_normalized(
        keypoints_px_warped, a4_corners, canvas_w, canvas_h, image_shape=img.shape[:2]
    )

    def failing_generate_json(*args, **kwargs):
        raise AssertionError("Gemini must not be called when corrected_keypoints are supplied")

    result = measure_garment_photo(
        image_bytes=image_bytes,
        category=GarmentCategory.tops,
        corrected_keypoints=normalized_keypoints,
        generate_json_fn=failing_generate_json,
    )

    assert result.keypoints_source == "user_corrected"
    expected_chest_flat = expected_cm_distance(offsets_mm, "left_armpit", "right_armpit")
    assert abs(result.measurements.chest_flat_cm - expected_chest_flat) <= TOLERANCE_CM
