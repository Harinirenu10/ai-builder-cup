from __future__ import annotations

from fitcore.garment_photo.measure import compute_confidence, compute_garment_measurements
from fitcore.garment_photo.rectify import CM_PER_PX
from fitcore.garment_photo.schemas import GarmentCategory, GarmentQualityFlags, KeypointXY

CANVAS_W, CANVAS_H = 1000, 1000


def _kp(x_px: float, y_px: float) -> KeypointXY:
    return KeypointXY(x=x_px / CANVAS_W * 1000.0, y=y_px / CANVAS_H * 1000.0)


def test_tops_measurements_match_known_pixel_distances():
    keypoints = {
        "left_armpit": _kp(100, 100),
        "right_armpit": _kp(100 + 200, 100),  # 200px apart
        "left_shoulder_seam": _kp(100, 50),
        "right_shoulder_seam": _kp(100 + 160, 50),  # 160px apart
        "back_neck_point": _kp(150, 20),
        "hem_center": _kp(150, 20 + 400),  # 400px apart
        "hem_left": _kp(80, 420),
        "hem_right": _kp(80 + 220, 420),  # 220px apart
    }
    result = compute_garment_measurements(
        keypoints=keypoints,
        category=GarmentCategory.tops,
        canvas_width_px=CANVAS_W,
        canvas_height_px=CANVAS_H,
        a4_confidence=0.9,
        quality=GarmentQualityFlags(),
    )

    assert abs(result.chest_flat_cm - 200 * CM_PER_PX) < 0.01
    assert abs(result.chest_circ_cm - 200 * CM_PER_PX * 2) < 0.01
    assert abs(result.shoulder_width_cm - 160 * CM_PER_PX) < 0.01
    assert abs(result.garment_length_cm - 400 * CM_PER_PX) < 0.01
    assert abs(result.hem_width_cm - 220 * CM_PER_PX) < 0.01
    # bottoms-only fields stay unset for a tops garment
    assert result.waist_flat_cm is None
    assert result.inseam_cm is None


def test_bottoms_measurements_match_known_pixel_distances():
    keypoints = {
        "waist_left": _kp(100, 100),
        "waist_right": _kp(100 + 180, 100),
        "hip_left": _kp(90, 150),
        "hip_right": _kp(90 + 210, 150),
        "crotch": _kp(195, 180),
        "left_hem_inner": _kp(195, 180 + 500),
        "left_hem_outer": _kp(195 + 60, 180 + 500),
    }
    result = compute_garment_measurements(
        keypoints=keypoints,
        category=GarmentCategory.bottoms,
        canvas_width_px=CANVAS_W,
        canvas_height_px=CANVAS_H,
        a4_confidence=0.9,
        quality=GarmentQualityFlags(),
    )

    assert abs(result.waist_flat_cm - 180 * CM_PER_PX) < 0.01
    assert abs(result.waist_circ_cm - 180 * CM_PER_PX * 2) < 0.01
    assert abs(result.hip_flat_cm - 210 * CM_PER_PX) < 0.01
    assert abs(result.inseam_cm - 500 * CM_PER_PX) < 0.01
    assert abs(result.leg_opening_cm - 60 * CM_PER_PX) < 0.01


def test_missing_keypoint_yields_none_for_dependent_measurement():
    keypoints = {"left_armpit": _kp(100, 100)}  # right_armpit missing
    result = compute_garment_measurements(
        keypoints=keypoints,
        category=GarmentCategory.tops,
        canvas_width_px=CANVAS_W,
        canvas_height_px=CANVAS_H,
        a4_confidence=0.9,
        quality=GarmentQualityFlags(),
    )
    assert result.chest_flat_cm is None
    assert result.chest_circ_cm is None


def test_confidence_reduced_by_quality_flags():
    base = compute_confidence(0.9, GarmentQualityFlags())
    degraded = compute_confidence(
        0.9, GarmentQualityFlags(folded_or_overlapping=True, not_flat=True)
    )
    assert degraded < base
    assert degraded >= 0.05
