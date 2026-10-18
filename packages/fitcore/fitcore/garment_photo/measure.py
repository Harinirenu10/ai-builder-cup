"""Deterministic centimeter measurement computation from garment keypoints.

Keypoints are normalized to a 0-1000 coordinate space over the rectified
canvas (see rectify.py); this module converts them back to canvas pixels
and applies the fixed rectify.CM_PER_PX scale factor to get real-world
centimeters. See fitcore.garment_photo.schemas.GARMENT_ZONE_DEFINITIONS for
the exact keypoint pair behind each measurement.
"""

from __future__ import annotations

import math

from fitcore.garment_photo.rectify import CM_PER_PX
from fitcore.garment_photo.schemas import (
    GarmentCategory,
    GarmentFlatMeasurements,
    GarmentQualityFlags,
    KeypointXY,
)

MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.99
QUALITY_FLAG_PENALTY = 0.15


def to_canvas_px(point: KeypointXY, canvas_w: int, canvas_h: int) -> tuple[float, float]:
    """Convert a normalized (0-1000) keypoint to actual pixel coordinates on the rectified canvas."""
    return (point.x / 1000.0) * canvas_w, (point.y / 1000.0) * canvas_h


def _distance_cm(
    keypoints: dict[str, KeypointXY], a: str, b: str, canvas_w: int, canvas_h: int
) -> float | None:
    if a not in keypoints or b not in keypoints:
        return None
    ax, ay = to_canvas_px(keypoints[a], canvas_w, canvas_h)
    bx, by = to_canvas_px(keypoints[b], canvas_w, canvas_h)
    px_dist = math.hypot(bx - ax, by - ay)
    return round(px_dist * CM_PER_PX, 2)


def compute_confidence(a4_confidence: float, quality: GarmentQualityFlags) -> float:
    """Derive an overall measurement confidence from A4 detection quality and garment quality flags."""
    confidence = a4_confidence
    for flag_set in (quality.folded_or_overlapping, quality.not_flat, quality.partly_out_of_frame):
        if flag_set:
            confidence -= QUALITY_FLAG_PENALTY
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))


def compute_garment_measurements(
    keypoints: dict[str, KeypointXY],
    category: GarmentCategory,
    canvas_width_px: int,
    canvas_height_px: int,
    a4_confidence: float,
    quality: GarmentQualityFlags,
) -> GarmentFlatMeasurements:
    """Compute all applicable flat-lay measurements in centimeters for the given category."""
    confidence = compute_confidence(a4_confidence, quality)
    d = lambda a, b: _distance_cm(keypoints, a, b, canvas_width_px, canvas_height_px)

    if category in (GarmentCategory.tops, GarmentCategory.kurtas):
        chest_flat = d("left_armpit", "right_armpit")
        shoulder_width = d("left_shoulder_seam", "right_shoulder_seam")
        sleeve_length = d("left_shoulder_seam", "left_cuff_outer")
        garment_length = d("back_neck_point", "hem_center")
        hem_width = d("hem_left", "hem_right")
        return GarmentFlatMeasurements(
            chest_flat_cm=chest_flat,
            chest_circ_cm=round(chest_flat * 2.0, 2) if chest_flat is not None else None,
            shoulder_width_cm=shoulder_width,
            sleeve_length_cm=sleeve_length,
            garment_length_cm=garment_length,
            hem_width_cm=hem_width,
            confidence=confidence,
        )

    waist_flat = d("waist_left", "waist_right")
    hip_flat = d("hip_left", "hip_right")
    inseam = d("crotch", "left_hem_inner")
    leg_opening = d("left_hem_inner", "left_hem_outer")
    return GarmentFlatMeasurements(
        waist_flat_cm=waist_flat,
        waist_circ_cm=round(waist_flat * 2.0, 2) if waist_flat is not None else None,
        hip_flat_cm=hip_flat,
        inseam_cm=inseam,
        leg_opening_cm=leg_opening,
        confidence=confidence,
    )
