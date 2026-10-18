"""Overlay PNG generation: keypoints, measurement lines/labels, and quality/confidence info."""

from __future__ import annotations

import cv2
import numpy as np

from fitcore.garment_photo.measure import to_canvas_px
from fitcore.garment_photo.schemas import (
    A4DetectionResult,
    GarmentCategory,
    GarmentFlatMeasurements,
    GarmentQualityFlags,
    KeypointXY,
)

POINT_COLOR = (0, 140, 255)  # BGR orange
LINE_COLOR = (255, 0, 140)  # BGR magenta
TEXT_COLOR = (20, 20, 20)
BG_PANEL_COLOR = (255, 255, 255)

_TOP_LINES: tuple[tuple[str, str, str], ...] = (
    ("left_armpit", "right_armpit", "chest"),
    ("left_shoulder_seam", "right_shoulder_seam", "shoulder"),
    ("left_shoulder_seam", "left_cuff_outer", "sleeve"),
    ("back_neck_point", "hem_center", "length"),
    ("hem_left", "hem_right", "hem"),
)
_BOTTOM_LINES: tuple[tuple[str, str, str], ...] = (
    ("waist_left", "waist_right", "waist"),
    ("hip_left", "hip_right", "hip"),
    ("crotch", "left_hem_inner", "inseam"),
    ("left_hem_inner", "left_hem_outer", "opening"),
)


def _measurement_label_for(label: str, measurements: GarmentFlatMeasurements) -> str | None:
    field_map = {
        "chest": measurements.chest_flat_cm,
        "shoulder": measurements.shoulder_width_cm,
        "sleeve": measurements.sleeve_length_cm,
        "length": measurements.garment_length_cm,
        "hem": measurements.hem_width_cm,
        "waist": measurements.waist_flat_cm,
        "hip": measurements.hip_flat_cm,
        "inseam": measurements.inseam_cm,
        "opening": measurements.leg_opening_cm,
    }
    val = field_map.get(label)
    return f"{label}: {val:.1f} cm" if val is not None else None


def generate_overlay_image(
    rectified_bgr: np.ndarray,
    keypoints: dict[str, KeypointXY],
    category: GarmentCategory,
    measurements: GarmentFlatMeasurements,
    quality: GarmentQualityFlags,
    a4: A4DetectionResult,
) -> bytes:
    """Draw keypoints, measurement lines/labels, and a quality/confidence panel. Returns PNG bytes."""
    canvas = rectified_bgr.copy()
    h, w = canvas.shape[:2]

    lines = _TOP_LINES if category in (GarmentCategory.tops, GarmentCategory.kurtas) else _BOTTOM_LINES

    for name_a, name_b, label in lines:
        if name_a not in keypoints or name_b not in keypoints:
            continue
        pa = tuple(int(v) for v in to_canvas_px(keypoints[name_a], w, h))
        pb = tuple(int(v) for v in to_canvas_px(keypoints[name_b], w, h))
        cv2.line(canvas, pa, pb, LINE_COLOR, 2)
        text = _measurement_label_for(label, measurements)
        if text:
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            cv2.putText(canvas, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    for name, pt in keypoints.items():
        px, py = (int(v) for v in to_canvas_px(pt, w, h))
        cv2.circle(canvas, (px, py), 5, POINT_COLOR, -1)
        cv2.putText(canvas, name, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

    panel_lines = [
        f"A4 found: {a4.a4_found}  confidence: {a4.a4_confidence:.2f}  skew: {a4.skew_degrees:.1f} deg",
        f"Measurement confidence: {measurements.confidence:.2f}",
        "Quality: "
        + ", ".join(
            name
            for name, val in (
                ("folded/overlapping", quality.folded_or_overlapping),
                ("not flat", quality.not_flat),
                ("partly out of frame", quality.partly_out_of_frame),
            )
            if val
        )
        or "Quality: ok",
    ]
    panel_h = 24 * len(panel_lines) + 12
    cv2.rectangle(canvas, (0, 0), (min(w, 620), panel_h), BG_PANEL_COLOR, -1)
    for i, line in enumerate(panel_lines):
        cv2.putText(
            canvas, line, (8, 20 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA
        )

    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Failed to encode overlay image.")
    return encoded.tobytes()
