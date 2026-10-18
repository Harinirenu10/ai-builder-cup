"""Synthetic test fixtures: A4 sheet + garment keypoints at known real-world mm offsets,
with an optional perspective warp applied to the whole scene, for garment_photo tests.

Not a test module itself (no test_ prefix) so pytest does not collect it directly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fitcore.garment_photo.rectify import compute_rectification_homography
from fitcore.garment_photo.schemas import GarmentCategory, KeypointXY

SRC_PX_PER_MM = 3.0
MARGIN_MM = 150.0
A4_W_MM, A4_H_MM = 210.0, 297.0

# Keypoint offsets (dx_mm, dy_mm) relative to a scene origin placed to the right of
# the A4 sheet. Distances are arbitrary but fixed, so expected cm is computed directly
# from these offsets in tests (not hardcoded), guaranteeing internal consistency.
TOP_KEYPOINT_OFFSETS_MM: dict[str, tuple[float, float]] = {
    "left_armpit": (0.0, 0.0),
    "right_armpit": (500.0, 0.0),
    "left_shoulder_seam": (40.0, -210.0),
    "right_shoulder_seam": (480.0, -195.0),
    "back_neck_point": (260.0, -280.0),
    "left_cuff_outer": (20.0, -760.0),
    "left_cuff_inner": (80.0, -700.0),
    "hem_left": (10.0, 640.0),
    "hem_right": (545.0, 610.0),
    "hem_center": (270.0, 630.0),
}

BOTTOM_KEYPOINT_OFFSETS_MM: dict[str, tuple[float, float]] = {
    "waist_left": (0.0, 0.0),
    "waist_right": (410.0, -10.0),
    "back_waist_center": (205.0, -30.0),
    "crotch": (205.0, 260.0),
    "hip_left": (-25.0, 215.0),
    "hip_right": (430.0, 205.0),
    "left_hem_inner": (160.0, 1080.0),
    "left_hem_outer": (-10.0, 1110.0),
}

CATEGORY_OFFSETS: dict[GarmentCategory, dict[str, tuple[float, float]]] = {
    GarmentCategory.tops: TOP_KEYPOINT_OFFSETS_MM,
    GarmentCategory.kurtas: TOP_KEYPOINT_OFFSETS_MM,
    GarmentCategory.bottoms: BOTTOM_KEYPOINT_OFFSETS_MM,
}

# Scene origin (mm), chosen so every keypoint offset above stays within the canvas.
ORIGIN_MM = (MARGIN_MM + A4_W_MM + 120.0, 1000.0)
CANVAS_W_MM = ORIGIN_MM[0] + 700.0
CANVAS_H_MM = 2100.0


def expected_cm_distance(offsets_mm: dict[str, tuple[float, float]], a: str, b: str) -> float:
    ax, ay = offsets_mm[a]
    bx, by = offsets_mm[b]
    return math.hypot(bx - ax, by - ay) / 10.0


def build_synthetic_scene(
    category: GarmentCategory, warp_strength: float = 0.03
) -> tuple[bytes, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Build a synthetic warped source photo with an A4 sheet.

    Returns:
        (jpeg_bytes, keypoint_px_in_warped_image, keypoint_offsets_mm)
        keypoint_px_in_warped_image maps keypoint name -> (x, y) pixel position in the
        *final warped* image (ground truth, independent of any detection algorithm).
    """
    px = SRC_PX_PER_MM
    w = int(CANVAS_W_MM * px)
    h = int(CANVAS_H_MM * px)
    img = np.full((h, w, 3), 40, dtype=np.uint8)

    a4_tl = (MARGIN_MM * px, MARGIN_MM * px)
    a4_tr = ((MARGIN_MM + A4_W_MM) * px, MARGIN_MM * px)
    a4_br = ((MARGIN_MM + A4_W_MM) * px, (MARGIN_MM + A4_H_MM) * px)
    a4_bl = (MARGIN_MM * px, (MARGIN_MM + A4_H_MM) * px)
    a4_corners = np.array([a4_tl, a4_tr, a4_br, a4_bl], dtype=np.float32)
    cv2.fillConvexPoly(img, a4_corners.astype(np.int32), (255, 255, 255))

    offsets_mm = CATEGORY_OFFSETS[category]
    keypoints_px_unwarped: dict[str, np.ndarray] = {}
    for name, (dx, dy) in offsets_mm.items():
        x_px = (ORIGIN_MM[0] + dx) * px
        y_px = (ORIGIN_MM[1] + dy) * px
        keypoints_px_unwarped[name] = np.array([x_px, y_px], dtype=np.float32)

    src_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_quad = src_quad + np.array(
        [
            [w * warp_strength, h * warp_strength * 0.5],
            [-w * warp_strength * 0.7, h * warp_strength],
            [-w * warp_strength, -h * warp_strength * 0.6],
            [w * warp_strength * 0.6, -h * warp_strength],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(img, homography, (w, h), borderValue=(40, 40, 40))

    pts = np.array(list(keypoints_px_unwarped.values()), dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    keypoints_px_warped = {
        name: (float(warped_pts[i][0]), float(warped_pts[i][1]))
        for i, name in enumerate(keypoints_px_unwarped.keys())
    }

    ok, encoded = cv2.imencode(".jpg", warped)
    assert ok
    return encoded.tobytes(), keypoints_px_warped, offsets_mm


def keypoints_to_normalized(
    keypoints_px_warped: dict[str, tuple[float, float]],
    a4_corners_detected: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    image_shape: tuple[int, int],
) -> dict[str, KeypointXY]:
    """Project warped-image pixel keypoints onto the rectified canvas (via the same
    homography the real pipeline would compute from the detected A4 corners) and
    normalize to the 0-1000 coordinate space.
    """
    homography, _, _ = compute_rectification_homography(a4_corners_detected, image_shape=image_shape)
    names = list(keypoints_px_warped.keys())
    pts = np.array([keypoints_px_warped[n] for n in names], dtype=np.float32).reshape(-1, 1, 2)
    canvas_pts = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)

    out: dict[str, KeypointXY] = {}
    for i, name in enumerate(names):
        x_px, y_px = canvas_pts[i]
        norm_x = min(1000.0, max(0.0, float(x_px) / canvas_w * 1000.0))
        norm_y = min(1000.0, max(0.0, float(y_px) / canvas_h * 1000.0))
        out[name] = KeypointXY(x=norm_x, y=norm_y)
    return out
