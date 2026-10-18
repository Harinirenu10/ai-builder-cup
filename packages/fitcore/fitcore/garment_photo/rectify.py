"""Homography rectification of a garment flat-lay photo using the detected A4 sheet.

The A4 sheet corners are mapped to its true physical size (210 x 297 mm) at a
fixed scale of 2 px/mm. That mapping (a homography, since the camera is not
guaranteed to be perfectly overhead) is then applied to the *entire* source
image, not just the A4 region, so a garment that extends beyond the sheet's
own footprint is still captured. The output canvas is sized to the bounding
box of the whole transformed image (plus the A4 rect itself), so nothing is
clipped regardless of how the garment is framed relative to the sheet.
"""

from __future__ import annotations

import cv2
import numpy as np

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PX_PER_MM = 2.0
CM_PER_PX = 1.0 / (PX_PER_MM * 10.0)  # 0.05 cm per pixel on the rectified canvas

A4_WIDTH_PX = A4_WIDTH_MM * PX_PER_MM  # 420
A4_HEIGHT_PX = A4_HEIGHT_MM * PX_PER_MM  # 594

# Safety margin added around the union of the A4 rect and the transformed image,
# in case of minor extrapolation error right at the canvas edge.
CANVAS_EDGE_PADDING_PX = 20.0

# Hard ceiling on canvas dimensions (pixels) to avoid pathological memory use on a
# badly-detected A4 quad (e.g. a near-degenerate homography blowing up the bbox).
MAX_CANVAS_DIM_PX = 8000


def compute_rectification_homography(
    a4_corners: np.ndarray, image_shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, int, int]:
    """Build the homography mapping the source image's A4 corners to a fixed-scale canvas.

    Args:
        a4_corners: (4, 2) float32 array [top_left, top_right, bottom_right, bottom_left]
            in source image pixel coordinates.
        image_shape: Optional (height, width) of the source image. When provided, the
            canvas is sized to contain the full transformed image (not just the A4 rect),
            so a garment positioned anywhere in frame is never clipped.

    Returns:
        (homography_3x3, canvas_width_px, canvas_height_px)
    """
    a4_dst = np.array(
        [[0.0, 0.0], [A4_WIDTH_PX, 0.0], [A4_WIDTH_PX, A4_HEIGHT_PX], [0.0, A4_HEIGHT_PX]],
        dtype=np.float32,
    )
    base_h = cv2.getPerspectiveTransform(a4_corners.astype(np.float32), a4_dst)

    # Union of the A4 rect itself and the whole transformed source image's bounding box.
    corners_to_bound = [a4_dst]
    if image_shape is not None:
        src_h, src_w = image_shape
        img_corners = np.array(
            [[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(img_corners, base_h).reshape(-1, 2)
        corners_to_bound.append(transformed)

    all_pts = np.vstack(corners_to_bound)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    pad = CANVAS_EDGE_PADDING_PX
    offset_x = -min_x + pad
    offset_y = -min_y + pad
    canvas_w = int(min(MAX_CANVAS_DIM_PX, round(max_x - min_x + 2 * pad)))
    canvas_h = int(min(MAX_CANVAS_DIM_PX, round(max_y - min_y + 2 * pad)))

    translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
    homography = translation @ base_h
    return homography.astype(np.float32), canvas_w, canvas_h


def rectify_image(image_bgr: np.ndarray, a4_corners: np.ndarray) -> np.ndarray:
    """Warp the full source image onto the fixed-scale rectified canvas."""
    homography, canvas_w, canvas_h = compute_rectification_homography(
        a4_corners, image_shape=image_bgr.shape[:2]
    )
    return cv2.warpPerspective(
        image_bgr, homography, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
