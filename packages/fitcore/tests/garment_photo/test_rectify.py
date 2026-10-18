from __future__ import annotations

import cv2
import numpy as np

from fitcore.garment_photo.rectify import (
    A4_HEIGHT_PX,
    A4_WIDTH_PX,
    compute_rectification_homography,
    rectify_image,
)


def test_canvas_contains_the_whole_transformed_source_image():
    # A4 drawn in a skewed source quad within a larger frame; rectification must
    # size the canvas to contain the entire warped source image, not clip it.
    src_img = np.full((1000, 1200, 3), 30, dtype=np.uint8)
    skewed_corners = np.array(
        [[100, 120], [700, 90], [720, 760], [110, 740]], dtype=np.float32
    )

    rectified = rectify_image(src_img, skewed_corners)
    canvas_h, canvas_w = rectified.shape[:2]

    homography, computed_w, computed_h = compute_rectification_homography(
        skewed_corners, image_shape=src_img.shape[:2]
    )
    assert canvas_w == computed_w
    assert canvas_h == computed_h

    # The whole 1200x1000 source frame, once transformed, must land inside the canvas.
    src_corners = np.array(
        [[0, 0], [1200, 0], [1200, 1000], [0, 1000]], dtype=np.float32
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(src_corners, homography).reshape(-1, 2)
    assert (transformed[:, 0] >= -1).all() and (transformed[:, 0] <= canvas_w + 1).all()
    assert (transformed[:, 1] >= -1).all() and (transformed[:, 1] <= canvas_h + 1).all()


def test_rectified_a4_region_has_correct_physical_aspect_ratio():
    src_img = np.full((1000, 1200, 3), 30, dtype=np.uint8)
    skewed_corners = np.array(
        [[100, 120], [700, 90], [720, 760], [110, 740]], dtype=np.float32
    )
    homography, _, _ = compute_rectification_homography(skewed_corners, image_shape=src_img.shape[:2])

    a4_src = skewed_corners.reshape(-1, 1, 2)
    a4_canvas = cv2.perspectiveTransform(a4_src, homography).reshape(-1, 2)
    tl, tr, br, bl = a4_canvas
    width_px = np.linalg.norm(tr - tl)
    height_px = np.linalg.norm(bl - tl)
    assert abs(width_px - A4_WIDTH_PX) < 1.0
    assert abs(height_px - A4_HEIGHT_PX) < 1.0
