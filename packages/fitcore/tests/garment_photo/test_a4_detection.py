from __future__ import annotations

import cv2
import numpy as np

from fitcore.garment_photo.a4_detection import A4_RATIO, detect_a4_sheet
from fitcore.garment_photo.schemas import GarmentCategory
from tests.garment_photo.synth_helpers import build_synthetic_scene


def test_detects_a4_sheet_under_perspective_warp():
    image_bytes, _, _ = build_synthetic_scene(GarmentCategory.tops, warp_strength=0.03)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result, corners = detect_a4_sheet(img)

    assert result.a4_found is True
    assert result.a4_confidence >= 0.5
    assert corners is not None
    assert corners.shape == (4, 2)


def test_corner_ratio_is_close_to_a4():
    image_bytes, _, _ = build_synthetic_scene(GarmentCategory.bottoms, warp_strength=0.02)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result, corners = detect_a4_sheet(img)
    assert corners is not None

    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    left = np.linalg.norm(bl - tl)
    long_side = max(top, left)
    short_side = min(top, left)
    assert abs(long_side / short_side - A4_RATIO) < 0.08


def test_no_a4_present_returns_not_found():
    img = np.full((800, 600, 3), 60, dtype=np.uint8)
    cv2.circle(img, (300, 400), 100, (200, 200, 200), -1)

    result, corners = detect_a4_sheet(img)

    assert result.a4_found is False
    assert result.a4_confidence == 0.0
    assert corners is None


def test_rejects_non_a4_ratio_rectangle():
    """A perfect square (1:1) should not be mistaken for an A4 sheet (1:1.414)."""
    img = np.full((800, 800, 3), 50, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (600, 600), (255, 255, 255), -1)

    result, _ = detect_a4_sheet(img)

    assert result.a4_found is False
