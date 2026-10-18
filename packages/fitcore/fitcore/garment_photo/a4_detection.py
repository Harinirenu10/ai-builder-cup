"""Deterministic A4 reference sheet detection via OpenCV contour analysis.

Pipeline: grayscale -> blur -> adaptive threshold & Canny edge candidates ->
contours -> convex quadrilateral filtering -> A4 side-ratio check (1.414 +/- 0.08).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from fitcore.garment_photo.schemas import A4DetectionResult

A4_RATIO = 297.0 / 210.0  # ~1.41421
A4_RATIO_TOLERANCE = 0.08
MIN_AREA_FRACTION = 0.01  # candidate quad must cover at least 1% of the frame


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmax(diff)]
    bottom_left = pts[np.argmin(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _side_lengths(corners: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = corners
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(bl - br))
    left = float(np.linalg.norm(tl - bl))
    return top, right, bottom, left


def _quad_candidates_from_contours(contours: list[np.ndarray], image_area: float) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * MIN_AREA_FRACTION:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        candidates.append(approx)
    return candidates


def detect_a4_sheet(image_bgr: np.ndarray) -> tuple[A4DetectionResult, np.ndarray | None]:
    """Locate the A4 sheet in a BGR image.

    Returns:
        (A4DetectionResult, ordered_corners) where ordered_corners is a (4, 2)
        float32 array [top_left, top_right, bottom_right, bottom_left] in the
        source image's pixel coordinates, or None if no sheet was found.
    """
    h, w = image_bgr.shape[:2]
    image_area = float(h * w)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
    )
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    all_candidates: list[np.ndarray] = []
    for src in (adaptive, edges):
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_candidates.extend(_quad_candidates_from_contours(contours, image_area))

    best_corners: np.ndarray | None = None
    best_confidence = 0.0
    best_area = 0.0

    for approx in all_candidates:
        corners = _order_corners(approx)
        top, right, bottom, left = _side_lengths(corners)
        if min(top, right, bottom, left) <= 1e-3:
            continue

        width_avg = (top + bottom) / 2.0
        height_avg = (left + right) / 2.0
        long_side = max(width_avg, height_avg)
        short_side = min(width_avg, height_avg)
        ratio = long_side / short_side

        ratio_error = abs(ratio - A4_RATIO)
        if ratio_error > A4_RATIO_TOLERANCE:
            continue

        # Confidence blends: ratio closeness, side-length symmetry (rectangularity),
        # and relative contour area (favors the larger, cleaner detection).
        ratio_score = max(0.0, 1.0 - ratio_error / A4_RATIO_TOLERANCE)
        symmetry_score = max(
            0.0,
            1.0 - (abs(top - bottom) / width_avg + abs(left - right) / height_avg) / 2.0,
        )
        area = cv2.contourArea(corners.reshape(4, 1, 2).astype(np.float32))
        area_score = min(1.0, area / image_area / 0.5)
        confidence = 0.5 * ratio_score + 0.3 * symmetry_score + 0.2 * area_score

        if confidence > best_confidence or (
            math.isclose(confidence, best_confidence) and area > best_area
        ):
            best_confidence = confidence
            best_corners = corners
            best_area = area

    if best_corners is None:
        return A4DetectionResult(a4_found=False, a4_confidence=0.0, skew_degrees=0.0), None

    tl, tr = best_corners[0], best_corners[1]
    dx, dy = float(tr[0] - tl[0]), float(tr[1] - tl[1])
    skew_degrees = math.degrees(math.atan2(dy, dx))

    result = A4DetectionResult(
        a4_found=True,
        a4_confidence=round(float(best_confidence), 3),
        skew_degrees=round(skew_degrees, 2),
    )
    return result, best_corners
