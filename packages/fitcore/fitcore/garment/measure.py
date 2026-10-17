"""Flat-lay photo (garment + A4 sheet) to centimetres per zone, with an overlay image showing what was measured."""

import math
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from fitcore.fit.config import get_config
from fitcore.garment.geometry import MeasureError, View, decode, detect_a4, rectify, segment_garment, snap_to_mask
from fitcore.garment.landmarks import (
    GARMENT_LENGTH_LANDMARKS, INTERIOR_OK, LandmarkFinder, LlmLandmarkFinder, MEASURABLE, PAIRS, required_landmarks,
)
from fitcore.schemas import Category, Measurement, MeasurementZone as Z

SNAP_MM = 25.0   # a landmark farther than this from the garment outline was not placed on the garment
PLAUSIBLE_CM = {Z.chest: (50, 200), Z.waist: (45, 200), Z.hip: (50, 210), Z.shoulder_width: (25, 75), Z.sleeve_length: (8, 90),
                Z.garment_length: (25, 160), Z.thigh: (30, 100), Z.inseam: (10, 110)}


class GarmentMeasurement(BaseModel):
    category: Category
    zones: dict[Z, Measurement]
    flags: list[str] = Field(default_factory=list)
    unsupported: list[Z] = Field(default_factory=list, description="Zones of this category that a front flat-lay cannot give")
    landmarks_px: dict[str, tuple[float, float]] = Field(default_factory=dict, description="Placed landmarks, in the rectified view's pixels")
    px_per_mm: float = 0.0
    a4_aspect_error: float = 0.0
    overlay_png: bytes = Field(default=b"", exclude=True)


def _draw_overlay(view: View, mask: np.ndarray, pts: dict[str, tuple[float, float]], zones: dict[Z, Measurement]) -> bytes:
    img = view.image.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (80, 200, 80), 2)
    cv2.polylines(img, [view.a4_rect.astype(np.int32)], True, (230, 120, 40), 3)
    cv2.putText(img, "A4", tuple(int(v) for v in view.a4_rect[0] + (8, 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 120, 40), 2, cv2.LINE_AA)

    def pt(n: str) -> tuple[int, int] | None:
        return (int(pts[n][0]), int(pts[n][1])) if n in pts else None

    def line(a: tuple[int, int] | None, b: tuple[int, int] | None, text: str, color: tuple[int, int, int]) -> None:
        if a and b:
            cv2.line(img, a, b, color, 3, cv2.LINE_AA)
            mid = ((a[0] + b[0]) // 2 + 6, (a[1] + b[1]) // 2 - 6)
            cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    palette = [(0, 0, 220), (200, 0, 200), (0, 140, 255), (200, 160, 0), (0, 180, 120), (140, 0, 0)]
    for i, (z, m) in enumerate(zones.items()):
        color = palette[i % len(palette)]
        if z in PAIRS:
            a, b, mult = PAIRS[z]
            line(pt(a), pt(b), f"{z.value.replace('_', ' ')} {m.value_cm:.1f}" + (" (x2)" if mult == 2 else ""), color)
        elif z == Z.garment_length and all(n in pts for n in GARMENT_LENGTH_LANDMARKS):
            top = tuple(int(v) for v in np.mean([pts["neck_left"], pts["neck_right"]], axis=0))
            bot = tuple(int(v) for v in np.mean([pts["hem_left"], pts["hem_right"]], axis=0))
            line(top, bot, f"length {m.value_cm:.1f}", color)
    for p in pts.values():
        cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), 9, (255, 255, 255), 2, cv2.LINE_AA)
    k = min(1.0, 1100 / max(img.shape[:2]))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


def measure_garment(image: bytes, category: Category, finder: LandmarkFinder | None = None, client: Any | None = None) -> GarmentMeasurement:
    """Measure one flat-lay photo. Raises MeasureError (no A4 sheet, no garment, a photo taken too steeply, ...)."""
    cfg = get_config()
    bgr = decode(image)
    a4 = detect_a4(bgr)
    view = rectify(bgr, a4)
    mask, mask_quality = segment_garment(view)
    category_zones = list(cfg.zones_by_category[category])
    zones_wanted = [z for z in category_zones if z in MEASURABLE]
    unsupported = [z for z in category_zones if z not in MEASURABLE]
    names = required_landmarks(category, zones_wanted)
    raw = (finder or LlmLandmarkFinder(client=client))(view, category, names)

    flags: list[str] = []
    if mask_quality < 0.9:
        flags.append("garment_touches_photo_edge")
    if a4.aspect_error > 0.06:
        flags.append("a4_perspective_strong")
    h, w = view.image.shape[:2]
    pts: dict[str, tuple[float, float]] = {}
    moved: dict[str, float] = {}
    snap_px = SNAP_MM * view.px_per_mm
    for n in names:
        p = raw.get(n)
        if p is None or not (0 <= p[0] < w and 0 <= p[1] < h):
            flags.append(f"landmark_missing:{n}")
            continue
        if n in INTERIOR_OK and mask[int(p[1]), int(p[0])] > 0:
            pts[n], moved[n] = p, 0.0
            continue
        sp, d = snap_to_mask(p, mask, snap_px)
        if d > snap_px:
            flags.append(f"landmark_off_garment:{n}")
            continue
        pts[n], moved[n] = sp, d
    if not pts:
        raise MeasureError("no_landmarks", "The landmarks could not be placed on this photo. Check the garment is fully in view and flat.")

    a4_q = max(0.5, 1.0 - a4.aspect_error * 4)
    zones: dict[Z, Measurement] = {}
    for z in zones_wanted:
        if z in PAIRS:
            a, b, mult = PAIRS[z]
            used = [a, b]
            value = view.cm(pts[a], pts[b]) * mult if a in pts and b in pts else None
        else:
            used = list(GARMENT_LENGTH_LANDMARKS)
            if all(n in pts for n in used):
                top = tuple(np.mean([pts["neck_left"], pts["neck_right"]], axis=0))
                bot = tuple(np.mean([pts["hem_left"], pts["hem_right"]], axis=0))
                value = view.cm(top, bot)  # type: ignore[arg-type]
            else:
                value = None
        if value is None:
            flags.append(f"zone_not_measured:{z.value}")
            continue
        lo, hi = PLAUSIBLE_CM[z]
        plausible = lo <= value <= hi
        if not plausible:
            flags.append(f"implausible:{z.value}")
        penalty = sum(min(moved.get(n, 0.0) / snap_px, 1.0) for n in used) / len(used) * 0.25
        conf = max(0.05, min(0.98, 0.95 * a4_q * mask_quality * (1 - penalty) * (1.0 if plausible else 0.4)))
        zones[z] = Measurement(value_cm=round(value, 1), confidence=round(conf, 2), source="flat_lay")
    return GarmentMeasurement(category=category, zones=zones, flags=flags, unsupported=unsupported, landmarks_px=pts, px_per_mm=view.px_per_mm,
                              a4_aspect_error=a4.aspect_error, overlay_png=_draw_overlay(view, mask, pts, zones))
