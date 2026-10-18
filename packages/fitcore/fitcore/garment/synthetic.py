"""Synthetic flat-lay photos with known measurements, for tests, samples and the CLI. SIMULATED: drawn, not photographed.

A garment polygon is built from target zone values (cm), placed on a brown surface next to a white A4 sheet in a millimetre
plane, then viewed through a random mild perspective. The truth (every landmark's pixel in the photo, and the zone values) comes back
with the image, so the measurer can be checked against it.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from fitcore.schemas import Category, MeasurementZone as Z

PLANE_W, PLANE_H = 1600.0, 1200.0     # mm; the same 4:3 as the photo, so pixels stay square
GEN_PX_PER_MM = 1.4
A4_POS = (1290.0, 220.0)               # top-left of the sheet in the plane
COLORS = [(52, 86, 168), (170, 60, 60), (60, 130, 80), (70, 70, 80), (200, 150, 40)]   # RGB garment colours
SURFACE = (168, 128, 92)


@dataclass
class SyntheticPhoto:
    jpeg: bytes
    landmarks: dict[str, tuple[float, float]]       # photo pixels
    zones_cm: dict[Z, float]
    size: tuple[int, int]
    seed: int
    sheet_corners: tuple[tuple[float, float], ...] = ()   # the A4 sheet's corners in the photo, tl tr br bl of the plane


def _top(category: Category, zones: dict[Z, float]) -> tuple[list[tuple[float, float]], dict[str, tuple[float, float]]]:
    cx = 600.0
    chest_w = zones[Z.chest] * 5.0
    sh_w = zones[Z.shoulder_width] * 10.0
    length = zones[Z.garment_length] * 10.0
    sleeve = zones.get(Z.sleeve_length, 0.0) * 10.0 if category != Category.dress else 0.0
    if category == Category.tshirt:
        sleeve = 200.0
    y0 = 120.0
    arm_y = y0 + max(170.0, 0.28 * length)
    tip_y = y0 + 28.0
    hem_w = zones[Z.hip] * 5.0 if Z.hip in zones else chest_w
    lm = {"neck_left": (cx - 90, y0), "neck_right": (cx + 90, y0),
          "shoulder_tip_left": (cx - sh_w / 2, tip_y), "shoulder_tip_right": (cx + sh_w / 2, tip_y),
          "armpit_left": (cx - chest_w / 2, arm_y), "armpit_right": (cx + chest_w / 2, arm_y),
          "hem_left": (cx - hem_w / 2, y0 + length), "hem_right": (cx + hem_w / 2, y0 + length)}
    side_l: list[tuple[float, float]] = []
    side_r: list[tuple[float, float]] = []
    if Z.waist in zones:
        lm["waist_left"], lm["waist_right"] = (cx - zones[Z.waist] * 2.5, y0 + 0.55 * length), (cx + zones[Z.waist] * 2.5, y0 + 0.55 * length)
        side_l.append(lm["waist_left"]); side_r.append(lm["waist_right"])
    if Z.hip in zones:
        lm["hip_left"], lm["hip_right"] = (cx - zones[Z.hip] * 2.5, y0 + 0.8 * length), (cx + zones[Z.hip] * 2.5, y0 + 0.8 * length)
        side_l.append(lm["hip_left"]); side_r.append(lm["hip_right"])
    a = math.radians(62 if category == Category.tshirt else 24)    # short sleeves lie flatter, so the armpit stays a real notch

    def sleeve_pts(sign: int) -> list[tuple[float, float]]:
        tip = lm["shoulder_tip_right" if sign > 0 else "shoulder_tip_left"]
        if sleeve <= 0:
            return []
        c = (tip[0] + sign * sleeve * math.sin(a), tip[1] + sleeve * math.cos(a))
        perp = (math.cos(a) * sign, -math.sin(a))
        outer = (c[0] + 48 * perp[0], c[1] + 48 * perp[1])
        inner = (c[0] - 48 * perp[0], c[1] - 48 * perp[1])
        lm["sleeve_cuff_right" if sign > 0 else "sleeve_cuff_left"] = c
        return [outer, inner] if sign > 0 else [inner, outer]

    right = sleeve_pts(1)
    left = sleeve_pts(-1)
    poly = [lm["neck_left"], (cx - 60, y0 + 55), (cx, y0 + 80), (cx + 60, y0 + 55), lm["neck_right"], lm["shoulder_tip_right"], *right,
            lm["armpit_right"], *side_r, lm["hem_right"], lm["hem_left"], *reversed(side_l), lm["armpit_left"], *left, lm["shoulder_tip_left"]]
    return poly, lm


def _bottom(zones: dict[Z, float]) -> tuple[list[tuple[float, float]], dict[str, tuple[float, float]]]:
    cx, y0, rise = 600.0, 50.0, 250.0
    waist_w, hip_w, tw = zones[Z.waist] * 5.0, zones[Z.hip] * 5.0, zones[Z.thigh] * 5.0
    inseam = zones[Z.inseam] * 10.0
    gap = 60.0
    hem_y = y0 + rise + math.sqrt(max(inseam ** 2 - gap ** 2, 1.0))
    leg_hem = 0.62 * tw
    lm = {"waist_left": (cx - waist_w / 2, y0), "waist_right": (cx + waist_w / 2, y0), "hip_left": (cx - hip_w / 2, y0 + 180),
          "hip_right": (cx + hip_w / 2, y0 + 180), "thigh_outer_left": (cx - tw, y0 + rise), "crotch": (cx, y0 + rise),
          "leg_hem_inner_left": (cx - gap, hem_y)}
    poly = [lm["waist_left"], lm["waist_right"], lm["hip_right"], (cx + tw, y0 + rise), (cx + gap + leg_hem, hem_y), (cx + gap, hem_y), lm["crotch"],
            lm["leg_hem_inner_left"], (cx - gap - leg_hem, hem_y), lm["thigh_outer_left"], lm["hip_left"]]
    return poly, lm


def render_flatlay(category: Category, zones_cm: dict[Z, float], seed: int = 0, *, tilt: float = 0.05, a4_landscape: bool = False,
                   photo_size: tuple[int, int] = (1600, 1200), noise: float = 3.0, color: tuple[int, int, int] | None = None,
                   scale: float = 1.0, with_a4: bool = True, with_garment: bool = True) -> SyntheticPhoto:
    """Draw a flat-lay and photograph it. `scale` < 1 moves the camera back (the garment gets smaller in the photo)."""
    rng = np.random.default_rng(seed)
    bottoms = category in (Category.trousers, Category.jeans)
    poly, lm = _bottom(zones_cm) if bottoms else _top(category, zones_cm)
    g = GEN_PX_PER_MM
    ss = 2
    cw, ch = int(PLANE_W * g), int(PLANE_H * g)
    canvas = np.full((ch * ss, cw * ss, 3), SURFACE[::-1], np.uint8)
    gradient = np.linspace(-14, 14, cw * ss, dtype=np.float32)[None, :, None]
    canvas = np.clip(canvas.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    garment = color or COLORS[seed % len(COLORS)]
    pts = (np.array(poly, np.float32) * g * ss).astype(np.int32)
    if with_garment:
        cv2.fillPoly(canvas, [pts], garment[::-1], lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, tuple(int(c * 0.7) for c in garment[::-1]), 3, cv2.LINE_AA)
    w_mm, h_mm = (297.0, 210.0) if a4_landscape else (210.0, 297.0)
    x0, y0 = A4_POS if not a4_landscape else (A4_POS[0] - 87, A4_POS[1] + 40)
    sheet = np.array([[x0, y0], [x0 + w_mm, y0 + h_mm]], np.float32) * g * ss
    if with_a4:
        cv2.rectangle(canvas, tuple(int(v) for v in sheet[0]), tuple(int(v) for v in sheet[1]), (250, 250, 250), cv2.FILLED)
    canvas = cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)

    pw, ph = photo_size
    base = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    shrink = (1 - scale) / 2
    dst = np.array([[pw * shrink, ph * shrink], [pw * (1 - shrink), ph * shrink], [pw * (1 - shrink), ph * (1 - shrink)], [pw * shrink, ph * (1 - shrink)]], np.float32)
    dst += rng.uniform(-tilt, tilt, (4, 2)).astype(np.float32) * np.array([pw, ph], np.float32)
    m = cv2.getPerspectiveTransform(np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32), dst)
    photo = cv2.warpPerspective(canvas, m, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:ph, 0:pw]
    vignette = 1 - 0.18 * (((xx - pw / 2) / (pw / 2)) ** 2 + ((yy - ph / 2) / (ph / 2)) ** 2)
    photo = np.clip(photo.astype(np.float32) * vignette[..., None] + rng.normal(0, noise, photo.shape), 0, 255).astype(np.uint8)
    landmarks_canvas = np.array([[v[0] * g, v[1] * g] for v in lm.values()], np.float32).reshape(1, -1, 2)
    landmarks_photo = cv2.perspectiveTransform(landmarks_canvas, m)[0]
    truth = {n: (float(p[0]), float(p[1])) for n, p in zip(lm, landmarks_photo)}
    sc = np.array([[x0, y0], [x0 + w_mm, y0], [x0 + w_mm, y0 + h_mm], [x0, y0 + h_mm]], np.float32) * g
    sheet_photo = cv2.perspectiveTransform(sc.reshape(1, 4, 2), m)[0]
    jpg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    return SyntheticPhoto(jpeg=jpg, landmarks=truth, zones_cm={z: v for z, v in zones_cm.items()}, size=photo_size, seed=seed,
                          sheet_corners=tuple((float(p[0]), float(p[1])) for p in sheet_photo))
