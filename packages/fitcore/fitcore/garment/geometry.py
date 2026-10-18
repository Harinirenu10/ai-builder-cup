"""Scale and perspective from an A4 sheet, and the garment outline. Deterministic: no model is involved here.

The sheet (210 x 297 mm) lies in the same plane as the garment. Its four corners give a homography from the photo to a
millimetre plane, so any two points on the garment can be measured in cm whatever the camera angle or distance.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

A4_SHORT_MM, A4_LONG_MM = 210.0, 297.0
A4_RATIO = A4_LONG_MM / A4_SHORT_MM          # 1.414
MAX_RECT_PX = 1600                            # longest side of the rectified view
MAX_EXTENT_MM = 2500.0                        # a flat-lay wider than this means the perspective is extreme


class MeasureError(Exception):
    """A photo that cannot be measured, with a code the UI can act on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class A4Detection:
    corners: np.ndarray                      # 4x2 float, photo pixels, ordered tl, tr, br, bl
    score: float
    area_frac: float
    aspect_error: float                      # relative error of the sheet's side ratio against 1.414


@dataclass
class View:
    """The photo seen from straight above, aligned to the sheet, in a known scale."""
    image: np.ndarray                        # BGR
    px_per_mm: float
    H: np.ndarray                            # photo pixel -> rectified pixel
    a4_rect: np.ndarray                      # sheet corners in rectified pixels
    valid: np.ndarray                        # 255 where the rectified pixel came from the photo
    a4: A4Detection = field(repr=False, default=None)  # type: ignore[assignment]

    def to_rect(self, pt: tuple[float, float]) -> tuple[float, float]:
        v = self.H @ np.array([pt[0], pt[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def cm(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) / self.px_per_mm / 10.0


def decode(data: bytes) -> np.ndarray:
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    except cv2.error:
        img = None
    if img is None or min(img.shape[:2]) < 200:
        raise MeasureError("bad_image", "The file is not a usable photo (too small or not an image).")
    return img


def _order(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return box.astype(np.float32) if cv2.contourArea(hull) / max(cv2.contourArea(box.astype(np.float32)), 1) > 0.92 else None


def _refine_corners(gray: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Re-find the sheet at full resolution with the threshold halfway between the sheet and the surface around it.

    A fixed brightness threshold bites into a sheet that is darker toward its edges (shading); the half-way level does not.
    """
    h, w = gray.shape
    q = quad.astype(np.float32)
    centre = q.mean(axis=0)

    def grown(f: float) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, (centre + (q - centre) * f).astype(np.int32), 255)
        return m

    inside = gray[grown(0.85) > 0]
    ring = (grown(1.25) > 0) & (grown(1.12) == 0)
    if inside.size < 500 or ring.sum() < 500:
        return None
    t = (float(np.median(inside)) + float(np.median(gray[ring]))) / 2
    window = grown(1.3)
    binary = ((cv2.GaussianBlur(gray, (3, 3), 0) > t) & (window > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.7 * cv2.contourArea(q):
        return None
    fine = _quad(biggest)
    if fine is None:
        return None
    ordered = _order(fine)
    fitted = _line_corners(biggest.reshape(-1, 2).astype(np.float32), ordered)
    start = ordered if fitted is None else fitted
    sub = _subpixel_corners(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32), start)
    return start if sub is None else sub


def _subpixel_corners(gray: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Each edge located to a fraction of a pixel: where brightness crosses half-way between sheet and surface, along many
    short profiles across the edge, then a line through the crossings."""
    h, w = gray.shape
    centre = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i].astype(np.float64), quad[(i + 1) % 4].astype(np.float64)
        d = b - a
        length = float(np.linalg.norm(d))
        u = d / length
        n = np.array([u[1], -u[0]])
        if np.dot(n, (a + b) / 2 - centre) < 0:
            n = -n                                              # n points away from the sheet
        offsets = np.arange(-9.0, 9.01, 0.25)
        crossings = []
        for t in np.linspace(0.15, 0.85, 45):
            c = a + t * d
            xs = c[0] + offsets * n[0]
            ys = c[1] + offsets * n[1]
            if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
                continue
            x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
            fx, fy = xs - x0, ys - y0
            prof = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x0 + 1] * fx * (1 - fy) + gray[y0 + 1, x0] * (1 - fx) * fy + gray[y0 + 1, x0 + 1] * fx * fy)
            inside, outside = float(prof[offsets < -4].mean()), float(prof[offsets > 4].mean())
            if inside - outside < 25:
                continue
            level = (inside + outside) / 2
            below = np.nonzero(prof < level)[0]
            below = below[below > 0]
            if len(below) == 0:
                continue
            k = below[0]
            if prof[k - 1] == prof[k]:
                continue
            f = (prof[k - 1] - level) / (prof[k - 1] - prof[k])
            crossings.append(c + (offsets[k - 1] + f * 0.25) * n)
        if len(crossings) < 15:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(crossings, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        m = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(m)) < 1e-6:
            return None
        corners.append(p1 + np.linalg.solve(m, p2 - p1)[0] * d1)
    out = np.array(corners, np.float32)
    return out if np.abs(out - quad).max() < 6.0 else None


def _line_corners(points: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Corners as the intersections of four lines fitted along the sheet's edges: sub-pixel, because each line uses hundreds of points.

    The sheet is small next to the garment, so a corner error of a pixel becomes a scale error of several percent at the garment.
    """
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 20:
            return None
        u = d / length
        rel = points - a
        along = rel @ u
        across = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        keep = points[(across < 4.0) & (along > 0.12 * length) & (along < 0.88 * length)]
        if len(keep) < 20:
            return None
        vx, vy, x0, y0 = cv2.fitLine(keep, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]       # corner i is where edge i-1 meets edge i
        m = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        if abs(np.linalg.det(m)) < 1e-6:
            return None
        t = np.linalg.solve(m, p2 - p1)[0]
        corners.append(p1 + t * d1)
    out = np.array(corners, np.float32)
    return out if np.abs(out - quad).max() < 6.0 else None   # a fit that wandered is worse than the polygon corners


def detect_a4(bgr: np.ndarray) -> A4Detection:
    """Find the white A4 sheet: the bright four-sided shape whose sides are in the 1.414 ratio."""
    h, w = bgr.shape[:2]
    k = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    best: tuple[float, np.ndarray, float, float] | None = None
    area_img = small.shape[0] * small.shape[1]
    for thr in sorted({max(otsu, 150), 170, 190, 210}):
        binary = cv2.morphologyEx((gray > thr).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            frac = cv2.contourArea(c) / area_img
            if not 0.01 <= frac <= 0.6:
                continue
            quad = _quad(c)
            if quad is None:
                continue
            q = _order(quad)
            a = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            b = (np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])) / 2
            if min(a, b) < 20:
                continue
            ratio = max(a, b) / min(a, b)
            err = abs(ratio - A4_RATIO) / A4_RATIO
            fill = cv2.contourArea(c) / max(cv2.contourArea(q), 1.0)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
            brightness = float(gray[mask > 0].mean()) / 255.0
            score = fill * 0.5 + brightness * 0.3 + max(0.0, 1 - err * 5) * 0.4 - 0.2
            if err < 0.18 and brightness > 0.55 and fill > 0.85 and (best is None or score > best[0]):
                best = (score, q / k, frac, err)
    if best is not None:
        refined = _refine_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), best[1])
        if refined is not None:
            best = (best[0], refined, best[2], best[3])
    if best is None:
        raise MeasureError("no_a4", "We could not find the A4 sheet. Lay a plain white A4 sheet flat next to the garment with all four corners in view.")
    return A4Detection(corners=best[1], score=round(best[0], 3), area_frac=round(best[2], 3), aspect_error=round(best[3], 3))


def rectify(bgr: np.ndarray, a4: A4Detection) -> View:
    """Warp the photo onto the millimetre plane of the sheet, at up to 3 px per mm."""
    q = a4.corners.astype(np.float32)
    width_px = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height_px = (np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])) / 2
    w_mm, h_mm = (A4_LONG_MM, A4_SHORT_MM) if width_px >= height_px else (A4_SHORT_MM, A4_LONG_MM)
    h_mm_plane = cv2.getPerspectiveTransform(q, np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], np.float32))
    h, w = bgr.shape[:2]
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32), h_mm_plane)[0]
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    if not np.all(np.isfinite(corners)) or (hi - lo).max() > MAX_EXTENT_MM:
        raise MeasureError("perspective_extreme", "The photo is taken at too steep an angle. Shoot from directly above.")
    s = min(3.0, MAX_RECT_PX / float((hi - lo).max()))
    t = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1]], np.float64)
    hmat = t @ h_mm_plane
    size = (int(math.ceil((hi[0] - lo[0]) * s)), int(math.ceil((hi[1] - lo[1]) * s)))
    image = cv2.warpPerspective(bgr, hmat, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), hmat, size, flags=cv2.INTER_NEAREST)
    a4_rect = cv2.perspectiveTransform(q.reshape(1, 4, 2), hmat)[0]
    return View(image=image, px_per_mm=float(s), H=hmat, a4_rect=a4_rect, valid=valid, a4=a4)


def _background_surface(lab: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """A smooth (quadratic) model of the surface colour, fitted on the pixels in `fit`. Copes with shading and vignetting."""
    h, w = fit.shape
    ys, xs = np.nonzero(fit)
    if len(xs) > 30000:
        pick = np.random.default_rng(0).choice(len(xs), 30000, replace=False)
        ys, xs = ys[pick], xs[pick]

    def basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    coef = np.linalg.lstsq(basis(xs / w, ys / h), lab[ys, xs], rcond=None)[0]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (basis(gx / w, gy / h) @ coef).astype(np.float32)


def segment_garment(view: View) -> tuple[np.ndarray, float]:
    """Garment mask (255 inside) from colour distance to the surface it lies on, and a 0-1 quality of the mask."""
    img, valid = view.image, view.valid
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sheet = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(sheet, view.a4_rect.astype(np.int32), 255)
    sheet = cv2.dilate(sheet, np.ones((9, 9), np.uint8))
    usable = cv2.erode(valid, np.ones((7, 7), np.uint8))
    usable[sheet > 0] = 0
    use = usable > 0
    # First guess at the surface colour: the usable pixels nearest the edge of the photo.
    inner = cv2.erode(usable, np.ones((int(60 * view.px_per_mm) | 1,) * 2, np.uint8))
    border = use & (inner == 0)
    if border.sum() < 200:
        border = use
    surface: np.ndarray = np.median(lab[border], axis=0)[None, None, :]
    mask = np.zeros(valid.shape, np.uint8)
    k = max(3, int(2.0 * view.px_per_mm) | 1)     # small: a bigger one fills the narrow notches (armpit, crotch) the landmarks sit in
    for step in range(3):          # fit the surface on what is not garment yet, then look again
        dist = np.linalg.norm(lab - surface, axis=2)
        dist[~use] = 0
        d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
        thr = max(cv2.threshold(d8[use].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 4, 12.0)
        mask = ((dist > thr) & use).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(3, k // 2) | 1,) * 2, np.uint8))
        grown = cv2.dilate(mask, np.ones((int(20 * view.px_per_mm) | 1,) * 2, np.uint8))
        fit = use & (grown == 0)
        if step < 2 and fit.sum() > 2000:
            surface = _background_surface(lab, fit)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        raise MeasureError("no_garment", "We could not see the garment against the surface. Use a surface that contrasts with its colour.")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    area_cm2 = float((filled > 0).sum()) / (view.px_per_mm ** 2) / 100.0
    if area_cm2 < 150:
        raise MeasureError("no_garment", "The garment looks too small in the photo. Move closer, keeping the sheet in view.")
    touches_edge = bool(filled[0, :].any() or filled[-1, :].any() or filled[:, 0].any() or filled[:, -1].any())
    return filled, 0.95 - (0.25 if touches_edge else 0.0)


def snap_to_mask(pt: tuple[float, float], mask: np.ndarray, radius_px: float) -> tuple[tuple[float, float], float]:
    """Move a landmark to the nearest point on the garment's outline if one is within radius_px. Returns (point, moved_px)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return pt, 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    d = np.linalg.norm(pts - np.array(pt, np.float32), axis=1)
    i = int(np.argmin(d))
    if d[i] <= radius_px:
        return (float(pts[i][0]), float(pts[i][1])), float(d[i])
    return pt, float(d[i])
