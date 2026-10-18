"""Photo-based body measurements and Ramanujan circumference calculations.

Extracts all 12 MeasurementZone values in centimeters with confidence scores:
1. Chest circumference
2. Waist circumference
3. Hip circumference
4. Thigh circumference
5. Neck circumference
6. Shoulder width
7. Sleeve length
8. Inseam
9. Torso length
10. Back rise
11. Upper back
12. Garment length

Uses:
- Known user height to convert pixel spans into metric centimeters.
- Ramanujan's second ellipse perimeter approximation for circumferences:
    P = pi * (a + b) * (1 + 3h / (10 + sqrt(4 - 3h))) * k_zone
    where a = front_width / 2, b = side_depth / 2, h = (a - b)^2 / (a + b)^2
- Configurable per-zone calibration factors (k_zone).
- Confidence degradation for low landmark visibility, side view depth disagreement (> 3cm),
  or missing views.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from fitcore.schemas import Measurement, Measurements, MeasurementZone
from services.geometry.landmarks import (
    ViewLandmarkResult,
    ViewSilhouette,
    extract_row_width,
)

logger = logging.getLogger("geometry.measurements")

# Default per-zone calibration factors (k=1.0)
DEFAULT_CALIBRATION_FACTORS: dict[MeasurementZone, float] = {
    MeasurementZone.chest: 1.0,
    MeasurementZone.waist: 1.0,
    MeasurementZone.hip: 1.0,
    MeasurementZone.thigh: 1.0,
    MeasurementZone.neck: 1.0,
    MeasurementZone.shoulder_width: 1.0,
    MeasurementZone.sleeve_length: 1.0,
    MeasurementZone.torso_length: 1.0,
    MeasurementZone.garment_length: 1.0,
    MeasurementZone.inseam: 1.0,
    MeasurementZone.back_rise: 1.0,
    MeasurementZone.upper_back: 1.0,
}


def ramanujan_ellipse_perimeter(width_cm: float, depth_cm: float, k: float = 1.0) -> float:
    """Calculate ellipse circumference using Ramanujan's second approximation.

    Args:
        width_cm: Full transverse diameter (front/back width) in centimeters.
        depth_cm: Full sagittal diameter (profile depth) in centimeters.
        k: Configurable zone calibration multiplier.

    Returns:
        Calibrated circumference in centimeters.
    """
    if width_cm <= 0.0 or depth_cm <= 0.0:
        return 0.0

    a = width_cm / 2.0
    b = depth_cm / 2.0
    sum_ab = a + b
    if sum_ab <= 0.0:
        return 0.0

    h = ((a - b) ** 2) / (sum_ab ** 2)
    # Ensure radical operand is strictly non-negative
    radical = max(0.0, 4.0 - 3.0 * h)
    denom = 10.0 + np.sqrt(radical)
    perimeter = np.pi * sum_ab * (1.0 + (3.0 * h) / denom)
    return float(perimeter * k)


def _euclidean_distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute 2D pixel distance between two points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


# Margin factors expand a landmark-derived anatomical span into a scan window,
# allowing for soft tissue/hair extending beyond the landmark points themselves
# while still excluding protrusions (hair, arms/hands held at the sides) that
# are part of the person segmentation mask but not the body region being
# measured. Chosen conservatively and validated against real test photos: wide
# enough to not clip genuine body width, tight enough to exclude hands resting
# at the hips and hair framing the neck. Not derived from tape measurements.
_FIXED_PAIR_MARGIN_FACTOR = 1.4
_TAPER_MARGIN_FACTOR = 1.2


def _landmark_pair_window(left_x: float, right_x: float, margin_factor: float) -> tuple[float, float]:
    """Expand a landmark-pair horizontal span into a bounded scan window."""
    lo, hi = min(left_x, right_x), max(left_x, right_x)
    center = (lo + hi) / 2.0
    half_width = (hi - lo) / 2.0 * margin_factor
    return center - half_width, center + half_width


def _interpolated_taper_window(
    y: float,
    top_left_x: float,
    top_right_x: float,
    top_y: float,
    bottom_left_x: float,
    bottom_right_x: float,
    bottom_y: float,
    margin_factor: float,
) -> tuple[float, float]:
    """Linearly interpolate the expected body boundary columns between two
    landmark-pair heights (e.g. shoulders -> hips, hips -> knees) at row y,
    then expand by margin_factor. The torso and thighs taper roughly linearly
    between these landmark pairs, so this keeps the scan window close to the
    true body width at any row in between while excluding protrusions -- like
    hands resting at the hips -- that fall outside this taper envelope.
    """
    span = bottom_y - top_y
    frac = 0.0 if abs(span) < 1e-6 else float(np.clip((y - top_y) / span, 0.0, 1.0))
    left_x = top_left_x + frac * (bottom_left_x - top_left_x)
    right_x = top_right_x + frac * (bottom_right_x - top_right_x)
    return _landmark_pair_window(left_x, right_x, margin_factor)


# Confidence penalty applied when a bounded width scan comes back flush
# against BOTH edges of its own exclusion window (see
# _window_is_saturated below).
#
# Value and floor interaction, evidenced on a synthetic SMPL round-trip
# (services/geometry/tests/test_synthetic_round_trip.py): window-saturation
# is a near-certain contamination signal (the scan found no background
# pixel anywhere in its own window), stronger than the other per-zone
# penalties this confidence formula already accumulates, so it is allowed
# to push confidence below the general 0.2 floor those other penalties
# share (see _WINDOW_SATURATION_MIN_CONFIDENCE below) -- at 0.2, the
# contaminated zone still pulled fit_smpl_shape's shared 10-beta
# optimization enough to fail an otherwise-clean sibling zone (waist) by
# 1.7-2.2cm; confidence swept down to ~0.02-0.05 cleanly resolved that
# collateral error (waist settled to +-0.2cm) without changing any
# measurement value, the optimizer, or any other zone's confidence.
_WINDOW_SATURATION_PENALTY = 0.85

# Confidence floor specifically for a window-saturated zone, used instead
# of the general 0.2 floor below. Chosen as the evidenced value from the
# sweep above (~0.05 was the smallest tested value that still reliably
# resolved the collateral waist error; 0.0 was also tested and behaved
# almost identically, but is avoided here because fit_smpl_shape treats
# confidence <= 0.0 as "exclude this zone from fitting entirely", a
# different, more drastic behavior than "weight it almost to nothing" --
# this keeps the zone a genuine, if negligible, weighted contributor,
# consistent with the existing confidence-WEIGHTING design).
_WINDOW_SATURATION_MIN_CONFIDENCE = 0.05


def _window_is_saturated(left_col: int, right_col: int, window_lo: float, window_hi: float) -> bool:
    """True if a bounded extract_row_width scan found foreground flush
    against both bounds of its own exclusion window -- i.e. no background
    pixel was found anywhere inside the window to separate the body region
    being measured from an overlapping protrusion (most often an arm). The
    window did not exclude anything in that case, it was simply filled
    edge-to-edge, which is the same contamination signature
    services/geometry/validator.py::check_chest_arm_separation checks for
    (and rejects the photo for) before chest measurement is attempted. A
    clean scan -- one that finds the true body edge strictly inside the
    window, with background on at least one side -- returns False.
    """
    window_lo_i = max(0, int(round(window_lo)))
    window_hi_i = int(round(window_hi))
    return left_col <= window_lo_i + 1 and right_col >= window_hi_i - 1


def calculate_photo_measurements(
    landmark_results: dict[str, ViewLandmarkResult],
    masks: dict[str, np.ndarray],
    user_height_cm: float,
    calibration_factors: dict[MeasurementZone, float] | None = None,
) -> Measurements:
    """Calculate all 12 body measurements from 4-view landmarks and segmentation masks.

    Args:
        landmark_results: Mapping of view name to ViewLandmarkResult.
        masks: Mapping of view name to cleaned 2D binary uint8 mask.
        user_height_cm: Confirmed subject stature in centimeters.
        calibration_factors: Optional custom k-factors per zone (defaults to 1.0).

    Returns:
        Measurements model containing all 12 validated zones with confidence scores.

    Raises:
        ValueError: If required front view is missing.
    """
    k_factors = dict(DEFAULT_CALIBRATION_FACTORS)
    if calibration_factors:
        k_factors.update(calibration_factors)

    if "front" not in landmark_results or "front" not in masks:
        raise ValueError("Front view landmark result and mask are strictly required for body measurement.")

    front_lm = landmark_results["front"]
    front_mask = masks["front"]
    if user_height_cm <= 50.0 or user_height_cm >= 260.0:
        raise ValueError(f"Invalid user height: {user_height_cm} cm. Expected between 50 and 260 cm.")

    cm_per_px_front = front_lm.cm_per_pixel
    if cm_per_px_front <= 0.0 or not np.isfinite(cm_per_px_front):
        raise ValueError(f"Invalid front view scale cm_per_pixel: {cm_per_px_front}")

    # Optional views with fallbacks
    back_lm = landmark_results.get("back", front_lm)
    back_mask = masks.get("back", front_mask)

    has_left = "left" in landmark_results and "left" in masks
    has_right = "right" in landmark_results and "right" in masks

    # Base scale
    cm_per_px = cm_per_px_front

    # 1. Anatomical Reference Landmarks (Front View)
    l_shoulder = front_lm.get("LEFT_SHOULDER")
    r_shoulder = front_lm.get("RIGHT_SHOULDER")
    l_hip = front_lm.get("LEFT_HIP")
    r_hip = front_lm.get("RIGHT_HIP")
    l_knee = front_lm.get("LEFT_KNEE")
    r_knee = front_lm.get("RIGHT_KNEE")
    l_ankle = front_lm.get("LEFT_ANKLE")
    r_ankle = front_lm.get("RIGHT_ANKLE")
    l_elbow = front_lm.get("LEFT_ELBOW")
    r_elbow = front_lm.get("RIGHT_ELBOW")
    l_wrist = front_lm.get("LEFT_WRIST")
    r_wrist = front_lm.get("RIGHT_WRIST")
    nose = front_lm.get("NOSE")
    l_ear = front_lm.get("LEFT_EAR")
    r_ear = front_lm.get("RIGHT_EAR")

    # Midpoints & Vertical levels
    y_shoulders = (l_shoulder.pixel_y + r_shoulder.pixel_y) / 2.0 if (l_shoulder and r_shoulder) else front_lm.image_height * 0.22
    y_hips = (l_hip.pixel_y + r_hip.pixel_y) / 2.0 if (l_hip and r_hip) else front_lm.image_height * 0.52
    y_ankles = (l_ankle.pixel_y + r_ankle.pixel_y) / 2.0 if (l_ankle and r_ankle) else front_lm.image_height * 0.90
    y_knees = (l_knee.pixel_y + r_knee.pixel_y) / 2.0 if (l_knee and r_knee) else front_lm.image_height * 0.72

    torso_span = max(20.0, y_hips - y_shoulders)
    leg_span = max(20.0, y_ankles - y_hips)

    # Crotch level: approximately 10-15% of leg span below hip joint center
    y_crotch = y_hips + 0.12 * leg_span

    # Neck base level (suprasternal / C7 plane)
    y_neck_base = y_shoulders - 0.10 * torso_span

    # Landmark visibility penalty if front view landmarks have lower confidence
    lm_vis_penalty = 0.0
    if front_lm.mean_confidence < 0.70:
        lm_vis_penalty = 0.15
    elif front_lm.mean_confidence < 0.85:
        lm_vis_penalty = 0.08

    # Helper: Sample side depth at corresponding anatomical height (crown-to-sole fraction)
    def _get_side_depth(
        y_front: int | float,
        x_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
        """Extract average depth in cm and depth disagreement penalty at anatomical body plane.

        Args:
            y_front: Row in the front-view mask defining the anatomical height.
            x_bounds: Optional per-side ("left"/"right") column window restricting
                the depth scan, e.g. a landmark-derived window excluding hair
                draping past the head/neck in the profile silhouette. When not
                provided, the scan is unbounded (existing behavior, unchanged
                for chest/waist/hip/thigh where no contamination was found).
        """
        front_top = front_lm.mask_top_row
        front_px_h = max(1.0, float(front_lm.pixel_height))
        # Normalized body-relative height fraction [0.0 = crown, 1.0 = sole]
        body_t = float(np.clip((y_front - front_top) / front_px_h, 0.0, 1.0))

        depths = []
        if has_left:
            side_res = landmark_results["left"]
            y_left = int(round(side_res.mask_top_row + body_t * side_res.pixel_height))
            bounds = x_bounds.get("left") if x_bounds else None
            if bounds:
                _, _, _, d_l = extract_row_width(masks["left"], y_left, side_res.cm_per_pixel, x_min=bounds[0], x_max=bounds[1])
            else:
                _, _, _, d_l = extract_row_width(masks["left"], y_left, side_res.cm_per_pixel)
            if d_l > 0.0:
                depths.append(d_l)

        if has_right:
            side_res = landmark_results["right"]
            y_right = int(round(side_res.mask_top_row + body_t * side_res.pixel_height))
            bounds = x_bounds.get("right") if x_bounds else None
            if bounds:
                _, _, _, d_r = extract_row_width(masks["right"], y_right, side_res.cm_per_pixel, x_min=bounds[0], x_max=bounds[1])
            else:
                _, _, _, d_r = extract_row_width(masks["right"], y_right, side_res.cm_per_pixel)
            if d_r > 0.0:
                depths.append(d_r)

        if not depths:
            # Fallback estimation: anatomical ratio with full missing-side penalty
            return 0.0, 0.25

        avg_depth = float(np.mean(depths))
        disagreement_penalty = 0.0
        if len(depths) == 2:
            diff = abs(depths[0] - depths[1])
            if diff > 3.0:
                disagreement_penalty = 0.15
        elif len(depths) == 1:
            disagreement_penalty = 0.10  # single side view penalty

        return avg_depth, disagreement_penalty

    def _nose_ear_depth_window(side_lm: ViewLandmarkResult) -> tuple[float, float] | None:
        """Bound the profile-view depth scan to a nose<->ear window (expanded by
        a margin), excluding hair that drapes past the head/neck further back
        in the silhouette. Falls back to None (unbounded) if nose/ear landmarks
        are missing or unreliable in this view.
        """
        nose = side_lm.get("NOSE")
        l_ear_s = side_lm.get("LEFT_EAR")
        r_ear_s = side_lm.get("RIGHT_EAR")
        ear = None
        if l_ear_s and r_ear_s:
            ear = l_ear_s if l_ear_s.visibility >= r_ear_s.visibility else r_ear_s
        else:
            ear = l_ear_s or r_ear_s
        if nose and ear and nose.visibility >= 0.5 and ear.visibility >= 0.5:
            return _landmark_pair_window(nose.pixel_x, ear.pixel_x, _FIXED_PAIR_MARGIN_FACTOR)
        return None


    zones: dict[MeasurementZone, Measurement] = {}

    # --- 1. Chest Circumference ---
    # Front width immediately below armpits (lower axillary hollow). This row
    # sits close enough to the shoulder/armpit that an arm resting near the
    # torso is often still part of the same contiguous mask region there, so
    # the scan is bounded to a shoulder->hip taper window (same technique as
    # waist) when those landmarks are available, excluding the arm from the
    # measured width instead of taking the raw mask's full extent.
    y_chest = int(round(y_shoulders + 0.22 * torso_span))
    chest_window_saturated_penalty = 0.0
    if l_shoulder and r_shoulder and l_hip and r_hip:
        chest_x_lo, chest_x_hi = _interpolated_taper_window(
            y_chest,
            l_shoulder.pixel_x, r_shoulder.pixel_x, y_shoulders,
            l_hip.pixel_x, r_hip.pixel_x, y_hips,
            _TAPER_MARGIN_FACTOR,
        )
        chest_left_col, chest_right_col, _, chest_w = extract_row_width(
            front_mask, y_chest, cm_per_px, x_min=chest_x_lo, x_max=chest_x_hi
        )
        if chest_w > 0.0 and _window_is_saturated(chest_left_col, chest_right_col, chest_x_lo, chest_x_hi):
            chest_window_saturated_penalty = _WINDOW_SATURATION_PENALTY
    else:
        _, _, _, chest_w = extract_row_width(front_mask, y_chest, cm_per_px)
    chest_d, chest_penalty = _get_side_depth(y_chest)
    chest_mask_penalty = 0.0
    if chest_w <= 0.0:
        chest_w = user_height_cm * 0.18
        chest_mask_penalty = 0.20
    if chest_d <= 0.0:
        chest_d = chest_w * 0.72  # anatomical fallback

    chest_cm = ramanujan_ellipse_perimeter(chest_w, chest_d, k_factors[MeasurementZone.chest])
    chest_conf_floor = _WINDOW_SATURATION_MIN_CONFIDENCE if chest_window_saturated_penalty > 0.0 else 0.2
    chest_conf = max(
        chest_conf_floor,
        min(0.98, 0.90 - chest_penalty - lm_vis_penalty - chest_mask_penalty - chest_window_saturated_penalty),
    )
    zones[MeasurementZone.chest] = Measurement(
        value_cm=round(max(40.0, chest_cm), 1),
        confidence=round(chest_conf, 2),
        source="photo_measurement",
    )

    # --- 2. Waist Circumference ---
    # Narrowest torso width between 45% and 65% of shoulder-to-hip span.
    # Scan columns are bounded to a shoulder->hip taper window (when those
    # landmarks are available) so arms/hands resting against the torso at
    # these rows are excluded from the width, rather than taking the raw
    # mask's leftmost/rightmost pixel regardless of what body part it is.
    scan_y_start = int(round(y_shoulders + 0.45 * torso_span))
    scan_y_end = int(round(y_shoulders + 0.65 * torso_span))
    waist_widths = []
    for y_scan in range(scan_y_start, max(scan_y_start + 1, scan_y_end)):
        if l_shoulder and r_shoulder and l_hip and r_hip:
            x_lo, x_hi = _interpolated_taper_window(
                y_scan,
                l_shoulder.pixel_x, r_shoulder.pixel_x, y_shoulders,
                l_hip.pixel_x, r_hip.pixel_x, y_hips,
                _TAPER_MARGIN_FACTOR,
            )
            _, _, _, w = extract_row_width(front_mask, y_scan, cm_per_px, x_min=x_lo, x_max=x_hi)
        else:
            _, _, _, w = extract_row_width(front_mask, y_scan, cm_per_px)
        if w > 0:
            waist_widths.append((w, y_scan))

    waist_mask_penalty = 0.0
    if waist_widths:
        waist_w, y_waist = min(waist_widths, key=lambda item: item[0])
    else:
        y_waist = int(round(y_shoulders + 0.55 * torso_span))
        _, _, _, waist_w = extract_row_width(front_mask, y_waist, cm_per_px)
        if waist_w <= 0.0:
            waist_w = user_height_cm * 0.16
            waist_mask_penalty = 0.20

    waist_d, waist_penalty = _get_side_depth(y_waist)
    if waist_d <= 0.0:
        waist_d = waist_w * 0.70

    waist_cm = ramanujan_ellipse_perimeter(waist_w, waist_d, k_factors[MeasurementZone.waist])
    waist_conf = max(0.2, min(0.98, 0.90 - waist_penalty - lm_vis_penalty - waist_mask_penalty))
    zones[MeasurementZone.waist] = Measurement(
        value_cm=round(max(35.0, waist_cm), 1),
        confidence=round(waist_conf, 2),
        source="photo_measurement",
    )

    # --- 3. Hip Circumference ---
    # Widest width between hip landmarks and crotch level. Scan columns are
    # bounded to a window around the hip landmarks (when available) so hands
    # resting against the hips are excluded, rather than taking the raw
    # mask's leftmost/rightmost pixel.
    hip_scan_start = int(round(y_hips - 0.05 * torso_span))
    hip_scan_end = int(round(y_crotch + 0.05 * leg_span))
    hip_x_bounds = _landmark_pair_window(l_hip.pixel_x, r_hip.pixel_x, _FIXED_PAIR_MARGIN_FACTOR) if (l_hip and r_hip) else None
    hip_widths = []
    for y_scan in range(hip_scan_start, max(hip_scan_start + 1, hip_scan_end)):
        if hip_x_bounds:
            _, _, _, w = extract_row_width(front_mask, y_scan, cm_per_px, x_min=hip_x_bounds[0], x_max=hip_x_bounds[1])
        else:
            _, _, _, w = extract_row_width(front_mask, y_scan, cm_per_px)
        if w > 0:
            hip_widths.append((w, y_scan))

    hip_mask_penalty = 0.0
    if hip_widths:
        hip_w, y_hip_max = max(hip_widths, key=lambda item: item[0])
    else:
        y_hip_max = int(round(y_hips))
        _, _, _, hip_w = extract_row_width(front_mask, y_hip_max, cm_per_px)
        if hip_w <= 0.0:
            hip_w = user_height_cm * 0.20
            hip_mask_penalty = 0.20

    hip_d, hip_penalty = _get_side_depth(y_hip_max)
    if hip_d <= 0.0:
        hip_d = hip_w * 0.75

    hip_cm = ramanujan_ellipse_perimeter(hip_w, hip_d, k_factors[MeasurementZone.hip])
    hip_conf = max(0.2, min(0.98, 0.90 - hip_penalty - lm_vis_penalty - hip_mask_penalty))
    zones[MeasurementZone.hip] = Measurement(
        value_cm=round(max(40.0, hip_cm), 1),
        confidence=round(hip_conf, 2),
        source="photo_measurement",
    )

    # --- 4. Thigh Circumference ---
    # One leg's width 5 cm below crotch. Scan columns are bounded to a
    # hip->knee taper window (when those landmarks are available) so hands
    # dangling past the wrist near this row are excluded from the width.
    thigh_offset_px = int(round(5.0 / cm_per_px))
    y_thigh = int(round(y_crotch + thigh_offset_px))
    if l_hip and r_hip and l_knee and r_knee:
        thigh_x_lo, thigh_x_hi = _interpolated_taper_window(
            y_thigh,
            l_hip.pixel_x, r_hip.pixel_x, y_hips,
            l_knee.pixel_x, r_knee.pixel_x, y_knees,
            _TAPER_MARGIN_FACTOR,
        )
        _, _, _, total_thigh_span = extract_row_width(front_mask, y_thigh, cm_per_px, x_min=thigh_x_lo, x_max=thigh_x_hi)
    else:
        _, _, _, total_thigh_span = extract_row_width(front_mask, y_thigh, cm_per_px)
    thigh_mask_penalty = 0.0
    if total_thigh_span <= 0.0:
        thigh_w = user_height_cm * 0.10
        thigh_mask_penalty = 0.20
    else:
        thigh_w = max(8.0, total_thigh_span * 0.48)

    thigh_d, thigh_penalty = _get_side_depth(y_thigh)
    if thigh_d <= 0.0:
        thigh_d = thigh_w * 0.95

    thigh_cm = ramanujan_ellipse_perimeter(thigh_w, thigh_d, k_factors[MeasurementZone.thigh])
    thigh_conf = max(0.2, min(0.95, 0.88 - thigh_penalty - lm_vis_penalty - thigh_mask_penalty))
    zones[MeasurementZone.thigh] = Measurement(
        value_cm=round(max(20.0, thigh_cm), 1),
        confidence=round(thigh_conf, 2),
        source="photo_measurement",
    )

    # --- 5. Neck Circumference ---
    # Horizontal plane directly superior to suprasternal notch. Scan columns
    # are bounded to an ear-span window (neck is never wider than the head),
    # or a conservative fraction of shoulder span if ears aren't reliably
    # detected, so hair framing the face/neck at this row is excluded from
    # the width instead of being read as part of the body silhouette.
    y_neck = int(round(y_neck_base - 0.08 * torso_span))
    if l_ear and r_ear and l_ear.visibility >= 0.5 and r_ear.visibility >= 0.5:
        neck_x_bounds = _landmark_pair_window(l_ear.pixel_x, r_ear.pixel_x, _FIXED_PAIR_MARGIN_FACTOR)
    elif l_shoulder and r_shoulder:
        sh_center = (l_shoulder.pixel_x + r_shoulder.pixel_x) / 2.0
        sh_half_span = abs(l_shoulder.pixel_x - r_shoulder.pixel_x) / 2.0
        neck_x_bounds = (sh_center - sh_half_span * 0.5, sh_center + sh_half_span * 0.5)
    else:
        neck_x_bounds = None

    if neck_x_bounds:
        _, _, _, neck_w = extract_row_width(front_mask, y_neck, cm_per_px, x_min=neck_x_bounds[0], x_max=neck_x_bounds[1])
    else:
        _, _, _, neck_w = extract_row_width(front_mask, y_neck, cm_per_px)
    neck_mask_penalty = 0.0
    if neck_w <= 0.0:
        neck_w = user_height_cm * 0.07
        neck_mask_penalty = 0.20

    # Bound the side-view depth scan to a nose<->ear window per side, when
    # available, excluding hair draping past the head/neck in the profile
    # silhouette (the equivalent problem to front-view hair contamination,
    # but in the depth direction). Falls back to the prior unbounded scan
    # for any side missing reliable nose/ear landmarks.
    neck_depth_bounds: dict[str, tuple[float, float]] = {}
    if has_left:
        b = _nose_ear_depth_window(landmark_results["left"])
        if b:
            neck_depth_bounds["left"] = b
    if has_right:
        b = _nose_ear_depth_window(landmark_results["right"])
        if b:
            neck_depth_bounds["right"] = b

    neck_d, neck_penalty = _get_side_depth(y_neck, x_bounds=neck_depth_bounds or None)
    neck_depth_implausible_penalty = 0.0
    if neck_d <= 0.0:
        neck_d = neck_w * 0.88
    elif neck_w > 0.0 and neck_d > neck_w * 1.3:
        # Even bounded, the depth component is implausibly large relative to
        # width (a neck's front-to-back depth is not normally much greater
        # than its side-to-side width). Rather than silently trusting or
        # clamping this value, report it with materially lower confidence --
        # the measurement is used as-is, but flagged as unreliable, not hidden.
        neck_depth_implausible_penalty = 0.30

    neck_cm = ramanujan_ellipse_perimeter(neck_w, neck_d, k_factors[MeasurementZone.neck])
    neck_conf = max(
        0.1,
        min(0.92, 0.85 - neck_penalty - lm_vis_penalty - neck_mask_penalty - neck_depth_implausible_penalty),
    )
    zones[MeasurementZone.neck] = Measurement(
        value_cm=round(max(20.0, neck_cm), 1),
        confidence=round(neck_conf, 2),
        source="photo_measurement",
    )

    # --- 6. Shoulder Width ---
    # Linear distance across back between left and right acromion
    b_l_sh = back_lm.get("LEFT_SHOULDER")
    b_r_sh = back_lm.get("RIGHT_SHOULDER")
    if b_l_sh and b_r_sh:
        sh_px = _euclidean_distance_2d(
            (b_l_sh.pixel_x, b_l_sh.pixel_y),
            (b_r_sh.pixel_x, b_r_sh.pixel_y),
        )
        sh_cm = sh_px * back_lm.cm_per_pixel * k_factors[MeasurementZone.shoulder_width]
        sh_conf = 0.92 if "back" in landmark_results else 0.82
    elif l_shoulder and r_shoulder:
        # Fallback to front view shoulders
        sh_px = _euclidean_distance_2d(
            (l_shoulder.pixel_x, l_shoulder.pixel_y),
            (r_shoulder.pixel_x, r_shoulder.pixel_y),
        )
        sh_cm = sh_px * cm_per_px * k_factors[MeasurementZone.shoulder_width]
        sh_conf = 0.80
    else:
        # Anatomical stature ratio fallback
        sh_cm = user_height_cm * 0.24 * k_factors[MeasurementZone.shoulder_width]
        sh_conf = 0.40

    sh_conf = max(0.2, min(0.98, sh_conf - lm_vis_penalty))
    zones[MeasurementZone.shoulder_width] = Measurement(
        value_cm=round(max(15.0, sh_cm), 1),
        confidence=round(sh_conf, 2),
        source="photo_measurement",
    )

    # --- 7. Sleeve Length ---
    # Contour path: Acromion -> Elbow -> Wrist
    sleeve_lengths = []
    if l_shoulder and l_elbow and l_wrist:
        len_l = (
            _euclidean_distance_2d((l_shoulder.pixel_x, l_shoulder.pixel_y), (l_elbow.pixel_x, l_elbow.pixel_y))
            + _euclidean_distance_2d((l_elbow.pixel_x, l_elbow.pixel_y), (l_wrist.pixel_x, l_wrist.pixel_y))
        )
        sleeve_lengths.append(len_l * cm_per_px)

    if r_shoulder and r_elbow and r_wrist:
        len_r = (
            _euclidean_distance_2d((r_shoulder.pixel_x, r_shoulder.pixel_y), (r_elbow.pixel_x, r_elbow.pixel_y))
            + _euclidean_distance_2d((r_elbow.pixel_x, r_elbow.pixel_y), (r_wrist.pixel_x, r_wrist.pixel_y))
        )
        sleeve_lengths.append(len_r * cm_per_px)

    if sleeve_lengths:
        avg_sleeve_cm = float(np.mean(sleeve_lengths)) * k_factors[MeasurementZone.sleeve_length]
        sleeve_conf = 0.90 if len(sleeve_lengths) == 2 else 0.78
    else:
        avg_sleeve_cm = user_height_cm * 0.33 * k_factors[MeasurementZone.sleeve_length]
        sleeve_conf = 0.45

    sleeve_conf = max(0.2, min(0.98, sleeve_conf - lm_vis_penalty))
    zones[MeasurementZone.sleeve_length] = Measurement(
        value_cm=round(max(20.0, avg_sleeve_cm), 1),
        confidence=round(sleeve_conf, 2),
        source="photo_measurement",
    )

    # --- 8. Inseam ---
    # Crotch / perineal level down to inner ankle
    if l_ankle and r_ankle:
        y_ankles_val = (l_ankle.pixel_y + r_ankle.pixel_y) / 2.0
        inseam_conf = 0.90
    elif l_ankle or r_ankle:
        y_ankles_val = l_ankle.pixel_y if l_ankle else r_ankle.pixel_y
        inseam_conf = 0.80
    else:
        y_ankles_val = front_lm.mask_bottom_row
        inseam_conf = 0.65

    inseam_px = y_ankles_val - y_crotch
    if inseam_px <= 10.0:
        inseam_cm = user_height_cm * 0.44 * k_factors[MeasurementZone.inseam]
        inseam_conf = 0.40
    else:
        inseam_cm = inseam_px * cm_per_px * k_factors[MeasurementZone.inseam]

    inseam_conf = max(0.2, min(0.98, inseam_conf - lm_vis_penalty))
    zones[MeasurementZone.inseam] = Measurement(
        value_cm=round(max(25.0, inseam_cm), 1),
        confidence=round(inseam_conf, 2),
        source="photo_measurement",
    )

    # --- 9. Torso Length ---
    # Cervical vertebra base (C7 / neck base) down to horizontal waist plane.
    # The dominant error found here this session was NOT this top reference
    # -- it was fitting.py's ground truth using the spine1 joint as a "waist
    # plane" stand-in, ~6-10cm from where the waist circumference actually
    # measures the waist (see fitting.py's torso_length comment, now fixed
    # to share waist_cm's own plane). Checked three top-reference variants
    # against the corrected ground truth (y_neck_base as already used here,
    # y_shoulders directly, and y_shoulders with a small fixed C7-above-
    # shoulder offset measured directly on this mesh): y_neck_base's
    # long-standing error (~1.4-2.1cm here) was the smallest of the three,
    # so it is left unchanged rather than "fixed" into a worse number.
    torso_len_px = y_waist - y_neck_base
    if torso_len_px <= 10.0:
        torso_len_cm = user_height_cm * 0.28 * k_factors[MeasurementZone.torso_length]
        torso_conf = 0.45
    else:
        torso_len_cm = torso_len_px * cm_per_px * k_factors[MeasurementZone.torso_length]
        torso_conf = 0.91

    torso_conf = max(0.2, min(0.98, torso_conf - lm_vis_penalty))
    zones[MeasurementZone.torso_length] = Measurement(
        value_cm=round(max(15.0, torso_len_cm), 1),
        confidence=round(torso_conf, 2),
        source="photo_measurement",
    )

    # --- 10. Back Rise ---
    # Midline distance along sacrum from posterior waist center down to crotch intersection
    back_rise_px = y_crotch - y_waist
    if back_rise_px <= 5.0:
        back_rise_cm = user_height_cm * 0.16 * k_factors[MeasurementZone.back_rise]
        back_conf = 0.45
    else:
        back_rise_cm = back_rise_px * cm_per_px * k_factors[MeasurementZone.back_rise]
        back_conf = 0.88 if "back" in landmark_results else 0.78

    back_conf = max(0.2, min(0.98, back_conf - lm_vis_penalty))
    zones[MeasurementZone.back_rise] = Measurement(
        value_cm=round(max(10.0, back_rise_cm), 1),
        confidence=round(back_conf, 2),
        source="photo_measurement",
    )

    # --- 11. Upper Back ---
    # Transverse distance across upper back between posterior armscye (armpit)
    # creases. This row sits close to the shoulder/armpit on the BACK view,
    # the same height band found (on the front view's chest scan) to fuse
    # arm and torso into one contiguous silhouette region -- confirmed on a
    # synthetic SMPL ground-truth check to affect this row too (unbounded
    # scan ~55.6cm vs. ~41.4cm true armscye-to-armscye distance). Bounded to
    # a shoulder->hip taper window on the back view's own landmarks, the
    # same technique chest/waist already use on the front view, when those
    # landmarks are available.
    y_upper_back = int(round(y_shoulders + 0.12 * torso_span))
    b_l_hip = back_lm.get("LEFT_HIP")
    b_r_hip = back_lm.get("RIGHT_HIP")
    upper_back_window_saturated_penalty = 0.0
    if b_l_sh and b_r_sh and b_l_hip and b_r_hip:
        b_y_shoulders = (b_l_sh.pixel_y + b_r_sh.pixel_y) / 2.0
        b_y_hips = (b_l_hip.pixel_y + b_r_hip.pixel_y) / 2.0
        ub_x_lo, ub_x_hi = _interpolated_taper_window(
            y_upper_back,
            b_l_sh.pixel_x, b_r_sh.pixel_x, b_y_shoulders,
            b_l_hip.pixel_x, b_r_hip.pixel_x, b_y_hips,
            _TAPER_MARGIN_FACTOR,
        )
        ub_left_col, ub_right_col, _, upper_back_w = extract_row_width(
            back_mask, y_upper_back, back_lm.cm_per_pixel, x_min=ub_x_lo, x_max=ub_x_hi
        )
        if upper_back_w > 0.0 and _window_is_saturated(ub_left_col, ub_right_col, ub_x_lo, ub_x_hi):
            upper_back_window_saturated_penalty = _WINDOW_SATURATION_PENALTY
    else:
        _, _, _, upper_back_w = extract_row_width(back_mask, y_upper_back, back_lm.cm_per_pixel)

    if upper_back_w <= 0.0:
        upper_back_cm = user_height_cm * 0.22 * k_factors[MeasurementZone.upper_back]
        upper_back_conf = 0.45
    else:
        upper_back_cm = upper_back_w * k_factors[MeasurementZone.upper_back]
        upper_back_conf = 0.90 if "back" in landmark_results else 0.80

    upper_back_conf_floor = _WINDOW_SATURATION_MIN_CONFIDENCE if upper_back_window_saturated_penalty > 0.0 else 0.2
    upper_back_conf = max(
        upper_back_conf_floor, min(0.98, upper_back_conf - lm_vis_penalty - upper_back_window_saturated_penalty)
    )
    zones[MeasurementZone.upper_back] = Measurement(
        value_cm=round(max(15.0, upper_back_cm), 1),
        confidence=round(upper_back_conf, 2),
        source="photo_measurement",
    )

    # --- 12. Garment Length ---
    # Reference distance along posterior midline from cervical base (C7,
    # approximated by y_shoulders -- see torso_length above) down to the hip
    # LINE. Uses y_hips (the hip landmark row) rather than y_hip_max (the
    # widest-point row the HIP zone searches for, left untouched above):
    # those are different anatomical references that happen to share the
    # word "hip" -- y_hip_max is deliberately searched for near/below the
    # glutes to find the hip CIRCUMFERENCE's widest point, which sits well
    # below the iliac-crest "hip line" a garment hem is referenced against.
    # Confirmed on a synthetic SMPL ground-truth check: y_hip_max landed
    # ~9-11cm further down than the hip-socket height y_hips tracks.
    garment_len_px = y_hips - y_shoulders
    if garment_len_px <= 10.0:
        garment_len_cm = user_height_cm * 0.42 * k_factors[MeasurementZone.garment_length]
        garment_conf = 0.45
    else:
        garment_len_cm = garment_len_px * cm_per_px * k_factors[MeasurementZone.garment_length]
        garment_conf = 0.90

    garment_conf = max(0.2, min(0.98, garment_conf - lm_vis_penalty))
    zones[MeasurementZone.garment_length] = Measurement(
        value_cm=round(max(20.0, garment_len_cm), 1),
        confidence=round(garment_conf, 2),
        source="photo_measurement",
    )

    return Measurements(zones=zones)
