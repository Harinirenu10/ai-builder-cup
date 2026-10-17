"""Unit tests for services/geometry/measurements.py and eval/calibrate_k.py.

Verifies:
1. Ramanujan ellipse circumference approximation accuracy.
2. Complete calculation of all 12 MeasurementZone fields.
3. Metric scaling from pixel coordinates using user height.
4. Confidence degradation on side view depth disagreement (> 3cm).
5. Safe handling and fallback when optional views (back, side) are missing.
6. Exception raised when mandatory front view is missing.
7. Least-squares zone calibration fitting (eval/calibrate_k.py).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from fitcore.schemas import MeasurementZone, Measurements
from services.geometry.landmarks import (
    PoseLandmarkPoint,
    ViewLandmarkResult,
    ViewSilhouette,
    extract_row_width,
)
from services.geometry.measurements import (
    calculate_photo_measurements,
    ramanujan_ellipse_perimeter,
)
from eval.calibrate_k import fit_calibration_factors, leave_one_subject_out_eval


def test_ramanujan_ellipse_perimeter_circle():
    """Verify Ramanujan formula for a circle (a = b = 10cm) yields 2*pi*r."""
    w = 20.0  # diameter = 20 cm, radius = 10 cm
    d = 20.0
    expected = 2.0 * np.pi * 10.0  # ~62.8318 cm
    actual = ramanujan_ellipse_perimeter(width_cm=w, depth_cm=d, k=1.0)
    assert abs(actual - expected) < 1e-3


def test_ramanujan_ellipse_perimeter_typical_chest():
    """Verify Ramanujan formula for typical chest cross section (width=34cm, depth=26cm)."""
    # a = 17, b = 13
    p = ramanujan_ellipse_perimeter(width_cm=34.0, depth_cm=26.0, k=1.0)
    # Expected perimeter is ~94.88 cm
    assert 94.0 <= p <= 96.0

    # Multiplying by k=1.05 increases perimeter by exactly 5%
    p_k = ramanujan_ellipse_perimeter(width_cm=34.0, depth_cm=26.0, k=1.05)
    assert abs(p_k - p * 1.05) < 1e-4


def test_ramanujan_ellipse_perimeter_zero_or_negative():
    """Invalid or zero dimensions must safely return 0.0."""
    assert ramanujan_ellipse_perimeter(0.0, 20.0) == 0.0
    assert ramanujan_ellipse_perimeter(20.0, -5.0) == 0.0


def _build_synthetic_view_data(
    img_w: int = 400,
    img_h: int = 800,
    user_height_cm: float = 175.0,
    body_width_px: int = 100,
    body_depth_px: int = 70,
) -> tuple[dict[str, ViewLandmarkResult], dict[str, np.ndarray]]:
    """Build consistent synthetic 4-view landmarks and binary masks for testing."""
    cm_per_px = user_height_cm / 600.0  # 600 pixel person height = 0.2917 cm/px

    views = ["front", "back", "left", "right"]
    landmark_results = {}
    masks = {}

    for v in views:
        # Mask: person between rows 100 and 700 (600 pixels)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        w_px = body_width_px if v in ("front", "back") else body_depth_px
        c_x = img_w // 2
        mask[100:700, c_x - w_px // 2 : c_x + w_px // 2] = 1
        masks[v] = mask

        # Landmarks
        points = []
        # Key landmark definitions in vertical order
        lm_configs = [
            (0, "NOSE", 0.5, 0.15),
            (11, "LEFT_SHOULDER", 0.68, 0.25),
            (12, "RIGHT_SHOULDER", 0.32, 0.25),
            (13, "LEFT_ELBOW", 0.73, 0.38),
            (14, "RIGHT_ELBOW", 0.27, 0.38),
            (15, "LEFT_WRIST", 0.76, 0.50),
            (16, "RIGHT_WRIST", 0.24, 0.50),
            (23, "LEFT_HIP", 0.58, 0.52),
            (24, "RIGHT_HIP", 0.42, 0.52),
            (25, "LEFT_KNEE", 0.56, 0.70),
            (26, "RIGHT_KNEE", 0.44, 0.70),
            (27, "LEFT_ANKLE", 0.55, 0.88),
            (28, "RIGHT_ANKLE", 0.45, 0.88),

        ]
        # Fill remaining up to 33
        existing_indices = {c[0] for c in lm_configs}
        for i in range(33):
            if i in existing_indices:
                cfg = [c for c in lm_configs if c[0] == i][0]
                points.append(
                    PoseLandmarkPoint(
                        index=i,
                        name=cfg[1],
                        x=cfg[2],
                        y=cfg[3],
                        z=0.0,
                        visibility=0.95,
                        presence=0.95,
                        pixel_x=cfg[2] * img_w,
                        pixel_y=cfg[3] * img_h,
                    )
                )
            else:
                points.append(
                    PoseLandmarkPoint(
                        index=i,
                        name=f"LANDMARK_{i}",
                        x=0.5,
                        y=0.1 + (i / 33.0) * 0.8,
                        z=0.0,
                        visibility=0.9,
                        presence=0.9,
                        pixel_x=0.5 * img_w,
                        pixel_y=(0.1 + (i / 33.0) * 0.8) * img_h,
                    )
                )

        landmark_results[v] = ViewLandmarkResult(
            view=v,
            landmarks=points,
            image_width=img_w,
            image_height=img_h,
            mask_top_row=100,
            mask_bottom_row=699,
            pixel_height=600,
            cm_per_pixel=cm_per_px,
            mean_confidence=0.92,
        )

    return landmark_results, masks


def _build_contaminated_front_view_data(
    img_w: int = 400,
    img_h: int = 800,
    user_height_cm: float = 175.0,
) -> tuple[dict[str, ViewLandmarkResult], dict[str, np.ndarray]]:
    """Build 4-view synthetic data where the FRONT mask has a narrow true
    torso silhouette (columns 160-240, 80px) plus deliberately injected wide
    contamination blobs simulating hair near the neck row and an arm/hand
    resting against the body near the waist/hip/thigh rows -- the real
    failure modes found testing calculate_photo_measurements with real
    consented photos (see services/geometry/measurements.py neck/waist/hip/
    thigh sections). Back/left/right views reuse the plain uncontaminated
    rectangle from _build_synthetic_view_data so only the front-view width
    extraction under test is affected.
    """
    cm_per_px = user_height_cm / 600.0
    landmark_results, masks = _build_synthetic_view_data(img_w, img_h, user_height_cm)
    landmark_results = dict(landmark_results)
    masks = dict(masks)

    front_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # True torso silhouette: constant 80px width, full body height.
    front_mask[100:700, 160:240] = 1
    # Hair contamination at the neck row (computed internally as ~row 161
    # for these landmarks): much wider than the true torso.
    front_mask[140:200, 80:320] = 1
    # Arm/hand contamination spanning the chest/waist/hip/thigh scan rows
    # (computed internally as ~rows 248-468 for these landmarks): simulates
    # an arm hanging at/near the side, merged into the torso silhouette with
    # no gap, exactly as seen in the real photos that exposed this bug (and,
    # for chest specifically, as confirmed via a synthetic SMPL round-trip
    # test where the arm and torso share the same contiguous silhouette
    # region near the armpit for any tested abduction angle).
    front_mask[248:470, 40:360] = 1
    masks["front"] = front_mask

    def _pt(index: int, name: str, x_frac: float, y_frac: float, vis: float = 0.95) -> PoseLandmarkPoint:
        return PoseLandmarkPoint(
            index=index, name=name, x=x_frac, y=y_frac, z=0.0,
            visibility=vis, presence=vis,
            pixel_x=x_frac * img_w, pixel_y=y_frac * img_h,
        )

    lm_configs = [
        (0, "NOSE", 0.50, 0.15),
        (7, "LEFT_EAR", 0.54, 0.16),
        (8, "RIGHT_EAR", 0.46, 0.16),
        (11, "LEFT_SHOULDER", 0.60, 0.25),
        (12, "RIGHT_SHOULDER", 0.40, 0.25),
        (13, "LEFT_ELBOW", 0.70, 0.38),
        (14, "RIGHT_ELBOW", 0.30, 0.38),
        (15, "LEFT_WRIST", 0.80, 0.50),
        (16, "RIGHT_WRIST", 0.20, 0.50),
        (23, "LEFT_HIP", 0.58, 0.52),
        (24, "RIGHT_HIP", 0.42, 0.52),
        (25, "LEFT_KNEE", 0.56, 0.70),
        (26, "RIGHT_KNEE", 0.44, 0.70),
        (27, "LEFT_ANKLE", 0.55, 0.88),
        (28, "RIGHT_ANKLE", 0.45, 0.88),
    ]
    existing_indices = {c[0] for c in lm_configs}
    points = []
    for i in range(33):
        if i in existing_indices:
            cfg = next(c for c in lm_configs if c[0] == i)
            points.append(_pt(*cfg))
        else:
            points.append(_pt(i, f"LANDMARK_{i}", 0.5, 0.1 + (i / 33.0) * 0.8, vis=0.9))

    landmark_results["front"] = ViewLandmarkResult(
        view="front",
        landmarks=points,
        image_width=img_w,
        image_height=img_h,
        mask_top_row=100,
        mask_bottom_row=699,
        pixel_height=600,
        cm_per_pixel=cm_per_px,
        mean_confidence=0.92,
    )

    return landmark_results, masks


def _build_neck_depth_contaminated_data(
    img_w: int = 400,
    img_h: int = 800,
    user_height_cm: float = 175.0,
) -> tuple[dict[str, ViewLandmarkResult], dict[str, np.ndarray]]:
    """Build 4-view synthetic data where the LEFT and RIGHT (side) masks have
    a narrow true depth silhouette plus a wide contamination blob at the neck
    row, simulating hair draping past the head/neck in profile -- the real
    failure found testing with real consented photos, where a side-view
    depth of ~18.9cm (versus a width of ~16.3cm) propagated into a ~55cm
    'neck' circumference even after the front-view width fix. Front/back
    views reuse the plain uncontaminated rectangle from
    _build_synthetic_view_data.
    """
    landmark_results, masks = _build_synthetic_view_data(img_w, img_h, user_height_cm)
    landmark_results = dict(landmark_results)
    masks = dict(masks)

    def _pt(index: int, name: str, x_frac: float, y_frac: float, vis: float = 0.95) -> PoseLandmarkPoint:
        return PoseLandmarkPoint(
            index=index, name=name, x=x_frac, y=y_frac, z=0.0,
            visibility=vis, presence=vis,
            pixel_x=x_frac * img_w, pixel_y=y_frac * img_h,
        )

    for side, nose_x_frac, ear_x_frac in (("left", 0.58, 0.52), ("right", 0.58, 0.52)):
        side_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # True depth silhouette: constant 60px width, full body height.
        side_mask[100:700, 170:230] = 1
        # Hair contamination near the neck row (~row 161 for these
        # landmarks, same formula as the front-view neck row): much wider
        # than the true depth, simulating hair draping past the head/neck.
        side_mask[150:175, 50:300] = 1
        masks[side] = side_mask

        base_points = list(landmark_results[side].landmarks)
        points = []
        for p in base_points:
            if p.name == "NOSE":
                points.append(_pt(p.index, "NOSE", nose_x_frac, p.y))
            elif p.name == "LANDMARK_7":
                points.append(_pt(p.index, "LEFT_EAR", ear_x_frac, p.y))
            elif p.name == "LANDMARK_8":
                points.append(_pt(p.index, "RIGHT_EAR", ear_x_frac + 0.005, p.y))
            else:
                points.append(p)

        old = landmark_results[side]
        landmark_results[side] = ViewLandmarkResult(
            view=side,
            landmarks=points,
            image_width=old.image_width,
            image_height=old.image_height,
            mask_top_row=old.mask_top_row,
            mask_bottom_row=old.mask_bottom_row,
            pixel_height=old.pixel_height,
            cm_per_pixel=old.cm_per_pixel,
            mean_confidence=old.mean_confidence,
        )

    return landmark_results, masks


def test_neck_depth_excludes_hair_contamination_in_side_views():
    """Regression test: hair draping past the head/neck in the LEFT/RIGHT
    profile silhouettes must not inflate the neck's depth component.
    Reproduces a real failure found testing with real consented photos: after
    fixing front-view neck width (hair contamination there), the neck
    circumference was still ~55cm because the side-view depth scan (~18.9cm,
    larger than the ~16.3cm width -- anatomically implausible for a neck) was
    never bounded at all.
    """
    lm_results, masks = _build_neck_depth_contaminated_data()
    measurements = calculate_photo_measurements(
        landmark_results=lm_results, masks=masks, user_height_cm=175.0,
    )
    neck_cm = measurements.get_cm(MeasurementZone.neck)
    neck_conf = measurements.zones[MeasurementZone.neck].confidence
    assert neck_cm is not None
    # The contamination blob is 250px (~73cm) wide; unbounded depth would
    # push the circumference well past 150cm. Bounded depth must land far
    # below that. (Not asserting an exact "plausible neck" number here --
    # this fixture's front-view width uses the shoulder-ratio fallback, not
    # a precisely anatomical neck width, so only the relative reduction from
    # excluding the depth contamination is being verified.)
    assert neck_cm < 100.0, f"neck ({neck_cm}cm) still appears depth-contaminated by hair"
    assert neck_conf > 0.3, f"neck confidence ({neck_conf}) unexpectedly collapsed"


def test_neck_width_excludes_hair_contamination():
    """Regression test: hair draped near the neck row must not inflate the
    measured neck width. Reproduces a real failure found testing with real
    consented photos, where the formulaic neck row landed in hair and
    produced an anatomically impossible ~63-70cm 'neck' (a 1011x1556 real
    photo's neck row measured 62px/25.2cm wide at a row still within the
    hair silhouette, not the bare neck).
    """
    lm_results, masks = _build_contaminated_front_view_data()
    front_lm = lm_results["front"]

    # Replicate the internal neck-row calculation to confirm the same row the
    # real pipeline uses falls inside the injected hair contamination.
    l_sh, r_sh = front_lm.get("LEFT_SHOULDER"), front_lm.get("RIGHT_SHOULDER")
    l_hip, r_hip = front_lm.get("LEFT_HIP"), front_lm.get("RIGHT_HIP")
    y_shoulders = (l_sh.pixel_y + r_sh.pixel_y) / 2.0
    y_hips = (l_hip.pixel_y + r_hip.pixel_y) / 2.0
    torso_span = y_hips - y_shoulders
    y_neck = round((y_shoulders - 0.10 * torso_span) - 0.08 * torso_span)

    # "Before" the fix: a raw, unbounded scan at that exact row picks up the
    # hair contamination instead of the true torso/neck width.
    _, _, _, w_cm_before = extract_row_width(masks["front"], y_neck, front_lm.cm_per_pixel)
    assert w_cm_before > 60.0, "test setup sanity check: hair should dominate an unbounded scan at this row"

    measurements = calculate_photo_measurements(
        landmark_results=lm_results, masks=masks, user_height_cm=175.0,
    )
    neck_cm_after = measurements.get_cm(MeasurementZone.neck)
    assert neck_cm_after is not None
    # The final circumference combines width with a depth component from the
    # (uncontaminated) side-view fixture, so this is not a 1:1 comparison
    # with w_cm_before -- but propagating the raw hair-contaminated width
    # through the same formula would produce a circumference over 150cm in
    # this fixture. The fix must land far below that, confirming the
    # landmark-derived bound actually excluded the hair.
    assert neck_cm_after < 70.0, f"neck ({neck_cm_after}cm) does not reflect the expected hair exclusion"


def test_chest_width_excludes_arm_contamination():
    """Regression test: an arm/hand near the armpit/chest row must not
    inflate chest width. Found via a synthetic SMPL ground-truth round trip
    (test_synthetic_round_trip.py): the formulaic chest row sits close
    enough to the armpit that the arm is part of the same contiguous
    silhouette region as the torso there, confirmed on a real photo too
    (unbounded chest width 43.5cm -> 35.4cm once bounded). Unlike the hair/
    hand contamination for neck/waist/hip/thigh, this does not fully resolve
    chest's accuracy (a known, documented, unresolved limitation -- see
    test_synthetic_round_trip.py's module docstring) -- this test only
    verifies the bounding reduces contamination, the same way it was
    verified on the real photo.
    """
    lm_results, masks = _build_contaminated_front_view_data()
    front_lm = lm_results["front"]

    # "Before" the fix: a raw, unbounded scan at the chest row (~row 248 for
    # these landmarks) picks up the arm/hand contamination blob.
    _, _, _, w_cm_before = extract_row_width(masks["front"], 248, front_lm.cm_per_pixel)
    assert w_cm_before > 80.0, "test setup sanity check: arm contamination should dominate an unbounded scan"

    measurements = calculate_photo_measurements(
        landmark_results=lm_results, masks=masks, user_height_cm=175.0,
    )
    chest_cm = measurements.get_cm(MeasurementZone.chest)
    assert chest_cm is not None
    # The contamination blob is 320px (~93cm) wide; unbounded extraction would
    # push the circumference well past 150cm. Bounded extraction must land
    # meaningfully below that, confirming the shoulder->hip taper bound
    # engaged (even though, per the docstring above, this does not fully
    # eliminate chest's contamination in every case -- see the xfail'd
    # synthetic round-trip test for the deeper, unresolved limitation).
    assert chest_cm < 110.0, f"chest ({chest_cm}cm) does not reflect the expected contamination reduction"


def test_waist_hip_thigh_width_exclude_arm_contamination():
    """Regression test: an arm/hand resting against the body must not inflate
    waist, hip, or thigh width. Reproduces a real failure where the hip scan
    row, visualized against a real photo's segmentation mask, ran directly
    through the subject's hands -- measuring hand-to-hand width (~124cm)
    instead of hip circumference, because the row-width scan had no notion of
    which body part it was touching.
    """
    lm_results, masks = _build_contaminated_front_view_data()
    front_lm = lm_results["front"]

    # "Before" the fix: a raw, unbounded scan anywhere in the contaminated
    # row band (e.g. the hip row used internally, ~row 416 for these
    # landmarks) picks up the arm/hand blob instead of the true torso width.
    _, _, _, w_cm_before = extract_row_width(masks["front"], 416, front_lm.cm_per_pixel)
    assert w_cm_before > 80.0, "test setup sanity check: arm contamination should dominate an unbounded scan"

    measurements = calculate_photo_measurements(
        landmark_results=lm_results, masks=masks, user_height_cm=175.0,
    )
    waist_cm = measurements.get_cm(MeasurementZone.waist)
    hip_cm = measurements.get_cm(MeasurementZone.hip)
    thigh_cm = measurements.get_cm(MeasurementZone.thigh)

    # The contamination blob is 320px (~93cm) wide; unbounded extraction
    # would push circumferences well over 150cm. True torso is 80px
    # (~23.3cm) wide. Bounded extraction must land far closer to the true
    # torso width than to the contaminated one.
    assert waist_cm is not None and waist_cm < 90.0, f"waist ({waist_cm}cm) still appears arm-contaminated"
    assert hip_cm is not None and hip_cm < 90.0, f"hip ({hip_cm}cm) still appears arm-contaminated"
    assert thigh_cm is not None and thigh_cm < 90.0, f"thigh ({thigh_cm}cm) still appears arm-contaminated"


def test_calculate_photo_measurements_all_12_zones():
    """Verify that all 12 measurement zones are populated with positive values and valid confidences."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)

    measurements = calculate_photo_measurements(
        landmark_results=lm_results,
        masks=masks,
        user_height_cm=175.0,
    )

    assert isinstance(measurements, Measurements)
    assert len(measurements.zones) == 12

    # Check all 12 zones explicitly
    for zone in MeasurementZone:
        m = measurements.zones.get(zone)
        assert m is not None, f"Missing measurement for zone: {zone.value}"
        assert m.value_cm > 0.0, f"Non-positive measurement for zone: {zone.value}"
        assert 0.0 <= m.confidence <= 1.0, f"Invalid confidence for zone: {zone.value}"
        assert m.source == "photo_measurement"

    # Plausibility checks for human proportions
    assert 70.0 <= measurements.get_cm(MeasurementZone.chest) <= 120.0
    assert 60.0 <= measurements.get_cm(MeasurementZone.waist) <= 110.0
    # Lower bound relaxed from 75.0: with landmark-bounded hip width scanning
    # (added to exclude hands resting at the hips from the measured width),
    # this fixture's plain rectangular synthetic mask -- whose hip landmarks
    # sit at only 64% of the mask's total width, narrower than the ~70-85%
    # ratio typical of real photos -- now yields a slightly tighter 74.1cm,
    # still a plausible human hip circumference.
    assert 70.0 <= measurements.get_cm(MeasurementZone.hip) <= 125.0
    assert 30.0 <= measurements.get_cm(MeasurementZone.shoulder_width) <= 60.0
    assert 60.0 <= measurements.get_cm(MeasurementZone.inseam) <= 95.0


def test_side_depth_disagreement_lowers_confidence():
    """Verify that left/right side depth discrepancy > 3cm degrades circumference confidence."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)

    # Artificially widen left side depth relative to right side depth by 20 pixels (~5.8 cm)
    masks["left"][100:700, 160:260] = 1  # 100 pixels wide
    masks["right"][100:700, 185:215] = 1  # 30 pixels wide

    m_disagree = calculate_photo_measurements(
        landmark_results=lm_results,
        masks=masks,
        user_height_cm=175.0,
    )

    # Disagreement penalty (0.15) should result in confidence <= 0.75
    chest_m = m_disagree.zones[MeasurementZone.chest]
    assert chest_m.confidence <= 0.75


def test_missing_front_view_raises_value_error():
    """Missing front view must raise ValueError."""
    lm_results, masks = _build_synthetic_view_data()
    del lm_results["front"]
    del masks["front"]

    with pytest.raises(ValueError, match="Front view landmark result and mask are strictly required"):
        calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)


def test_missing_optional_views_computes_with_penalties():
    """Missing side or back views must still succeed using fallbacks with reduced confidence."""
    lm_results, masks = _build_synthetic_view_data()
    del lm_results["left"]
    del masks["left"]
    del lm_results["right"]
    del masks["right"]

    m = calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)
    assert len(m.zones) == 12
    # Without side views, chest confidence degrades due to anatomical estimation
    assert m.zones[MeasurementZone.chest].confidence < 0.80


def test_fit_calibration_factors_from_data():
    """Verify least squares calibration fitting in eval/calibrate_k.py."""
    # Create dataset where tape measurements are consistently 5% higher for chest and 10% lower for waist
    data = [
        {"person": "p1", "zone": "chest", "estimated_cm": 100.0, "tape_cm": 105.0},
        {"person": "p2", "zone": "chest", "estimated_cm": 90.0, "tape_cm": 94.5},
        {"person": "p1", "zone": "waist", "estimated_cm": 80.0, "tape_cm": 72.0},
        {"person": "p2", "zone": "waist", "estimated_cm": 70.0, "tape_cm": 63.0},
    ]

    k_factors, metrics = fit_calibration_factors(data)

    # Chest k factor should be ~1.05
    assert abs(k_factors[MeasurementZone.chest] - 1.05) < 0.01
    assert metrics["chest"]["mae_after"] < metrics["chest"]["mae_before"]

    # Waist k factor should be ~0.90
    assert abs(k_factors[MeasurementZone.waist] - 0.90) < 0.01
    assert metrics["waist"]["mae_after"] < metrics["waist"]["mae_before"]

    # Zones without samples should default to 1.0
    assert k_factors[MeasurementZone.hip] == 1.0


def test_leave_one_subject_out_eval_does_not_leak_held_out_subject_into_fit():
    """Verify LOSO eval fits k on training subjects only, excluding the held-out subject's data."""
    # p1 and p2 share a consistent +5% bias; p3 is an outlier with a huge, inconsistent bias.
    data = [
        {"person": "p1", "zone": "chest", "estimated_cm": 100.0, "tape_cm": 105.0},
        {"person": "p2", "zone": "chest", "estimated_cm": 90.0, "tape_cm": 94.5},
        {"person": "p3", "zone": "chest", "estimated_cm": 80.0, "tape_cm": 1000.0},
    ]

    loso = leave_one_subject_out_eval(data)

    assert loso["chest"]["n_subjects"] == 3
    assert loso["chest"]["held_out_sample_count"] == 3
    assert loso["chest"]["train_sample_count"] == 6

    # Independently compute the expected per-fold errors by fitting k on exactly the
    # two training subjects for each fold (never the held-out one), and compare the
    # mean to leave_one_subject_out_eval's result. A mismatch would mean the held-out
    # subject's row leaked into its own fold's k fit.
    rows_by_person = {
        "p1": (100.0, 105.0),
        "p2": (90.0, 94.5),
        "p3": (80.0, 1000.0),
    }
    expected_errors = []
    for held in rows_by_person:
        train_rows = [
            {"person": p, "zone": "chest", "estimated_cm": est, "tape_cm": tape}
            for p, (est, tape) in rows_by_person.items()
            if p != held
        ]
        k_fold, _ = fit_calibration_factors(train_rows)
        held_est, held_tape = rows_by_person[held]
        expected_errors.append(abs(held_est * k_fold[MeasurementZone.chest] - held_tape))

    assert abs(loso["chest"]["held_out_mae"] - round(sum(expected_errors) / len(expected_errors), 2)) < 0.02


def test_leave_one_subject_out_eval_requires_at_least_two_subjects():
    """Zones with data from fewer than 2 subjects cannot be held out and are skipped, not fabricated."""
    data = [
        {"person": "p1", "zone": "chest", "estimated_cm": 100.0, "tape_cm": 105.0},
    ]
    loso = leave_one_subject_out_eval(data)
    assert "chest" not in loso


def test_invalid_user_height_raises_value_error():
    """Height outside human bounds (50-260 cm) must raise ValueError."""
    lm_results, masks = _build_synthetic_view_data()
    with pytest.raises(ValueError, match="Invalid user height"):
        calculate_photo_measurements(lm_results, masks, user_height_cm=30.0)

    with pytest.raises(ValueError, match="Invalid user height"):
        calculate_photo_measurements(lm_results, masks, user_height_cm=270.0)


def test_invalid_cm_per_pixel_scale_raises_value_error():
    """Non-positive or NaN scale must raise ValueError."""
    lm_results, masks = _build_synthetic_view_data()
    lm_results["front"].cm_per_pixel = -0.5
    with pytest.raises(ValueError, match="Invalid front view scale"):
        calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)


def test_anatomical_row_alignment_different_framing():
    """Verify front and side views with different image offsets sample corresponding anatomical planes."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)

    # In the left view, shift subject vertically by +150 pixels (top=250 instead of 100)
    left_mask = np.zeros((1000, 400), dtype=np.uint8)
    left_mask[250:850, 165:235] = 1  # 70 pixels wide depth
    masks["left"] = left_mask
    lm_results["left"].mask_top_row = 250
    lm_results["left"].mask_bottom_row = 849
    lm_results["left"].pixel_height = 600
    lm_results["left"].image_height = 1000
    lm_results["left"].cm_per_pixel = 175.0 / 600.0

    measurements = calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)
    # Measurements must still succeed and calculate realistic chest circumference
    chest_cm = measurements.get_cm(MeasurementZone.chest)
    assert 70.0 <= chest_cm <= 120.0
    assert measurements.zones[MeasurementZone.chest].confidence >= 0.85


def test_fit_calibration_factors_corrupt_data_handled_safely():
    """Verify calibration fitting safely skips invalid, zero, or corrupt CSV rows."""
    data = [
        {"person": "p1", "zone": "chest", "estimated_cm": 0.0, "tape_cm": 105.0},  # 0.0 estimate
        {"person": "p2", "zone": "chest", "estimated_cm": -50.0, "tape_cm": 95.0}, # negative estimate
        {"person": "p3", "zone": "chest", "estimated_cm": "not_a_number", "tape_cm": 95.0}, # string
        {"person": "p4", "zone": "unknown_zone", "estimated_cm": 100.0, "tape_cm": 100.0}, # invalid zone
        {"person": "p5", "zone": "chest", "estimated_cm": 100.0, "tape_cm": 105.0}, # valid
        {"person": "p6", "zone": "chest", "estimated_cm": 100.0, "tape_cm": 105.0}, # valid
    ]

    k_factors, metrics = fit_calibration_factors(data)
    assert abs(k_factors[MeasurementZone.chest] - 1.05) < 0.01
    assert metrics["chest"]["sample_count"] == 2


def test_low_landmark_visibility_applies_confidence_penalty():
    """Verify that low landmark detection confidence degrades overall zone confidence."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)
    lm_results["front"].mean_confidence = 0.65  # low landmark confidence (< 0.70)

    m = calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)
    chest_m = m.zones[MeasurementZone.chest]
    # Base 0.90 - 0.15 visibility penalty = 0.75
    assert chest_m.confidence <= 0.75


def test_missing_landmarks_uses_anatomical_fallbacks_safely():
    """Verify missing arm and shoulder landmarks do not crash and use safe anatomical fallbacks."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)

    # Filter out shoulder and arm landmarks from front and back views
    for v in ["front", "back"]:
        lm_results[v].landmarks = [
            pt for pt in lm_results[v].landmarks
            if "SHOULDER" not in pt.name and "ELBOW" not in pt.name and "WRIST" not in pt.name
        ]

    m = calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)
    assert len(m.zones) == 12

    # Shoulder width and sleeve length should fall back to stature ratios
    sh_m = m.zones[MeasurementZone.shoulder_width]
    assert 30.0 <= sh_m.value_cm <= 50.0
    assert sh_m.confidence <= 0.50

    sl_m = m.zones[MeasurementZone.sleeve_length]
    assert 45.0 <= sl_m.value_cm <= 70.0
    assert sl_m.confidence <= 0.50


def test_empty_mask_row_falls_back_to_anatomical_dimensions_no_zero():
    """Verify corrupted or zero mask rows at chest/waist/hip do not result in zero measurements."""
    lm_results, masks = _build_synthetic_view_data(user_height_cm=175.0)

    # Blank out chest, waist, and hip rows on front mask (e.g. segmentation hole)
    masks["front"][150:500, :] = 0

    m = calculate_photo_measurements(lm_results, masks, user_height_cm=175.0)
    assert len(m.zones) == 12

    for zone in [MeasurementZone.chest, MeasurementZone.waist, MeasurementZone.hip]:
        zone_m = m.zones[zone]
        assert zone_m.value_cm > 0.0
        assert zone_m.confidence <= 0.70  # Mask penalty applied


def test_fit_calibration_factors_comma_and_non_dict_rows():
    """Verify calibration fitting handles European comma decimals and non-dict rows."""
    data = [
        None,  # non-dict
        "raw string row",  # non-dict
        {"person": "p1", "zone": "chest", "estimated_cm": "100,0", "tape_cm": "105,0"},
        {"person": "p2", "zone": "chest", "estimated_cm": "90,0", "tape_cm": "94,5"},
    ]
    k_factors, metrics = fit_calibration_factors(data)
    assert abs(k_factors[MeasurementZone.chest] - 1.05) < 0.01
    assert metrics["chest"]["sample_count"] == 2


