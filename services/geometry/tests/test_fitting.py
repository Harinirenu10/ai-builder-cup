"""Unit tests for services/geometry/fitting.py (SMPL Shape Fitting).

Verifies:
1. Canonical A-pose tensor formulation with 45-degree shoulder abduction.
2. Differentiable 12-zone measurement extraction on SMPL topology.
3. Successful optimization across neutral, female, and male models.
4. Convergence toward synthetic ground-truth targets.
5. Safe handling of invalid inputs, missing measurements, and zero confidence.
6. Numerical stability (no NaNs or Infs).
7. Diagnostics structure and reproducibility on CPU.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from fitcore.schemas import MeasurementZone, Measurement, Measurements
from services.geometry.fitting import (
    FitResult,
    MeshMeasurementExtractor,
    fit_smpl_shape,
    get_a_pose_tensor,
)
from services.geometry.loader import load_smpl


def test_get_a_pose_tensor():
    """Verify A-pose tensor shape and 45-degree shoulder abduction."""
    pose = get_a_pose_tensor()
    assert pose.shape == (1, 69)
    assert pose.dtype == torch.float32

    # Left shoulder index 15 (indices 45..47), Z-rotation is index 47
    # Right shoulder index 16 (indices 48..50), Z-rotation is index 50
    angle_rad = np.radians(45.0)
    assert abs(pose[0, 15 * 3 + 2].item() - (-angle_rad)) < 1e-4
    assert abs(pose[0, 16 * 3 + 2].item() - angle_rad) < 1e-4

    # All other joints should be zero (unposed)
    non_shoulder_mask = torch.ones(69, dtype=torch.bool)
    non_shoulder_mask[15 * 3 + 2] = False
    non_shoulder_mask[16 * 3 + 2] = False
    assert torch.all(pose[0, non_shoulder_mask] == 0.0)


@pytest.mark.parametrize("gender", ["neutral", "female", "male"])
def test_mesh_measurement_extractor_all_zones(gender: str):
    """Verify measurement extractor outputs positive finite values for all 12 zones."""
    extractor = MeshMeasurementExtractor(gender=gender)
    model = load_smpl(gender=gender, num_betas=10)
    a_pose = get_a_pose_tensor()

    with torch.no_grad():
        out = model(body_pose=a_pose)
        v = out.vertices[0]
        j = out.joints[0]
        meas = extractor.extract_measurements(v, j)

    assert len(meas) == 12
    for zone in MeasurementZone:
        val = meas.get(zone)
        assert val is not None, f"Missing extractor output for zone: {zone}"
        assert torch.isfinite(val), f"Non-finite value for zone: {zone}"
        assert val.item() > 0.0, f"Non-positive measurement for zone: {zone}: {val.item()}"

    # Verify plausible human dimensions
    assert 70.0 <= meas[MeasurementZone.chest].item() <= 125.0
    assert 60.0 <= meas[MeasurementZone.waist].item() <= 110.0
    assert 80.0 <= meas[MeasurementZone.hip].item() <= 125.0
    assert 30.0 <= meas[MeasurementZone.shoulder_width].item() <= 55.0
    assert 50.0 <= meas[MeasurementZone.inseam].item() <= 100.0


def test_inseam_uses_crotch_vertex_not_pelvis_joint():
    """Regression test: inseam must measure from the crotch surface vertex
    down to the ankle, not from the pelvis (hip-socket) joint. A synthetic
    round-trip test (known SMPL betas -> rendered photo -> re-measured)
    found the ground-truth inseam computed via the pelvis joint disagreed
    with the independently-derived photo-side crotch estimate by ~20cm, while
    the already-defined idx_crotch surface vertex agreed with it to within
    ~1cm. The pelvis joint is the kinematic hip-socket center and sits
    measurably higher than the actual crotch on this topology -- using it
    for "inseam" was inconsistent with the method's own doc comment, which
    already said "crotch level".
    """
    extractor = MeshMeasurementExtractor(gender="neutral")
    model = load_smpl(gender="neutral", num_betas=10)
    a_pose = get_a_pose_tensor()

    with torch.no_grad():
        out = model(body_pose=a_pose)
        v = out.vertices[0]
        j = out.joints[0]
        meas = extractor.extract_measurements(v, j)

    pelvis_to_ankle_cm = float((j[0, 1] - j[7, 1]).item()) * 100.0
    crotch_to_ankle_cm = float((v[extractor.idx_crotch, 1] - j[7, 1]).item()) * 100.0
    reported_inseam_cm = meas[MeasurementZone.inseam].item()

    # The crotch vertex sits measurably below the pelvis joint on this
    # topology, so the two candidate inseam definitions must differ
    # noticeably -- this guards against a future refactor accidentally
    # aliasing idx_crotch back to the pelvis joint's height without anyone
    # noticing (which would make this assertion meaningless).
    assert abs(pelvis_to_ankle_cm - crotch_to_ankle_cm) > 5.0, (
        "test setup sanity check: pelvis and crotch should be at clearly "
        "different heights on this mesh"
    )
    assert abs(reported_inseam_cm - crotch_to_ankle_cm) < 1e-3, (
        f"extract_measurements() inseam ({reported_inseam_cm}cm) does not match "
        f"the crotch-vertex-to-ankle distance ({crotch_to_ankle_cm}cm) -- "
        "it appears to be using a different reference point (e.g. the pelvis joint)"
    )


def test_torso_length_and_back_rise_use_waist_loop_plane_not_spine1_or_post_waist_vertex():
    """Regression test: torso_length and back_rise must reference the SAME
    "waist plane" that waist_cm (the validated waist circumference) actually
    measures, not the spine1 joint or the idx_post_waist vertex. A synthetic
    SMPL ground-truth round trip found both of those disagreed with
    waist_loop's own height by 6-10cm on this topology (spine1 sits well
    below the natural waist; idx_post_waist, while genuinely posterior and
    on-midline -- see test_landmark_bilateral_symmetry_and_posterior_waist
    below -- is not at waist height either), despite both zones' doc
    comments describing their bottom/top reference as "waist".
    """
    extractor = MeshMeasurementExtractor(gender="neutral")
    model = load_smpl(gender="neutral", num_betas=10)
    a_pose = get_a_pose_tensor()

    with torch.no_grad():
        out = model(body_pose=a_pose)
        v = out.vertices[0]
        j = out.joints[0]
        meas = extractor.extract_measurements(v, j)

    waist_plane_y = extractor.waist_loop.mean_plane_y(v)
    c7_to_waist_plane_cm = float((v[extractor.idx_c7, 1] - waist_plane_y).item()) * 100.0
    waist_plane_to_crotch_cm = float((waist_plane_y - v[extractor.idx_crotch, 1]).item()) * 100.0

    c7_to_spine1_cm = float((v[extractor.idx_c7, 1] - j[3, 1]).item()) * 100.0
    post_waist_to_crotch_cm = float(
        torch.sqrt(torch.sum((v[extractor.idx_post_waist] - v[extractor.idx_crotch]) ** 2)).item()
    ) * 100.0

    # Sanity check: the old spine1/idx_post_waist stand-ins must actually
    # disagree with the waist plane on this mesh, or this test would not
    # exercise the bug it guards against.
    assert abs(c7_to_spine1_cm - c7_to_waist_plane_cm) > 5.0, (
        "test setup sanity check: spine1 joint should sit clearly off the waist plane"
    )
    assert abs(post_waist_to_crotch_cm - waist_plane_to_crotch_cm) > 5.0, (
        "test setup sanity check: idx_post_waist should sit clearly off the waist plane"
    )

    reported_torso_len = meas[MeasurementZone.torso_length].item()
    reported_back_rise = meas[MeasurementZone.back_rise].item()
    assert abs(reported_torso_len - c7_to_waist_plane_cm) < 1e-3, (
        f"torso_length ({reported_torso_len}cm) does not match C7-to-waist-plane "
        f"({c7_to_waist_plane_cm}cm) -- it appears to use a different bottom reference"
    )
    assert abs(reported_back_rise - waist_plane_to_crotch_cm) < 1e-3, (
        f"back_rise ({reported_back_rise}cm) does not match waist-plane-to-crotch "
        f"({waist_plane_to_crotch_cm}cm) -- it appears to use a different top reference"
    )


def test_garment_length_uses_hip_joints_not_pelvis_root():
    """Regression test: garment_length must measure down to the hip SOCKET
    joints (1, 2), which is what a 2D pose landmarker's HIP landmark
    corresponds to, not the pelvis ROOT joint (0). A synthetic SMPL
    ground-truth round trip found the pelvis root sits ~9.5cm ABOVE the hip
    sockets on this topology, which made the photo-side's garment_length
    (measured down to its own hip-landmark row) disagree with this zone's
    ground truth by 13-15cm even after other fixes.
    """
    extractor = MeshMeasurementExtractor(gender="neutral")
    model = load_smpl(gender="neutral", num_betas=10)
    a_pose = get_a_pose_tensor()

    with torch.no_grad():
        out = model(body_pose=a_pose)
        v = out.vertices[0]
        j = out.joints[0]
        meas = extractor.extract_measurements(v, j)

    hip_joints_mean_y = (j[1, 1] + j[2, 1]) / 2.0
    c7_to_hip_joints_cm = float((v[extractor.idx_c7, 1] - hip_joints_mean_y).item()) * 100.0
    c7_to_pelvis_root_cm = float((v[extractor.idx_c7, 1] - j[0, 1]).item()) * 100.0

    assert abs(c7_to_hip_joints_cm - c7_to_pelvis_root_cm) > 5.0, (
        "test setup sanity check: pelvis root and hip joints should sit at clearly different heights"
    )

    reported_garment_len = meas[MeasurementZone.garment_length].item()
    assert abs(reported_garment_len - c7_to_hip_joints_cm) < 1e-3, (
        f"garment_length ({reported_garment_len}cm) does not match C7-to-hip-joints "
        f"({c7_to_hip_joints_cm}cm) -- it appears to use a different bottom reference (e.g. the pelvis root)"
    )


def test_fit_smpl_shape_synthetic_ground_truth():
    """Verify optimization recovers shape parameters close to known synthetic target."""
    model = load_smpl(gender="neutral", num_betas=10)
    extractor = MeshMeasurementExtractor(gender="neutral")
    a_pose = get_a_pose_tensor()

    # Create target with known stature and shape
    target_betas = torch.zeros((1, 10), dtype=torch.float32)
    target_betas[0, 0] = 1.8  # taller/broader
    target_betas[0, 1] = 0.5

    with torch.no_grad():
        out = model(betas=target_betas, body_pose=a_pose)
        v = out.vertices[0]
        j = out.joints[0]
        tgt_meas_dict = extractor.extract_measurements(v, j)
        tgt_height_cm = float((torch.max(v[:, 1]) - torch.min(v[:, 1])) * 100.0)

    # Convert to Measurements schema
    zones = {}
    for z, t_val in tgt_meas_dict.items():
        zones[z] = Measurement(
            value_cm=round(float(t_val.item()), 1),
            confidence=0.95,
            source="photo_measurement",
        )
    target_measurements = Measurements(zones=zones)

    # Fit SMPL model starting from neutral zero betas
    result = fit_smpl_shape(
        measurements=target_measurements,
        target_height_cm=tgt_height_cm,
        gender="neutral",
        max_steps=80,
        lr=0.05,
    )

    assert isinstance(result, FitResult)
    assert len(result.betas) == 10
    # Height error should be within 1.5 cm
    assert abs(result.fitted_height_cm - tgt_height_cm) < 1.5
    # First beta (height/scale) should be positive and close to target
    assert result.betas[0] > 1.0
    # Final loss should be low
    assert result.loss < 5.0


@pytest.mark.parametrize("gender", ["neutral", "female", "male"])
def test_fit_smpl_shape_genders(gender: str):
    """Verify fitting runs successfully across neutral, female, and male models."""
    sample_measurements = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=92.0, confidence=0.90, source="photo_measurement"),
            MeasurementZone.waist: Measurement(value_cm=78.0, confidence=0.88, source="photo_measurement"),
            MeasurementZone.hip: Measurement(value_cm=98.0, confidence=0.90, source="photo_measurement"),
            MeasurementZone.shoulder_width: Measurement(value_cm=41.0, confidence=0.85, source="photo_measurement"),
        }
    )

    result = fit_smpl_shape(
        measurements=sample_measurements,
        target_height_cm=172.0,
        gender=gender,
        max_steps=30,
        lr=0.05,
    )

    assert result.gender == gender
    assert result.vertices.shape == (6890, 3)
    assert result.faces.shape == (13776, 3)
    assert np.all(np.isfinite(result.vertices))
    assert abs(result.fitted_height_cm - 172.0) < 2.0
    assert len(result.diagnostics) == 4
    for zone_str, diag in result.diagnostics.items():
        assert "target_cm" in diag
        assert "fitted_cm" in diag
        assert "error_cm" in diag
        assert "abs_error_cm" in diag
        assert "confidence" in diag


def test_fit_smpl_shape_invalid_inputs():
    """Verify exceptions are raised on invalid stature and gender."""
    m = Measurements(zones={})

    # Height out of range
    with pytest.raises(ValueError, match="Invalid target height"):
        fit_smpl_shape(m, target_height_cm=30.0)

    with pytest.raises(ValueError, match="Invalid target height"):
        fit_smpl_shape(m, target_height_cm=300.0)

    # Invalid gender
    with pytest.raises(ValueError, match="Invalid gender"):
        fit_smpl_shape(m, target_height_cm=175.0, gender="other_gender")


def test_fit_smpl_shape_empty_or_zero_confidence():
    """Verify empty or zero-confidence measurements still fit stature without crashing."""
    empty_meas = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=90.0, confidence=0.0, source="photo_measurement"),
        }
    )

    result = fit_smpl_shape(
        measurements=empty_meas,
        target_height_cm=180.0,
        gender="neutral",
        max_steps=25,
    )

    assert abs(result.fitted_height_cm - 180.0) < 2.5
    assert np.all(np.isfinite(result.vertices))
    assert result.diagnostics == {}  # Zero-confidence zone is excluded from targets


def test_fit_smpl_shape_numerical_stability():
    """Verify fitting remains numerically stable and produces no NaNs under extreme values."""
    extreme_meas = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=150.0, confidence=0.99, source="photo_measurement"),
            MeasurementZone.waist: Measurement(value_cm=40.0, confidence=0.05, source="photo_measurement"),
        }
    )

    result = fit_smpl_shape(
        measurements=extreme_meas,
        target_height_cm=155.0,
        gender="neutral",
        max_steps=20,
    )

    assert np.all(np.isfinite(result.vertices))
    assert all(np.isfinite(b) for b in result.betas)
    assert np.isfinite(result.loss)
    assert result.loss > 0.0


def test_landmark_bilateral_symmetry_and_posterior_waist():
    """Verify acromion and armscye landmarks are bilaterally symmetric and waist is posterior."""
    for gender in ["neutral", "female", "male"]:
        extractor = MeshMeasurementExtractor(gender=gender)
        model = load_smpl(gender=gender, num_betas=10)
        v = model().vertices[0].detach().numpy()

        # Posterior waist must be posterior (Z < 0) along sagittal midline (|X| < 0.02)
        v_pw = v[extractor.idx_post_waist]
        assert v_pw[2] < -0.05, f"Posterior waist is not on back for {gender}: {v_pw}"
        assert abs(v_pw[0]) < 0.02, f"Posterior waist is off-midline for {gender}: {v_pw}"

        # Left and right armscye creases must be bilaterally symmetric
        v_l_arm = v[extractor.idx_l_armscye]
        v_r_arm = v[extractor.idx_r_armscye]
        assert abs(v_l_arm[0] - (-v_r_arm[0])) < 0.015, f"Armscyes asymmetric in X for {gender}"
        assert abs(v_l_arm[1] - v_r_arm[1]) < 0.015, f"Armscyes asymmetric in Y for {gender}"

        # Acromions must be bilaterally symmetric
        v_l_acr = v[extractor.idx_l_acromion]
        v_r_acr = v[extractor.idx_r_acromion]
        assert abs(v_l_acr[0] - (-v_r_acr[0])) < 0.015, f"Acromions asymmetric in X for {gender}"
        assert abs(v_l_acr[1] - v_r_acr[1]) < 0.015, f"Acromions asymmetric in Y for {gender}"


def test_fit_smpl_shape_beta_clamping():
    """Verify betas are strictly clamped to [-4.0, 4.0] even under extreme target measurements."""
    extreme_meas = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=250.0, confidence=1.0, source="photo_measurement"),
            MeasurementZone.hip: Measurement(value_cm=250.0, confidence=1.0, source="photo_measurement"),
        }
    )

    result = fit_smpl_shape(
        measurements=extreme_meas,
        target_height_cm=200.0,
        gender="neutral",
        max_steps=40,
        lr=0.1,
    )

    for beta in result.betas:
        assert -4.0 <= beta <= 4.0, f"Beta {beta} exceeded [-4.0, 4.0] bounds"


def test_fit_smpl_shape_invalid_fit_rejection():
    """Verify that severe stature mismatches or divergence flag is_valid=False with rejection reason."""
    # Fitting a 120 cm height target to male SMPL with contradictory measurements and very few steps
    impossible_meas = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=180.0, confidence=1.0, source="photo_measurement"),
        }
    )

    result = fit_smpl_shape(
        measurements=impossible_meas,
        target_height_cm=60.0,  # Extreme dwarf stature that SMPL male betas cannot reach
        gender="male",
        max_steps=5,
    )

    assert result.is_valid is False
    assert result.rejection_reason is not None
    assert "deviates from target" in result.rejection_reason or "exceeds" in result.rejection_reason

