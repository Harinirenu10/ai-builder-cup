"""Unit tests for services/geometry/exporter.py (3D GLB Avatar Export).

Verifies:
1. Successful binary .glb export from SMPL vertices and faces.
2. Exact preservation of 6,890 vertices and 13,776 faces on re-import.
3. Grounding alignment (soles resting at Y=0.0 in meters).
4. Export from FitResult structure.
5. Proper rejection of invalid, non-finite, or corrupted meshes.
6. Binary GLTF magic header integrity and validation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from fitcore.schemas import Measurements
from services.geometry.exporter import (
    GLTF_MAGIC,
    export_fit_result_to_glb,
    export_twin_glb,
    validate_glb_bytes,
)
from services.geometry.fitting import fit_smpl_shape
from services.geometry.loader import load_smpl


def test_export_twin_glb_neutral_model(tmp_path: Path):
    """Verify export of neutral SMPL model to a valid, re-parseable .glb file."""
    model = load_smpl("neutral")
    v = model().vertices[0].detach().numpy()
    f = model.faces

    out_file = tmp_path / "twin.glb"
    glb_bytes = export_twin_glb(vertices=v, faces=f, output_path=out_file)

    assert isinstance(glb_bytes, bytes)
    assert len(glb_bytes) > 50_000
    assert glb_bytes[:4] == GLTF_MAGIC
    assert out_file.exists()
    assert out_file.stat().st_size == len(glb_bytes)

    # Re-import and validate
    is_valid, err, meta = validate_glb_bytes(glb_bytes)
    assert is_valid is True, f"GLB validation failed: {err}"
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776
    assert 1.4 <= meta["height_m"] <= 2.1


def test_export_twin_glb_grounded():
    """Verify ground_mesh=True shifts the soles to Y=0.0."""
    model = load_smpl("neutral")
    v = model().vertices[0].detach().numpy()
    f = model.faces

    # Export with ground_mesh=True
    glb_grounded = export_twin_glb(v, f, ground_mesh=True)
    _, _, meta_g = validate_glb_bytes(glb_grounded)

    # Lowest Y should be at 0.0 (or within 1mm due to float rounding)
    assert abs(meta_g["bounds_min"][1] - 0.0) < 1e-3

    # Export with ground_mesh=False
    glb_raw = export_twin_glb(v, f, ground_mesh=False)
    _, _, meta_raw = validate_glb_bytes(glb_raw)
    assert meta_raw["bounds_min"][1] < -0.5  # Raw SMPL has pelvis near 0 and soles near -1.1


def test_export_fit_result_to_glb():
    """Verify direct export from a completed FitResult."""
    fit = fit_smpl_shape(
        measurements=Measurements(zones={}),
        target_height_cm=175.0,
        gender="neutral",
        max_steps=5,
    )

    glb_bytes = export_fit_result_to_glb(fit)
    is_valid, err, meta = validate_glb_bytes(glb_bytes)
    assert is_valid is True, f"Export from FitResult failed: {err}"
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776
    assert abs(meta["height_cm"] - 175.0) < 3.0


def test_export_fit_result_invalid_raises():
    """Verify that exporting an invalid FitResult raises ValueError."""
    fit = fit_smpl_shape(
        measurements=Measurements(zones={}),
        target_height_cm=175.0,
        gender="neutral",
        max_steps=5,
    )
    # Mark fit as invalid
    fit.is_valid = False
    fit.rejection_reason = "Stature error exceeded threshold"

    with pytest.raises(ValueError, match="Cannot export invalid FitResult"):
        export_fit_result_to_glb(fit)


def test_export_twin_glb_invalid_inputs():
    """Verify exceptions are raised on NaN coordinates, wrong dimensions, or bad indices."""
    v_valid = np.zeros((10, 3), dtype=np.float32)
    f_valid = np.array([[0, 1, 2]], dtype=np.int32)

    # 1. NaN in vertices
    v_nan = v_valid.copy()
    v_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Infinite"):
        export_twin_glb(v_nan, f_valid)

    # 2. Wrong shape (N, 2)
    v_2d = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        export_twin_glb(v_2d, f_valid)

    # 3. Face index out of bounds
    f_oob = np.array([[0, 1, 99]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of bounds"):
        export_twin_glb(v_valid, f_oob)

    # 4. Empty arrays
    with pytest.raises(ValueError, match="must not be empty"):
        export_twin_glb(np.zeros((0, 3)), f_valid)


def test_validate_glb_bytes_corrupt():
    """Verify corrupt or truncated byte sequences fail validation safely."""
    # 1. Empty buffer
    ok, err, _ = validate_glb_bytes(b"")
    assert ok is False
    assert "truncated" in err.lower()

    # 2. Corrupt magic
    ok, err, _ = validate_glb_bytes(b"NOT_A_GLTF_HEADER_1234567890")
    assert ok is False
    assert "header magic" in err.lower()
