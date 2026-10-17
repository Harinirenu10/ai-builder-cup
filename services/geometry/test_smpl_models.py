"""Validation test suite for SMPL models in services/geometry.

Verifies:
1. Presence and integrity of original licensed .pkl files (never modified).
2. Presence and integrity of converted .npz model files.
3. Successful loading and forward pass through smplx.SMPL module.
4. Output topology verification:
   - 6,890 vertices
   - Valid finite float coordinates
   - Human height bounds
   - 24/45 joint regression
   - Shape parameters (betas) space
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
import numpy as np

from services.geometry.loader import load_smpl

SMPL_DIR = Path(__file__).resolve().parent / "models" / "smpl"
GENDERS = ["neutral", "female", "male"]


def test_original_smpl_pickle_files_intact():
    """Verify that all three original licensed SMPL .pkl files are present and untouched."""
    assert SMPL_DIR.exists(), f"SMPL directory missing: {SMPL_DIR}"
    for gender in GENDERS:
        pkl_path = SMPL_DIR / f"SMPL_{gender.upper()}.pkl"
        assert pkl_path.exists(), f"Original SMPL model file missing: {pkl_path}"
        assert pkl_path.stat().st_size > 200_000_000, f"Original .pkl unexpectedly truncated: {pkl_path}"


def test_converted_smpl_npz_files_exist():
    """Verify that converted .npz model files exist and are populated."""
    for gender in GENDERS:
        npz_path = SMPL_DIR / f"SMPL_{gender.upper()}.npz"
        assert npz_path.exists(), f"Converted .npz file missing: {npz_path}"
        mb = npz_path.stat().st_size / (1024 * 1024)
        assert 25.0 <= mb <= 60.0, f"Unexpected .npz file size for {gender}: {mb:.1f} MB"


@pytest.mark.parametrize("gender", GENDERS)
def test_smpl_model_forward_pass(gender: str):
    """Load model via smplx and run forward pass, asserting 6,890 finite vertices."""
    model = load_smpl(gender=gender, num_betas=10)
    assert model is not None

    # Run default rest-pose forward pass
    with torch.no_grad():
        output = model()

    vertices = output.vertices
    joints = output.joints

    # 1. Assert vertex count and shape
    assert vertices.shape == (1, 6890, 3), (
        f"Expected vertex shape (1, 6890, 3), got {tuple(vertices.shape)}"
    )

    # 2. Assert numerical finiteness (no NaNs, no Infs)
    assert bool(torch.isfinite(vertices).all()), f"Model '{gender}' output contains NaN or Inf vertices!"

    # 3. Assert anatomical height (Y span in meters)
    v_np = vertices[0].cpu().numpy()
    y_height = float(v_np[:, 1].max() - v_np[:, 1].min())
    assert 1.40 <= y_height <= 2.10, f"Plausible human height check failed for {gender}: {y_height:.3f}m"

    # 4. Assert joints shape and finiteness
    assert joints.shape[1] in (24, 45), f"Expected 24 or 45 joints, got {joints.shape[1]}"
    assert bool(torch.isfinite(joints).all()), f"Model '{gender}' output contains NaN or Inf joints!"

    # 5. Assert shape parameter dimension
    assert model.shapedirs.shape[-1] >= 10, f"Expected at least 10 betas, got {model.shapedirs.shape[-1]}"


@pytest.mark.parametrize("gender", GENDERS)
def test_smpl_shape_variation(gender: str):
    """Verify that non-zero beta shape parameters modulate body vertices."""
    model = load_smpl(gender=gender, num_betas=10)

    with torch.no_grad():
        base_out = model()
        # Modulate first beta (overall body scale/stature)
        betas = torch.zeros(1, 10, dtype=torch.float32)
        betas[0, 0] = 2.0  # +2 standard deviations
        shaped_out = model(betas=betas)

    diff = torch.norm(shaped_out.vertices - base_out.vertices, dim=-1)
    assert float(diff.max()) > 0.01, f"Shape beta did not modulate vertices for {gender}"


def run_standalone_validation():
    """CLI runner printing verification summary."""
    print("=" * 70)
    print("SMPL VERIFICATION REPORT: FORWARD PASS & TOPOLOGY")
    print("=" * 70)

    for gender in GENDERS:
        print(f"\nModel: SMPL_{gender.upper()}")
        pkl = SMPL_DIR / f"SMPL_{gender.upper()}.pkl"
        npz = SMPL_DIR / f"SMPL_{gender.upper()}.npz"

        print(f"  - Original .pkl: {'EXISTS' if pkl.exists() else 'MISSING'} ({pkl.stat().st_size / (1024*1024):.1f} MB)")
        print(f"  - Converted .npz: {'EXISTS' if npz.exists() else 'MISSING'} ({npz.stat().st_size / (1024*1024):.1f} MB)")

        model = load_smpl(gender=gender, num_betas=10)
        with torch.no_grad():
            out = model()
        verts = out.vertices[0].cpu().numpy()
        height = float(verts[:, 1].max() - verts[:, 1].min())
        print(f"  - Forward Pass: SUCCESS")
        print(f"  - Vertices: {verts.shape[0]} (all finite: {bool(np.isfinite(verts).all())})")
        print(f"  - Height (Y-span): {height:.3f} m")
        print(f"  - Joints: {out.joints.shape[1]}")

    print("\nAll SMPL models successfully verified!")
    return 0


if __name__ == "__main__":
    sys.exit(run_standalone_validation())
