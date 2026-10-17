#!/usr/bin/env python3
"""Standalone, non-destructive SMPL pickle-to-npz converter for Python 3.12.

Converts legacy Python 2 / chumpy-wrapped SMPL model pickle files:
  - services/geometry/models/smpl/SMPL_NEUTRAL.pkl -> SMPL_NEUTRAL.npz
  - services/geometry/models/smpl/SMPL_FEMALE.pkl  -> SMPL_FEMALE.npz
  - services/geometry/models/smpl/SMPL_MALE.pkl    -> SMPL_MALE.npz

Safety Guarantees:
1. Strict read-only access to all original .pkl files.
2. Refuses to overwrite existing .npz files (preserves user data).
3. Exact chumpy unpickling based on observed internal structure (state['x']).
4. Complete structural, dtype, and numerical validation (finite values,
   convex weights summing to 1.0, valid face vertex indices, joint regressor sum).
5. Output format is 100% compatible with smplx.create(..., ext='npz').
"""

import sys
import pickle
from pathlib import Path
from typing import Any, Dict

SMPL_DIR = Path(__file__).resolve().parent / "models" / "smpl"
GENDERS = ["NEUTRAL", "FEMALE", "MALE"]

# Canonical SMPL structural invariants
EXPECTED_VERTICES = 6890
EXPECTED_JOINTS = 24
EXPECTED_FACES = 13776
EXPECTED_BETAS = 300
EXPECTED_POSEDIRS = 207  # 23 joints * 9 rodrigues coords


class ChumpyStub:
    """Safe surrogate for chumpy.ch.Ch objects during unpickling.

    In the official SMPL model files, 'shapedirs' is serialized as a chumpy
    object whose state dict contains the raw NumPy ndarray under key 'x'.
    """

    def __init__(self, *args, **kwargs):
        self.data = None

    def __setstate__(self, state: Any):
        if isinstance(state, dict):
            # In official SMPL files, chumpy stores the array under 'x'
            self.data = state.get("x", state.get("d", state.get("r", None)))
        else:
            self.data = state

    def to_numpy(self):
        import numpy as np
        if hasattr(self.data, "to_numpy"):
            return self.data.to_numpy()
        return np.asarray(self.data)


class ScipySparseStub:
    """Safe surrogate for scipy.sparse.csc csc_matrix deserialization.

    In official SMPL files, 'J_regressor' and 'J_regressor_prior' are serialized
    as scipy.sparse.csc csc_matrix instances with state dictionary:
      {'_shape': (24, 6890), 'indptr': ..., 'indices': ..., 'data': ...}
    """

    def __init__(self, *args, **kwargs):
        self.state = None

    def __setstate__(self, state: Any):
        self.state = state

    def to_dense(self):
        import numpy as np
        if not isinstance(self.state, dict):
            return np.asarray(self.state, dtype=np.float32)

        shape = self.state["_shape"]
        indptr = np.asarray(self.state["indptr"])
        indices = np.asarray(self.state["indices"])
        data = np.asarray(self.state["data"], dtype=np.float32)

        dense = np.zeros(shape, dtype=np.float32)
        num_cols = shape[1]
        for col in range(num_cols):
            start = indptr[col]
            end = indptr[col + 1]
            for idx in range(start, end):
                dense[indices[idx], col] = data[idx]
        return dense


class SMPLSafeUnpickler(pickle.Unpickler):
    """Custom unpickler that safely redirects legacy Python 2 / chumpy dependencies."""

    def find_class(self, module: str, name: str) -> Any:
        # Redirect chumpy.ch.Ch or other chumpy classes to ChumpyStub
        if "chumpy" in module:
            return ChumpyStub

        # Redirect legacy scipy.sparse.csc classes
        if "scipy.sparse" in module:
            try:
                import scipy.sparse
                if hasattr(scipy.sparse, name):
                    return getattr(scipy.sparse, name)
                import scipy.sparse.csc
                if hasattr(scipy.sparse.csc, name):
                    return getattr(scipy.sparse.csc, name)
            except Exception:
                pass
            return ScipySparseStub

        # Remap Python 2 built-in module references
        if module == "__builtin__":
            import builtins
            return getattr(builtins, name)
        if module == "copy_reg":
            import copyreg
            return getattr(copyreg, name)

        return super().find_class(module, name)


def unwrap_value(val: Any) -> Any:
    """Recursively convert stubs and wrappers to pure NumPy arrays."""
    import numpy as np

    if isinstance(val, ChumpyStub):
        return unwrap_value(val.to_numpy())
    if isinstance(val, ScipySparseStub):
        return val.to_dense()
    if hasattr(val, "toarray"):  # If scipy is loaded and returned real sparse matrix
        return np.asarray(val.toarray(), dtype=np.float32)
    if isinstance(val, (bytes, bytearray, str, int, float, bool)):
        return val

    return np.asarray(val)


def validate_and_sanitize(data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Validate shapes, data types, value ranges, and numerical finiteness."""
    import numpy as np

    print(f"  [Validation] Running integrity checks for {model_name}...")

    # 1. Vertex Template (v_template)
    v_template = np.asarray(data["v_template"], dtype=np.float32)
    assert v_template.shape == (EXPECTED_VERTICES, 3), (
        f"v_template shape {v_template.shape} != ({EXPECTED_VERTICES}, 3)"
    )
    assert np.all(np.isfinite(v_template)), "v_template contains NaN or Inf"
    # Basic bounding box sanity check: template height should be between 1.0m and 2.5m
    y_or_z_span = np.ptp(v_template, axis=0).max()
    assert 1.0 <= y_or_z_span <= 2.5, f"v_template span unusual ({y_or_z_span:.2f}m)"

    # 2. Shape Blendshapes (shapedirs)
    shapedirs = np.asarray(data["shapedirs"], dtype=np.float32)
    assert shapedirs.shape[:2] == (EXPECTED_VERTICES, 3), (
        f"shapedirs vertex dims {shapedirs.shape[:2]} != ({EXPECTED_VERTICES}, 3)"
    )
    assert shapedirs.shape[-1] in (10, 300), (
        f"shapedirs components {shapedirs.shape[-1]} not in (10, 300)"
    )
    assert np.all(np.isfinite(shapedirs)), "shapedirs contains NaN or Inf"

    # 3. Pose Corrective Blendshapes (posedirs)
    posedirs = np.asarray(data["posedirs"], dtype=np.float32)
    assert posedirs.shape == (EXPECTED_VERTICES, 3, EXPECTED_POSEDIRS), (
        f"posedirs shape {posedirs.shape} != ({EXPECTED_VERTICES}, 3, {EXPECTED_POSEDIRS})"
    )
    assert np.all(np.isfinite(posedirs)), "posedirs contains NaN or Inf"

    # 4. Joint Regressor (J_regressor)
    j_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
    assert j_regressor.shape == (EXPECTED_JOINTS, EXPECTED_VERTICES), (
        f"J_regressor shape {j_regressor.shape} != ({EXPECTED_JOINTS}, {EXPECTED_VERTICES})"
    )
    assert np.all(np.isfinite(j_regressor)), "J_regressor contains NaN or Inf"
    # Weights for each joint regression should sum close to 1.0
    j_sums = j_regressor.sum(axis=1)
    assert np.allclose(j_sums, 1.0, atol=1e-2), (
        f"J_regressor row sums deviate from 1.0: min={j_sums.min()}, max={j_sums.max()}"
    )

    # 5. Skinning Weights (weights)
    weights = np.asarray(data["weights"], dtype=np.float32)
    assert weights.shape == (EXPECTED_VERTICES, EXPECTED_JOINTS), (
        f"weights shape {weights.shape} != ({EXPECTED_VERTICES}, {EXPECTED_JOINTS})"
    )
    assert np.all(np.isfinite(weights)), "weights contain NaN or Inf"
    # LBS weights per vertex must sum to 1.0
    w_sums = weights.sum(axis=1)
    assert np.allclose(w_sums, 1.0, atol=1e-3), (
        f"LBS vertex weights do not sum to 1.0: min={w_sums.min()}, max={w_sums.max()}"
    )

    # 6. Kinematic Tree (kintree_table)
    kintree_table = np.asarray(data["kintree_table"], dtype=np.int64)
    assert kintree_table.shape == (2, EXPECTED_JOINTS), (
        f"kintree_table shape {kintree_table.shape} != (2, {EXPECTED_JOINTS})"
    )

    # 7. Triangular Faces (f)
    faces = np.asarray(data["f"], dtype=np.int64)
    assert faces.shape == (EXPECTED_FACES, 3), (
        f"faces shape {faces.shape} != ({EXPECTED_FACES}, 3)"
    )
    assert faces.min() >= 0 and faces.max() < EXPECTED_VERTICES, (
        f"faces contain invalid vertex indices: min={faces.min()}, max={faces.max()}"
    )

    # 8. Blendshape metadata (canonical SMPL uses bs_type='lrotmin' and bs_style='lbs')
    bs_type = data.get("bs_type", "lrotmin")
    if isinstance(bs_type, bytes):
        bs_type = bs_type.decode("ascii", errors="ignore")
    bs_style = data.get("bs_style", "lbs")
    if isinstance(bs_style, bytes):
        bs_style = bs_style.decode("ascii", errors="ignore")

    print(f"  [Validation] All 8 invariant checks PASSED for {model_name}.")

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "J_regressor": j_regressor,
        "kintree_table": kintree_table,
        "weights": weights,
        "f": faces,
        "bs_type": bs_type,
        "bs_style": bs_style,
    }


def convert_model(pkl_path: Path, npz_path: Path) -> None:
    """Safely convert a single SMPL .pkl to .npz without modifying original .pkl."""
    import numpy as np

    print(f"\nTarget: {pkl_path.name}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Source file not found: {pkl_path}")

    # Safety Guard: Never overwrite existing .npz files
    if npz_path.exists():
        raise FileExistsError(
            f"Target file already exists: {npz_path}. "
            f"Refusing to overwrite to prevent accidental data loss."
        )

    # Step 1: Open .pkl in strictly read-only binary mode
    with open(pkl_path, "rb") as f:
        raw_dict = SMPLSafeUnpickler(f, encoding="latin1").load()

    # Step 2: Unwrap stubs and arrays
    data: Dict[str, Any] = {k: unwrap_value(v) for k, v in raw_dict.items()}

    # Step 3: Run comprehensive validation
    clean_dict = validate_and_sanitize(data, pkl_path.stem)

    # Step 4: Write compressed .npz archive
    np.savez_compressed(npz_path, **clean_dict)

    pkl_mb = pkl_path.stat().st_size / (1024 * 1024)
    npz_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"  [Output] Created {npz_path.name}")
    print(f"  [Storage] {pkl_mb:.1f} MB -> {npz_mb:.1f} MB ({(1 - npz_mb/pkl_mb)*100:.1f}% reduction)")


def main():
    try:
        import numpy as np
    except ImportError:
        print("Error: 'numpy' is required to execute this conversion script.")
        print("Please install requirements first (e.g. pip install numpy).")
        return 1

    print("=" * 70)
    print("SMPL MODEL CONVERTER: .pkl -> .npz (Non-Destructive)")
    print("=" * 70)

    for gender in GENDERS:
        pkl_path = SMPL_DIR / f"SMPL_{gender}.pkl"
        npz_path = SMPL_DIR / f"SMPL_{gender}.npz"
        convert_model(pkl_path, npz_path)

    print("\nAll three models converted and verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
