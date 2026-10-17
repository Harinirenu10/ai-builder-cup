"""Loader for canonical SMPL models in services/geometry."""

from pathlib import Path
import numpy as np
import smplx

SMPL_DIR = Path(__file__).resolve().parent / "models" / "smpl"


def load_smpl(gender: str = "neutral", num_betas: int = 10) -> smplx.SMPL:
    """Load a calibrated SMPL body model from converted .npz assets.

    Args:
        gender: One of 'neutral', 'female', 'male'.
        num_betas: Number of shape components to activate (default 10).

    Returns:
        smplx.SMPL PyTorch module ready for forward kinematics and fitting.
    """
    gender_clean = gender.lower().strip()
    if gender_clean not in ("neutral", "female", "male"):
        raise ValueError(f"Invalid gender '{gender}'. Expected 'neutral', 'female', or 'male'.")

    npz_path = SMPL_DIR / f"SMPL_{gender_clean.upper()}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"SMPL model asset not found: {npz_path}. "
            f"Please run services/geometry/convert_smpl_to_npz.py first."
        )

    raw_data = dict(np.load(npz_path, allow_pickle=True))
    data_struct = smplx.body_models.Struct(**raw_data)

    model = smplx.SMPL(
        model_path="",
        data_struct=data_struct,
        gender=gender_clean,
        num_betas=num_betas,
    )
    return model
