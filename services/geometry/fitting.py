"""SMPL 3D body shape fitting pipeline from measurements and stature.

Optimizes the 10 SMPL shape parameters (betas) using PyTorch Adam on CPU
to match confidence-weighted photo-derived body measurements and user height.

Loss formulation:
    L = L_measurements + L_height + L_regularization
Where:
    L_measurements = sum(w_z * (mesh_cm_z - target_cm_z)^2) / sum(w_z)
    L_height       = (mesh_height_cm - target_height_cm)^2
    L_regularization = 0.01 * ||betas||^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import torch

from fitcore.schemas import MeasurementZone, Measurements
from services.geometry.loader import load_smpl


@dataclass
class FitResult:
    """Structured result of SMPL shape optimization."""
    gender: str
    betas: list[float]
    vertices: np.ndarray  # (6890, 3) float32 in meters
    faces: np.ndarray  # (13776, 3) int32
    fitted_height_cm: float
    target_height_cm: float
    loss: float
    steps: int
    converged: bool
    is_valid: bool = True
    rejection_reason: str | None = None
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    mesh_measurements: dict[MeasurementZone, float] = field(default_factory=dict)


def get_a_pose_tensor(device: torch.device | str = "cpu") -> torch.Tensor:
    """Create canonical A-pose tensor with shoulders abducted ~45 degrees.

    SMPL joint 16 is left_shoulder (body_pose index 15: indices 45..47)
    SMPL joint 17 is right_shoulder (body_pose index 16: indices 48..50)
    Rotation around Z-axis abducts arms downwards into A-pose.
    """
    body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)
    angle = math.radians(45.0)
    # Left shoulder down around Z-axis
    body_pose[0, 15 * 3 + 2] = -angle
    # Right shoulder down around Z-axis
    body_pose[0, 16 * 3 + 2] = angle
    return body_pose


class CircumferenceLoop:
    """Precomputed planar triangle-slice loop on fixed SMPL topology."""

    def __init__(
        self,
        idx_a1: torch.Tensor,
        idx_b1: torch.Tensor,
        t1: torch.Tensor,
        idx_a2: torch.Tensor,
        idx_b2: torch.Tensor,
        t2: torch.Tensor,
    ):
        self.idx_a1 = idx_a1
        self.idx_b1 = idx_b1
        self.t1 = t1
        self.idx_a2 = idx_a2
        self.idx_b2 = idx_b2
        self.t2 = t2

    def compute_perimeter_cm(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute differentiable planar loop perimeter in centimeters."""
        p1 = (1.0 - self.t1) * vertices[self.idx_a1] + self.t1 * vertices[self.idx_b1]
        p2 = (1.0 - self.t2) * vertices[self.idx_a2] + self.t2 * vertices[self.idx_b2]
        diff = p1 - p2
        # Epsilon ensures strictly non-zero gradient at origin
        seg_lens = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)
        return torch.sum(seg_lens) * 100.0


class MeshMeasurementExtractor:
    """Differentiable 12-zone measurement extractor for SMPL models."""

    def __init__(self, gender: str = "neutral", device: torch.device | str = "cpu"):
        self.gender = gender
        self.device = device
        self._init_topology()

    def _init_topology(self) -> None:
        """Analyze template mesh topology to build loops and landmark indices."""
        model = load_smpl(gender=self.gender, num_betas=10)
        with torch.no_grad():
            out = model()
            v = out.vertices[0].cpu().numpy()
            f = model.faces.astype(np.int64)

        raw_weights = model.lbs_weights.cpu().numpy()
        part_indices = np.argmax(raw_weights, axis=-1)

        # 1. Torso faces (exclude arm joints: 16, 17, 18, 19, 20, 21, 22, 23)
        arm_joints = {16, 17, 18, 19, 20, 21, 22, 23}
        face_is_arm = np.any(np.isin(part_indices[f], list(arm_joints)), axis=1)
        torso_faces = f[~face_is_arm]

        # 2. Left leg faces (joints: 1, 4, 7, 10)
        left_leg_joints = {1, 4, 7, 10}
        face_is_left_leg = np.all(np.isin(part_indices[f], list(left_leg_joints)), axis=1)
        left_leg_faces = f[face_is_left_leg]

        # 3. Neck faces
        neck_joints = {12, 13, 14, 9}
        face_is_neck = np.any(np.isin(part_indices[f], list(neck_joints)), axis=1) & ~face_is_arm
        v_midline = np.abs(v[:, 0]) < 0.12
        neck_faces = f[face_is_neck & np.all(v_midline[f], axis=1)]

        # Precompute circumference loops
        self.chest_loop = self._build_loop(v, torso_faces, y_plane=0.15)
        self.waist_loop = self._build_loop(v, torso_faces, y_plane=-0.05)
        self.hip_loop = self._build_loop(v, torso_faces, y_plane=-0.22)
        self.neck_loop = self._build_loop(v, neck_faces, y_plane=0.27)
        self.thigh_loop = self._build_loop(v, left_leg_faces, y_plane=-0.42)

        # Predefined landmark vertices on SMPL topology
        self.idx_l_acromion = 1829
        self.idx_r_acromion = 5290
        self.idx_c7 = 1301
        self.idx_crotch = 4419
        self.idx_l_armscye = 1878
        self.idx_r_armscye = 5337
        self.idx_post_waist = 1784

    def _build_loop(self, v: np.ndarray, faces: np.ndarray, y_plane: float) -> CircumferenceLoop:
        """Extract edge intersections crossing horizontal plane."""
        a1_list, b1_list, t1_list = [], [], []
        a2_list, b2_list, t2_list = [], [], []

        for face in faces:
            pts = v[face]
            y_vals = pts[:, 1]
            if y_vals.min() <= y_plane <= y_vals.max():
                edge_pairs = [(0, 1), (1, 2), (2, 0)]
                crossings = []
                for i1, i2 in edge_pairs:
                    y1, y2 = y_vals[i1], y_vals[i2]
                    if (y1 <= y_plane <= y2) or (y2 <= y_plane <= y1):
                        diff_y = abs(y2 - y1)
                        if diff_y > 1e-8:
                            t = (y_plane - y1) / (y2 - y1)
                            crossings.append((face[i1], face[i2], float(t)))
                if len(crossings) == 2:
                    a1_list.append(crossings[0][0])
                    b1_list.append(crossings[0][1])
                    t1_list.append(crossings[0][2])
                    a2_list.append(crossings[1][0])
                    b2_list.append(crossings[1][1])
                    t2_list.append(crossings[1][2])

        return CircumferenceLoop(
            idx_a1=torch.tensor(a1_list, dtype=torch.long, device=self.device),
            idx_b1=torch.tensor(b1_list, dtype=torch.long, device=self.device),
            t1=torch.tensor(t1_list, dtype=torch.float32, device=self.device).unsqueeze(-1),
            idx_a2=torch.tensor(a2_list, dtype=torch.long, device=self.device),
            idx_b2=torch.tensor(b2_list, dtype=torch.long, device=self.device),
            t2=torch.tensor(t2_list, dtype=torch.float32, device=self.device).unsqueeze(-1),
        )

    def extract_measurements(
        self,
        vertices: torch.Tensor,
        joints: torch.Tensor,
    ) -> dict[MeasurementZone, torch.Tensor]:
        """Extract all 12 zones differentiably from SMPL vertices and joints.

        Args:
            vertices: (6890, 3) PyTorch tensor in meters.
            joints: (24 or 45, 3) PyTorch tensor in meters.

        Returns:
            Mapping of MeasurementZone to scalar tensor in centimeters.
        """
        # Circumferences
        chest_cm = self.chest_loop.compute_perimeter_cm(vertices)
        waist_cm = self.waist_loop.compute_perimeter_cm(vertices)
        hip_cm = self.hip_loop.compute_perimeter_cm(vertices)
        neck_cm = self.neck_loop.compute_perimeter_cm(vertices)
        thigh_cm = self.thigh_loop.compute_perimeter_cm(vertices)

        # Linear: Shoulder Width (acromion to acromion)
        sh_diff = vertices[self.idx_l_acromion] - vertices[self.idx_r_acromion]
        shoulder_width_cm = torch.sqrt(torch.sum(sh_diff * sh_diff) + 1e-8) * 100.0

        # Linear: Sleeve Length (left shoulder tip -> elbow (18) -> wrist (20))
        seg1 = joints[16] - joints[18]
        seg2 = joints[18] - joints[20]
        sleeve_cm = (torch.sqrt(torch.sum(seg1 * seg1) + 1e-8) + torch.sqrt(torch.sum(seg2 * seg2) + 1e-8)) * 100.0

        # Linear: Inseam (crotch level down to left ankle joint 7).
        # Uses the crotch surface vertex, not the pelvis joint: the pelvis
        # joint is the kinematic hip-socket center, ~9-19cm above the actual
        # crotch on this topology (confirmed by diagnostic comparison against
        # the photo-side crotch estimate, which is derived independently from
        # hip landmarks and agreed with idx_crotch to within ~1cm but differed
        # from the pelvis joint by ~20cm). The previous code used joints[0]
        # (pelvis) despite this comment already saying "crotch level".
        inseam_cm = (vertices[self.idx_crotch, 1] - joints[7, 1]) * 100.0

        # Linear: Torso Length (C7 vertebra down to waist plane / spine1 joint 3)
        torso_len_cm = (vertices[self.idx_c7, 1] - joints[3, 1]) * 100.0

        # Linear: Back Rise (posterior waist down to crotch bottom)
        br_diff = vertices[self.idx_post_waist] - vertices[self.idx_crotch]
        back_rise_cm = torch.sqrt(torch.sum(br_diff * br_diff) + 1e-8) * 100.0

        # Linear: Upper Back (posterior armscye to armscye)
        ub_diff = vertices[self.idx_l_armscye] - vertices[self.idx_r_armscye]
        upper_back_cm = torch.sqrt(torch.sum(ub_diff * ub_diff) + 1e-8) * 100.0

        # Linear: Garment Length (C7 down to hip / pelvis plane)
        garment_len_cm = (vertices[self.idx_c7, 1] - joints[0, 1]) * 100.0

        return {
            MeasurementZone.chest: chest_cm,
            MeasurementZone.waist: waist_cm,
            MeasurementZone.hip: hip_cm,
            MeasurementZone.thigh: thigh_cm,
            MeasurementZone.neck: neck_cm,
            MeasurementZone.shoulder_width: shoulder_width_cm,
            MeasurementZone.sleeve_length: sleeve_cm,
            MeasurementZone.inseam: inseam_cm,
            MeasurementZone.torso_length: torso_len_cm,
            MeasurementZone.back_rise: back_rise_cm,
            MeasurementZone.upper_back: upper_back_cm,
            MeasurementZone.garment_length: garment_len_cm,
        }


def fit_smpl_shape(
    measurements: Measurements,
    target_height_cm: float,
    gender: str = "neutral",
    num_betas: int = 10,
    lr: float = 0.05,
    max_steps: int = 300,
    loss_tolerance: float = 1e-4,
    step_callback: Callable[[int, float, list[float]], None] | None = None,
) -> FitResult:
    """Fit 10 SMPL shape parameters (betas) to match photo measurements and height.

    Args:
        measurements: Measured body dimensions with per-zone confidence.
        target_height_cm: User stature in centimeters (50.0 to 260.0).
        gender: Model base template ('neutral', 'female', 'male').
        num_betas: Shape dimension count (default 10).
        lr: Adam learning rate (default 0.05).
        max_steps: Maximum optimization iterations (default 300).
        loss_tolerance: Plateau threshold for early stopping.
        step_callback: Optional hook called per step (step, loss, betas).

    Returns:
        FitResult containing optimized betas, mesh vertices, faces, and diagnostics.
    """
    if target_height_cm <= 50.0 or target_height_cm >= 260.0:
        raise ValueError(f"Invalid target height: {target_height_cm} cm. Expected between 50 and 260 cm.")

    gender_clean = gender.lower().strip()
    if gender_clean not in ("neutral", "female", "male"):
        raise ValueError(f"Invalid gender '{gender}'. Expected 'neutral', 'female', or 'male'.")

    # Ensure deterministic CPU fitting
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Load canonical model and measurement extractor
    model = load_smpl(gender=gender_clean, num_betas=num_betas)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    extractor = MeshMeasurementExtractor(gender=gender_clean, device=device)
    a_pose = get_a_pose_tensor(device=device)

    # Initialize optimizable betas
    betas = torch.zeros((1, num_betas), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=lr)

    # Filter target measurements to valid, positive entries
    active_targets: dict[MeasurementZone, tuple[float, float]] = {}
    for zone, m in measurements.zones.items():
        if m.value_cm > 0.0 and m.confidence > 0.0 and np.isfinite(m.value_cm):
            active_targets[zone] = (float(m.value_cm), float(m.confidence))

    # Weighting setup
    total_conf = sum(conf for _, conf in active_targets.values())
    meas_scale = 1.0 / max(1.0, total_conf)

    prev_loss = float("inf")
    converged = False
    best_loss = float("inf")
    best_betas = torch.zeros((1, num_betas), dtype=torch.float32)
    actual_steps = 0

    for step in range(1, max_steps + 1):
        actual_steps = step
        optimizer.zero_grad()

        # Forward pass in A-pose
        output = model(betas=betas, body_pose=a_pose)
        v = output.vertices[0]
        j = output.joints[0]

        # 1. Height term
        mesh_h_cm = (torch.max(v[:, 1]) - torch.min(v[:, 1])) * 100.0
        h_diff = mesh_h_cm - target_height_cm
        loss_height = (h_diff * h_diff)

        # 2. Measurement errors
        mesh_meas = extractor.extract_measurements(v, j)
        loss_meas = torch.tensor(0.0, device=device)
        for zone, (tgt_val, conf) in active_targets.items():
            est_val = mesh_meas[zone]
            diff = est_val - tgt_val
            loss_meas = loss_meas + conf * (diff * diff)
        loss_meas = loss_meas * meas_scale

        # 3. Shape regularization (0.01 * ||betas||^2)
        loss_reg = 0.01 * torch.sum(betas * betas)

        # Total loss
        loss = loss_meas + loss_height + loss_reg

        if not torch.isfinite(loss):
            # Guard against NaN/Inf: revert to best known betas
            betas.data = best_betas.clone()
            break

        current_loss = float(loss.item())
        if current_loss < best_loss:
            best_loss = current_loss
            best_betas = betas.detach().clone()

        loss.backward()
        optimizer.step()
        # Clamp betas to statistically plausible CAESAR bounds [-4.0, 4.0]
        betas.data.clamp_(-4.0, 4.0)

        if step_callback:
            step_callback(step, current_loss, betas.detach().cpu().numpy()[0].tolist())

        # Convergence check
        loss_delta = abs(prev_loss - current_loss)
        if step > 20 and loss_delta < loss_tolerance:
            converged = True
            break
        prev_loss = current_loss

    # Use best parameters found
    with torch.no_grad():
        final_output = model(betas=best_betas, body_pose=a_pose)
        final_v = final_output.vertices[0].cpu().numpy().astype(np.float32)
        final_j = final_output.joints[0]
        final_meas_tensors = extractor.extract_measurements(
            torch.tensor(final_v, device=device),
            final_j,
        )
        final_h = float((np.max(final_v[:, 1]) - np.min(final_v[:, 1])) * 100.0)

    # Compute diagnostics
    diagnostics: dict[str, dict[str, float]] = {}
    mesh_measurements_out: dict[MeasurementZone, float] = {}
    for zone, est_tensor in final_meas_tensors.items():
        est_val = float(est_tensor.item())
        mesh_measurements_out[zone] = round(est_val, 1)

        if zone in active_targets:
            tgt_val, conf = active_targets[zone]
            err = round(est_val - tgt_val, 2)
            diagnostics[zone.value] = {
                "target_cm": round(tgt_val, 1),
                "fitted_cm": round(est_val, 1),
                "error_cm": err,
                "abs_error_cm": abs(err),
                "confidence": conf,
            }

    faces_np = model.faces.astype(np.int32)
    fitted_betas = [float(b) for b in best_betas.cpu().numpy()[0]]

    # Validity checks: ensure fitted mesh matches human bounds and has finite geometry
    is_valid = True
    rejection_reason = None
    if not np.all(np.isfinite(final_v)):
        is_valid = False
        rejection_reason = "Non-finite coordinates detected in fitted mesh vertices."
    elif abs(final_h - target_height_cm) > 8.0:
        is_valid = False
        rejection_reason = f"Fitted stature {final_h:.1f}cm deviates from target {target_height_cm:.1f}cm by > 8cm."
    elif best_loss > 500.0:
        is_valid = False
        rejection_reason = f"Fitting loss {best_loss:.1f} exceeds maximum acceptable threshold."

    return FitResult(
        gender=gender_clean,
        betas=fitted_betas,
        vertices=final_v,
        faces=faces_np,
        fitted_height_cm=round(final_h, 1),
        target_height_cm=round(target_height_cm, 1),
        loss=round(best_loss, 4),
        steps=actual_steps,
        converged=converged,
        is_valid=is_valid,
        rejection_reason=rejection_reason,
        diagnostics=diagnostics,
        mesh_measurements=mesh_measurements_out,
    )
