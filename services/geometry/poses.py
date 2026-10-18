"""Named SMPL pose presets and deterministic event-to-posture mapping.

Prompt 5, Phase 1 (pose foundation). Purely additive: the fixed A-pose used
throughout fit_smpl_shape and the measurement pipeline (fitting.py) is not
modified. This module adds named presets alongside it and a separate,
post-fit mesh-regeneration path -- shape fitting (betas) and pose selection
are independent steps here, not a change to how betas are fitted.

Pose joint-rotation values below are a geometric approximation chosen and
checked against real SMPL joint positions (see test_poses.py): the seated
preset is verified to (a) lower the overall mesh height, (b) rotate the
hip->knee segment from vertical to horizontal (genuine thigh flexion), and
(c) leave the knee->ankle segment's orientation unchanged relative to its
standing value (the knee rotation cancels the hip rotation for the shin, so
the lower leg keeps hanging the way it does when standing). That is evidence
the rotation composes the way a seated pose should, not a claim that the
exact 90-degree angles are a biomechanically or motion-capture-verified
sitting posture -- no such reference was available to check against.
"""

from __future__ import annotations

import datetime
import math
from typing import Literal

import numpy as np
import torch

from fitcore.schemas import CalendarEvent
from services.geometry.fitting import get_a_pose_tensor
from services.geometry.loader import load_smpl

PoseName = Literal["standing", "seated"]

_HIP_FLEX_DEG = 90.0
_KNEE_FLEX_DEG = 90.0

# SMPL body_pose is a flat (69,) axis-angle vector covering joints 1..23;
# joint 0 (pelvis) is the separate global_orient and is not part of it. Flat
# index for joint N is (N-1)*3 .. (N-1)*3+3 -- the same convention already
# relied on by get_a_pose_tensor's shoulder rotation (joint 16 -> index 45)
# and by fitting.py's own joint lookups (joints[1]/[2] left/right hip,
# joints[4]/[5] left/right knee, joints[7] left ankle).
_LEFT_HIP_JOINT = 1
_RIGHT_HIP_JOINT = 2
_LEFT_KNEE_JOINT = 4
_RIGHT_KNEE_JOINT = 5


def _joint_slice(joint_index: int) -> tuple[int, int]:
    start = (joint_index - 1) * 3
    return start, start + 3


def _get_seated_pose_tensor(device: torch.device | str = "cpu") -> torch.Tensor:
    """Hip and knee flexion about the local X (flexion) axis.

    The sign combination below was determined empirically, by comparing
    SMPL joint positions under trial rotations, not assumed from a formula:
    left/right hip take opposite signs and left/right knee take opposite
    signs (with the knee's sign opposite its same-side hip) -- this is the
    combination that folds the leg into a seated shape rather than an
    inverted/crouched one. See test_poses.py for the exact invariants this
    produces.
    """
    body_pose = get_a_pose_tensor(device=device).clone()
    hip_angle = math.radians(_HIP_FLEX_DEG)
    knee_angle = math.radians(_KNEE_FLEX_DEG)

    l_hip_s, _ = _joint_slice(_LEFT_HIP_JOINT)
    r_hip_s, _ = _joint_slice(_RIGHT_HIP_JOINT)
    l_knee_s, _ = _joint_slice(_LEFT_KNEE_JOINT)
    r_knee_s, _ = _joint_slice(_RIGHT_KNEE_JOINT)

    body_pose[0, l_hip_s + 0] = hip_angle
    body_pose[0, r_hip_s + 0] = -hip_angle
    body_pose[0, l_knee_s + 0] = -knee_angle
    body_pose[0, r_knee_s + 0] = knee_angle
    return body_pose


def get_pose_tensor(pose: PoseName = "standing", device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the body_pose tensor for a named preset.

    'standing' returns exactly get_a_pose_tensor()'s own tensor -- the one
    already used by fit_smpl_shape and the measurement pipeline -- so
    selecting it is a no-op relative to existing behavior.
    """
    if pose == "standing":
        return get_a_pose_tensor(device=device)
    if pose == "seated":
        return _get_seated_pose_tensor(device=device)
    raise ValueError(f"Unknown pose preset '{pose}'. Expected 'standing' or 'seated'.")


def generate_posed_mesh(
    betas: list[float],
    gender: str = "neutral",
    pose: PoseName = "standing",
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate SMPL vertices/faces for already-fitted shape betas in a
    named pose. Does not refit shape: betas come from fit_smpl_shape
    unchanged, so this is a pose-only rendering step on top of the existing
    fitting pipeline, not a replacement for it. Returns (vertices, faces) in
    the same (6890,3) float32 meters / (13776,3) int32 shape as
    FitResult.vertices/faces, so the existing GLB exporter
    (export_twin_glb) consumes it unmodified.
    """
    num_betas = len(betas)
    model = load_smpl(gender=gender, num_betas=num_betas)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    betas_t = torch.tensor([betas], dtype=torch.float32, device=device)
    body_pose = get_pose_tensor(pose, device=device)

    with torch.no_grad():
        output = model(betas=betas_t, body_pose=body_pose)
        vertices = output.vertices[0].cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)
    return vertices, faces


def generate_posed_output_with_joints(
    betas: list[float],
    gender: str = "neutral",
    pose: PoseName = "standing",
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Same pose-only regeneration as generate_posed_mesh, but also returns
    the SMPL joints tensor (as torch, not numpy) so a caller can run
    MeshMeasurementExtractor.extract_measurements (which needs both
    vertices and joints) on a posed mesh -- e.g. to compare a twin's
    length-type measurements (torso_length, inseam, ...) across poses. A
    separate function so generate_posed_mesh's (vertices, faces) contract
    for GLB export callers is unchanged.
    """
    num_betas = len(betas)
    model = load_smpl(gender=gender, num_betas=num_betas)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    betas_t = torch.tensor([betas], dtype=torch.float32, device=device)
    body_pose = get_pose_tensor(pose, device=device)

    with torch.no_grad():
        output = model(betas=betas_t, body_pose=body_pose)
        vertices = output.vertices[0]
        joints = output.joints[0]
    faces = model.faces.astype(np.int32)
    return vertices, joints, faces


# --------------------------------------------------------------------------- Event -> posture mapping

# Deterministic keyword rule, invented for this phase in the absence of an
# original Prompt 5 specification (none was found in the repository -- see
# the Phase-1 audit). Classifies an event as entirely one posture or the
# other; this is a documented simplification, not a claim that any real
# event is actually single-posture throughout. If both a seated and a
# standing keyword match, seated wins (meetings/dinners are assumed to be
# the dominant posture over an ancillary standing portion).
_SEATED_KEYWORDS = ("meeting", "dinner", "lunch", "flight", "drive", "interview", "lecture", "class", "call")
_STANDING_KEYWORDS = ("reception", "networking", "tour", "standing", "exhibition", "mixer", "cocktail", "rally")

_DEFAULT_DURATION_MINUTES = 60.0


def event_to_poses(event: CalendarEvent) -> dict[str, float]:
    """Deterministic posture-duration estimate (minutes) for a calendar event.

    Rule:
    1. Duration is end_time - start_time in minutes; if end_time is missing,
       default to 60.0 minutes.
    2. The event's title + description are matched (case-insensitive
       substring) against the keyword lists above. A seated match wins,
       else a standing match wins, else standing is the default (most
       unclassified calendar time is not purely seated).
    Always returns both keys; exactly one equals the event's full duration
    and the other is 0.0.
    """
    start = datetime.datetime.fromisoformat(event.start_time)
    if event.end_time:
        end = datetime.datetime.fromisoformat(event.end_time)
        duration_minutes = max(0.0, (end - start).total_seconds() / 60.0)
    else:
        duration_minutes = _DEFAULT_DURATION_MINUTES

    text = f"{event.title} {event.description or ''}".lower()
    is_seated = any(keyword in text for keyword in _SEATED_KEYWORDS)
    is_standing = any(keyword in text for keyword in _STANDING_KEYWORDS)

    if is_seated:
        return {"standing": 0.0, "seated": duration_minutes}
    if is_standing:
        return {"standing": duration_minutes, "seated": 0.0}
    return {"standing": duration_minutes, "seated": 0.0}
