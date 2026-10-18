"""Unit tests for services/geometry/poses.py (Prompt 5, Phase 1: pose foundation).

Verifies:
1. The 'standing' preset is exactly the existing A-pose tensor (zero drift
   from the fitting/measurement pipeline's current default behavior).
2. The 'seated' preset produces geometrically distinct, self-consistent
   joint positions -- checked via concrete joint-position invariants
   (height drop, thigh flexion, shin orientation), not merely "it rendered".
3. generate_posed_mesh's 'standing' output exactly matches what the
   existing fitting.py forward pass already produces for the same betas.
4. event_to_poses is deterministic and follows its documented keyword rule.
5. Unknown pose names are rejected.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from fitcore.schemas import CalendarEvent
from services.geometry.fitting import get_a_pose_tensor
from services.geometry.loader import load_smpl
from services.geometry.poses import event_to_poses, generate_posed_mesh, get_pose_tensor


def test_standing_pose_tensor_matches_existing_a_pose_exactly():
    standing = get_pose_tensor("standing")
    reference = get_a_pose_tensor()
    assert torch.equal(standing, reference), "standing preset must be a no-op relative to the existing A-pose"


def test_unknown_pose_name_rejected():
    with pytest.raises(ValueError, match="Unknown pose preset"):
        get_pose_tensor("crouching")  # type: ignore[arg-type]


def test_seated_pose_geometry_is_distinct_from_standing():
    """Concrete joint-position checks, not a render-succeeded check."""
    model = load_smpl(gender="neutral", num_betas=10)
    model.eval()
    betas = torch.zeros((1, 10), dtype=torch.float32)

    with torch.no_grad():
        standing_out = model(betas=betas, body_pose=get_pose_tensor("standing"))
        seated_out = model(betas=betas, body_pose=get_pose_tensor("seated"))

    standing_v, standing_j = standing_out.vertices[0], standing_out.joints[0]
    seated_v, seated_j = seated_out.vertices[0], seated_out.joints[0]

    assert torch.all(torch.isfinite(seated_v)), "seated mesh must have finite coordinates"

    standing_h_cm = float((torch.max(standing_v[:, 1]) - torch.min(standing_v[:, 1])) * 100.0)
    seated_h_cm = float((torch.max(seated_v[:, 1]) - torch.min(seated_v[:, 1])) * 100.0)
    assert seated_h_cm < standing_h_cm - 20.0, (
        f"seated overall height ({seated_h_cm:.1f}cm) should be well below standing ({standing_h_cm:.1f}cm)"
    )

    # Thigh (left hip -> left knee, joints 1 -> 4): vertical when standing,
    # should become predominantly horizontal (small |dy|, large |dz|) seated.
    standing_thigh_dy = float(standing_j[4, 1] - standing_j[1, 1])
    standing_thigh_dz = float(standing_j[4, 2] - standing_j[1, 2])
    seated_thigh_dy = float(seated_j[4, 1] - seated_j[1, 1])
    seated_thigh_dz = float(seated_j[4, 2] - seated_j[1, 2])
    assert abs(standing_thigh_dy) > abs(standing_thigh_dz), "sanity check: standing thigh is vertical, not horizontal"
    assert abs(seated_thigh_dz) > abs(seated_thigh_dy), "seated thigh (hip->knee) should be predominantly horizontal"

    # Shin (left knee -> left ankle, joints 4 -> 7): the knee counter-rotation
    # should leave its orientation close to the standing value (hanging
    # downward from the now-horizontal thigh), not flipped upward.
    standing_shin_dy = float(standing_j[7, 1] - standing_j[4, 1])
    seated_shin_dy = float(seated_j[7, 1] - seated_j[4, 1])
    assert standing_shin_dy < 0.0, "sanity check: standing ankle sits below the knee"
    assert seated_shin_dy < 0.0, "seated ankle should still sit below the knee, not fold upward above it"
    assert abs(seated_shin_dy - standing_shin_dy) < 0.02, (
        "seated shin orientation should stay close to its standing value "
        f"(standing={standing_shin_dy * 100:.1f}cm, seated={seated_shin_dy * 100:.1f}cm)"
    )


def test_generate_posed_mesh_standing_matches_existing_fitting_forward_pass():
    """Prove the new additive path does not drift from the existing default."""
    betas = [0.4, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gender = "neutral"

    v_new, f_new = generate_posed_mesh(betas, gender=gender, pose="standing")

    model = load_smpl(gender=gender, num_betas=10)
    model.eval()
    betas_t = torch.tensor([betas], dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=betas_t, body_pose=get_a_pose_tensor())
        v_existing = out.vertices[0].cpu().numpy().astype(np.float32)
    f_existing = model.faces.astype(np.int32)

    assert v_new.shape == (6890, 3)
    assert f_new.shape == (13776, 3)
    np.testing.assert_array_equal(v_new, v_existing)
    np.testing.assert_array_equal(f_new, f_existing)


def test_generate_posed_mesh_seated_is_finite_and_differs_from_standing():
    betas = [0.0] * 10
    v_standing, _ = generate_posed_mesh(betas, pose="standing")
    v_seated, faces = generate_posed_mesh(betas, pose="seated")

    assert v_seated.shape == (6890, 3)
    assert faces.shape == (13776, 3)
    assert np.all(np.isfinite(v_seated))
    assert not np.allclose(v_standing, v_seated), "seated mesh must differ from standing for the same betas"


def _event(event_id, title, start_time, end_time=None, description=None):
    return CalendarEvent(id=event_id, title=title, start_time=start_time, end_time=end_time, description=description)


def test_event_to_poses_is_deterministic():
    event = _event(
        "event-client-meeting", "10:00 Client meeting", "2026-01-05T10:00:00", "2026-01-05T11:30:00",
        "Q4 strategic review and enterprise product roadmap presentation.",
    )
    first = event_to_poses(event)
    second = event_to_poses(event)
    assert first == second


def test_event_to_poses_seated_meeting():
    event = _event(
        "event-client-meeting", "10:00 Client meeting", "2026-01-05T10:00:00", "2026-01-05T11:30:00",
        "Q4 strategic review and enterprise product roadmap presentation.",
    )
    poses = event_to_poses(event)
    assert poses == {"standing": 0.0, "seated": 90.0}


def test_event_to_poses_standing_networking_event():
    event = _event(
        "event-mixer", "Evening networking mixer", "2026-01-05T18:00:00", "2026-01-05T19:00:00",
    )
    poses = event_to_poses(event)
    assert poses == {"standing": 60.0, "seated": 0.0}


def test_event_to_poses_defaults_duration_when_end_time_missing():
    event = _event("event-no-end", "Flight to the venue", "2026-01-05T06:00:00")
    poses = event_to_poses(event)
    # "flight" is a seated keyword; no end_time -> default 60.0-minute duration.
    assert poses == {"standing": 0.0, "seated": 60.0}


def test_event_to_poses_seated_keyword_wins_when_both_match():
    event = _event(
        "event-wedding-reception", "19:00 Wedding reception", "2026-01-05T19:00:00", "2026-01-05T22:30:00",
        "Evening wedding dinner celebration and reception.",
    )
    poses = event_to_poses(event)
    # Title/description match both a standing keyword ("reception") and a
    # seated keyword ("dinner"); documented rule: seated wins.
    assert poses == {"standing": 0.0, "seated": 210.0}
