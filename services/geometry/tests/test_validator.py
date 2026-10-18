"""Unit tests for services/geometry/validator.py.

Verifies:
1. Adult age consent gate (status=rejected, AGE_CONSENT_REQUIRED).
2. Safety & policy rejection (child imagery, inappropriate apparel).
3. Quality check retake request (framing, loose clothing, blur).
4. Successful validation (status=measuring, can_proceed=True).
5. Local image quality heuristics (blur and illumination).
6. Ephemeral file cleanup on success and failure paths.
7. Front-photo retention opt-in behavior.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unittest.mock import patch
import cv2
import numpy as np
import pytest

from fitcore.schemas import JobStatus

from services.geometry.validator import (
    GeminiPhotoCheckResponse,
    PhotoSafetyAssessment,
    ValidationOutcome,
    ViewValidationResult,
    check_chest_arm_separation,
    check_local_image_quality,
    cleanup_local_photos,
    temporary_photo_workspace,
    validate_photos,
)


def _make_dummy_jpeg(color: int = 128, width: int = 200, height: int = 300) -> bytes:
    """Generate a clean synthetic test JPEG image with clear contrast."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    # Add high-frequency gradient pattern to prevent blur detection
    cv2.rectangle(arr, (20, 20), (width - 20, height - 20), (255, 255, 255), 3)
    cv2.line(arr, (50, 50), (150, 250), (0, 0, 0), 2)
    _, encoded = cv2.imencode(".jpg", arr)
    return encoded.tobytes()


@pytest.fixture
def clean_dummy_images() -> dict[str, bytes]:
    return {
        "front": _make_dummy_jpeg(120),
        "back": _make_dummy_jpeg(130),
        "left": _make_dummy_jpeg(140),
        "right": _make_dummy_jpeg(150),
    }


def test_missing_adult_confirmation_rejects_immediately(clean_dummy_images):
    """Missing 18+ consent must reject immediately without calling external services."""
    outcome = validate_photos(
        front=clean_dummy_images["front"],
        back=clean_dummy_images["back"],
        left=clean_dummy_images["left"],
        right=clean_dummy_images["right"],
        age_18_confirmed=False,
    )
    assert outcome.status == JobStatus.rejected
    assert outcome.can_proceed is False
    assert outcome.error_code == "AGE_CONSENT_REQUIRED"
    assert "18 years or older" in outcome.error_message


def test_safety_violation_child_detected(clean_dummy_images):
    """Detection of child imagery must result in neutral rejection."""
    mock_gemini_resp = GeminiPhotoCheckResponse(
        views={
            "front": ViewValidationResult(ok=True),
            "back": ViewValidationResult(ok=True),
            "left": ViewValidationResult(ok=True),
            "right": ViewValidationResult(ok=True),
        },
        safety=PhotoSafetyAssessment(
            is_child=True,
            is_inappropriate=False,
            safety_notes="Subject appears to be under 18 years old.",
        ),
        overall_summary="Policy check triggered.",
    )

    with patch("services.geometry.validator.generate_json", return_value=mock_gemini_resp):
        outcome = validate_photos(
            front=clean_dummy_images["front"],
            back=clean_dummy_images["back"],
            left=clean_dummy_images["left"],
            right=clean_dummy_images["right"],
            age_18_confirmed=True,
        )

    assert outcome.status == JobStatus.rejected
    assert outcome.can_proceed is False
    assert outcome.error_code == "POLICY_VIOLATION"


def test_safety_violation_unclothed_or_underwear(clean_dummy_images):
    """Detection of underwear or unclothed subject must result in neutral rejection."""
    mock_gemini_resp = GeminiPhotoCheckResponse(
        views={
            "front": ViewValidationResult(ok=True),
            "back": ViewValidationResult(ok=True),
            "left": ViewValidationResult(ok=True),
            "right": ViewValidationResult(ok=True),
        },
        safety=PhotoSafetyAssessment(
            is_child=False,
            is_inappropriate=True,
            safety_notes="Subject in underwear.",
        ),
        overall_summary="Policy check triggered.",
    )

    with patch("services.geometry.validator.generate_json", return_value=mock_gemini_resp):
        outcome = validate_photos(
            front=clean_dummy_images["front"],
            back=clean_dummy_images["back"],
            left=clean_dummy_images["left"],
            right=clean_dummy_images["right"],
            age_18_confirmed=True,
        )

    assert outcome.status == JobStatus.rejected
    assert outcome.can_proceed is False
    assert outcome.error_code == "POLICY_VIOLATION"


def test_quality_deficiency_triggers_needs_retake(clean_dummy_images):
    """Quality or framing issues must result in needs_retake with fix instructions."""
    mock_gemini_resp = GeminiPhotoCheckResponse(
        views={
            "front": ViewValidationResult(
                ok=False,
                issues=["loose_clothing"],
                fix_instruction="Please wear form-fitting apparel like leggings.",
            ),
            "back": ViewValidationResult(ok=True),
            "left": ViewValidationResult(
                ok=False,
                issues=["head_or_feet_cropped"],
                fix_instruction="Step back until entire body from head to feet is framed.",
            ),
            "right": ViewValidationResult(ok=True),
        },
        safety=PhotoSafetyAssessment(is_child=False, is_inappropriate=False),
        overall_summary="Two views need retake.",
    )

    with patch("services.geometry.validator.generate_json", return_value=mock_gemini_resp):
        outcome = validate_photos(
            front=clean_dummy_images["front"],
            back=clean_dummy_images["back"],
            left=clean_dummy_images["left"],
            right=clean_dummy_images["right"],
            age_18_confirmed=True,
        )

    assert outcome.status == JobStatus.needs_retake
    assert outcome.can_proceed is False
    assert outcome.error_code == "PHOTO_QUALITY_CHECK_FAILED"
    assert "form-fitting" in outcome.error_message
    assert "Step back" in outcome.error_message


def test_successful_validation_transitions_to_measuring(clean_dummy_images):
    """When all 4 views pass and safety is satisfied, job transitions to measuring."""
    mock_gemini_resp = GeminiPhotoCheckResponse(
        views={
            "front": ViewValidationResult(ok=True, issues=[]),
            "back": ViewValidationResult(ok=True, issues=[]),
            "left": ViewValidationResult(ok=True, issues=[]),
            "right": ViewValidationResult(ok=True, issues=[]),
        },
        safety=PhotoSafetyAssessment(is_child=False, is_inappropriate=False),
        overall_summary="All 4 views meet photogrammetry criteria.",
    )

    with patch("services.geometry.validator.generate_json", return_value=mock_gemini_resp):
        outcome = validate_photos(
            front=clean_dummy_images["front"],
            back=clean_dummy_images["back"],
            left=clean_dummy_images["left"],
            right=clean_dummy_images["right"],
            age_18_confirmed=True,
        )

    assert outcome.status == JobStatus.measuring
    assert outcome.can_proceed is True
    assert outcome.error_code is None
    assert outcome.error_message is None
    assert all(res.ok for res in outcome.views.values())


def test_local_blur_and_lighting_heuristics():
    """Deterministic local checks must flag underexposed or severely blurred images."""
    # 1. Severely underexposed image (black)
    black_img = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()
    is_ok, issues, fix = check_local_image_quality(black_img)
    assert is_ok is False
    assert "poor_lighting" in issues

    # 2. Overexposed image (pure white)
    white_img = cv2.imencode(".jpg", np.full((100, 100, 3), 255, dtype=np.uint8))[1].tobytes()
    is_ok, issues, fix = check_local_image_quality(white_img)
    assert is_ok is False
    assert "poor_lighting" in issues


def test_sharp_subject_with_large_flat_background_is_not_flagged_as_blurry():
    """Regression test: a studio-style photo (sharp subject, large plain backdrop
    margins) must not be rejected as blurry just because most of the frame is
    flat background. The blur measure must be based on the central subject
    region, not diluted by surrounding empty space.

    This reproduces a real false positive found testing with real consented
    photos: a sharp 721x1075 side-profile photo against a plain grey backdrop
    scored a full-frame Laplacian variance of 71.0 (below the 80.0 threshold,
    flagged 'blurry'), while the same region cropped to the subject scored
    168.9 (comfortably sharp).
    """
    width, height = 720, 1080
    # Large flat (near-zero-gradient) background filling most of the frame.
    arr = np.full((height, width, 3), 235, dtype=np.uint8)
    # Sharp, high-frequency detail confined to the central ~50% width where a
    # person would actually be standing in a centered portrait/product photo.
    rng = np.random.default_rng(42)
    x0, x1 = int(width * 0.35), int(width * 0.65)
    y0, y1 = int(height * 0.1), int(height * 0.9)
    arr[y0:y1, x0:x1] = rng.integers(0, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sharp_img = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    is_ok, issues, fix = check_local_image_quality(sharp_img)
    assert is_ok is True, f"Sharp subject wrongly flagged: issues={issues}, fix={fix}"
    assert "poor_lighting" not in issues


def test_genuinely_blurred_subject_is_still_flagged():
    """The subject-crop fix must not weaken real blur detection: a heavily
    blurred version of the same central-detail layout must still fail."""
    width, height = 720, 1080
    arr = np.full((height, width, 3), 235, dtype=np.uint8)
    rng = np.random.default_rng(42)
    x0, x1 = int(width * 0.35), int(width * 0.65)
    y0, y1 = int(height * 0.1), int(height * 0.9)
    arr[y0:y1, x0:x1] = rng.integers(0, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(arr, (31, 31), 0)
    blurred_img = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    is_ok, issues, fix = check_local_image_quality(blurred_img)
    assert is_ok is False
    assert "poor_lighting" in issues


def test_full_body_photo_with_mostly_plain_clothing_is_not_flagged_as_blurry():
    """Regression test for a second real false positive found testing with real
    consented photos: a full-body studio photo (sharp hair/face, but mostly
    plain/solid clothing covering most of the body) scored a cropped-subject
    Laplacian variance of ~49-59 even though it was visually confirmed sharp
    (hair strands and skin texture clearly resolved on close inspection).
    Plain clothing fabric inherently has far less edge density than a face
    closeup, so a full-body photo's overall variance is legitimately lower
    than a portrait's, independent of actual focus. The threshold (45.0) was
    chosen to admit this real photo while still rejecting genuine blur (see
    test_genuinely_blurred_subject_is_still_flagged).
    """
    width, height = 720, 1080
    arr = np.full((height, width, 3), 235, dtype=np.uint8)
    rng = np.random.default_rng(7)
    x0, x1 = int(width * 0.35), int(width * 0.65)
    y0, y1 = int(height * 0.1), int(height * 0.9)
    # Low-frequency texture (locally-averaged noise) approximates plain fabric:
    # real detail present, but far less high-frequency edge content than hair/skin.
    noise = rng.integers(0, 256, size=((y1 - y0) // 4 + 1, (x1 - x0) // 4 + 1, 3), dtype=np.uint8)
    low_freq = cv2.resize(noise, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    arr[y0:y1, x0:x1] = low_freq
    photo_like_img = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    is_ok, issues, fix = check_local_image_quality(photo_like_img)
    assert is_ok is True, f"Real-world-style sharp full-body photo wrongly flagged: issues={issues}, fix={fix}"


def test_temporary_photo_workspace_cleans_up_on_success(clean_dummy_images):
    """Temporary workspace must unlink all files when opt_in_keep_front is False."""
    saved_paths = []
    with temporary_photo_workspace(clean_dummy_images, opt_in_keep_front=False) as paths:
        for view, p in paths.items():
            assert p.exists()
            assert p.is_file()
            saved_paths.append(p)

    # After exiting context manager, all files must be unlinked
    for p in saved_paths:
        assert not p.exists()


def test_temporary_photo_workspace_cleans_up_on_exception(clean_dummy_images):
    """Temporary workspace must unlink files even if an exception occurs inside the block."""
    saved_paths = []
    with pytest.raises(RuntimeError):
        with temporary_photo_workspace(clean_dummy_images, opt_in_keep_front=False) as paths:
            saved_paths = list(paths.values())
            for p in saved_paths:
                assert p.exists()
            raise RuntimeError("Simulated processing error")

    # Files must still be deleted
    for p in saved_paths:
        assert not p.exists()


def test_temporary_photo_workspace_preserves_front_on_opt_in(clean_dummy_images):
    """When opt_in_keep_front is True, front photo must be retained while others are deleted."""
    front_path = None
    other_paths = []
    with temporary_photo_workspace(clean_dummy_images, opt_in_keep_front=True) as paths:
        front_path = paths["front"]
        other_paths = [paths["back"], paths["left"], paths["right"]]
        for p in paths.values():
            assert p.exists()

    # Front photo should still exist; other photos must be deleted
    try:
        assert front_path.exists()
        for p in other_paths:
            assert not p.exists()
    finally:
        # Clean up retained front photo manually
        if front_path and front_path.exists():
            front_path.unlink()
            if front_path.parent.exists():
                try:
                    front_path.parent.rmdir()
                except Exception:
                    pass


# --- check_chest_arm_separation: deterministic chest-specific gap gate ---
#
# Shared synthetic landmark geometry for all cases below (400x800 canvas,
# torso centered at x=200, matching the convention used elsewhere in this
# test suite): shoulders at y=200 (x=160/240), hips at y=416 (x=168/232).
# These give a chest row at round(200 + 0.22*(416-200)) = 247.52 -> 248.
_CHEST_GATE_IMG_W = 400
_CHEST_GATE_IMG_H = 800
_CHEST_GATE_Y_SHOULDERS = 200.0
_CHEST_GATE_Y_HIPS = 416.0
_CHEST_GATE_L_SHOULDER_X = 240.0
_CHEST_GATE_R_SHOULDER_X = 160.0
_CHEST_GATE_L_HIP_X = 232.0
_CHEST_GATE_R_HIP_X = 168.0
_CHEST_GATE_CM_PER_PX = 175.0 / 600.0
_CHEST_GATE_CHEST_ROW = 248


def _check_chest_gate(mask: np.ndarray) -> tuple[bool, str]:
    return check_chest_arm_separation(
        mask,
        _CHEST_GATE_Y_SHOULDERS, _CHEST_GATE_Y_HIPS,
        _CHEST_GATE_L_SHOULDER_X, _CHEST_GATE_R_SHOULDER_X,
        _CHEST_GATE_L_HIP_X, _CHEST_GATE_R_HIP_X,
        _CHEST_GATE_CM_PER_PX,
    )


def test_chest_gate_passes_when_arms_genuinely_separated():
    """A front mask with a clear background gap between torso and arms must pass."""
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1  # torso, 80px wide
    mask[150:400, 60:140] = 1   # left arm, separated by a 20px background gap
    mask[150:400, 260:340] = 1  # right arm, separated by a 20px background gap

    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is True
    assert msg == ""


def test_chest_gate_rejects_connected_arm_and_torso():
    """A front mask where the arm is part of the same contiguous region as the
    torso (no background gap) must fail with an arms_against_body-style
    instruction, reproducing the real failure mode found testing with a real
    photo and a synthetic SMPL round trip.
    """
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1
    mask[150:400, 60:340] = 1  # merges directly into the torso, no gap at all

    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is False
    assert "arm" in msg.lower()
    assert "retake" in msg.lower()


def test_chest_gate_rejects_partial_apose_separated_below_chest_but_merged_at_chest_row():
    """Reproduces the real generated-photo failure (local-test-data A-pose set):
    a natural, moderate A-pose where the FOREARMS swing clear of the body lower
    down, but the UPPER arms remain against the torso at the chest row itself.

    Direct pixel analysis of that front mask (984x1599) showed exactly this: a
    single contiguous arm+torso segment across the 0.10-0.30 torso-span band
    (which includes the 0.22 chest row), splitting into three clean segments
    (left arm | torso | right arm, 31-73px gaps) only at 0.40+. The gate reads
    the chest row specifically, so it must STILL reject here -- separation lower
    down does not make the chest line measurable. This is correct behaviour, not
    a false rejection: measuring chest at a contaminated row would inflate it
    (the real photo's raw chest read ~102cm vs a plausible ~73cm).
    """
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1          # torso
    # Upper arms merged with the torso AT the chest row (248): fill arm->torso edge, no gap.
    mask[200:270, 120:160] = 1          # left upper arm, touching torso edge at the chest row
    mask[200:270, 240:280] = 1          # right upper arm, touching torso edge at the chest row
    # Forearms swing out below the chest row -> clear background gaps there (not at 248).
    mask[300:420, 60:120] = 1           # left forearm, 40px gap from torso below the chest row
    mask[300:420, 280:340] = 1          # right forearm, 40px gap from torso below the chest row

    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is False, "arms merged at the chest row must be rejected even if they separate lower"
    assert "arm" in msg.lower() and "retake" in msg.lower()


def test_chest_gate_retake_message_guides_to_shoulder_height_pose():
    """The retake guidance must tell the user the pose that actually clears the gate -- arms
    raised to ~shoulder height with a clear gap at chest level -- not just 'arms slightly away',
    which produces a pose the gate still rejects (the capture-guidance/validator mismatch fix)."""
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1
    mask[150:400, 60:340] = 1  # arms merged into torso -> gate fails, returns the retake guidance

    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is False
    low = msg.lower()
    assert "shoulder height" in low, f"retake guidance should request arms at shoulder height: {msg!r}"
    assert "gap" in low and "chest" in low, f"retake guidance should mention a clear gap at chest level: {msg!r}"
    assert "slightly" not in low, f"retake guidance must no longer say 'slightly' (the under-abducted pose): {msg!r}"


def test_chest_gate_tolerates_isolated_mask_noise():
    """A single stray foreground pixel just outside the expected torso window
    must not cause a false rejection -- this is the exact scenario a real
    segmentation mask could produce from minor noise, distinct from genuine
    arm contamination (which spans many contiguous pixels, not one).
    """
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1
    mask[150:400, 60:140] = 1
    mask[150:400, 260:340] = 1
    mask[_CHEST_GATE_CHEST_ROW, 145] = 1  # single stray pixel just outside the torso window

    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is True, f"isolated noise pixel incorrectly triggered a retake: {msg}"


def test_chest_gate_rejects_torso_boundary_unidentifiable():
    """An empty mask (or one with no foreground in the expected torso window)
    must fail safe (needs_retake), not silently pass, since the gate cannot
    verify anything about arm separation without a torso to anchor on."""
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    is_ok, msg = _check_chest_gate(mask)
    assert is_ok is False
    assert "boundary" in msg.lower() or "retake" in msg.lower()


def test_chest_gate_rejects_invalid_landmark_geometry():
    """Degenerate landmark input (e.g. hips not below shoulders, zero scale)
    must fail safe rather than crash or silently pass."""
    mask = np.zeros((_CHEST_GATE_IMG_H, _CHEST_GATE_IMG_W), dtype=np.uint8)
    mask[100:700, 160:240] = 1
    # hips above shoulders (invalid torso_span <= 0)
    is_ok, msg = check_chest_arm_separation(
        mask, 400.0, 200.0,
        _CHEST_GATE_L_SHOULDER_X, _CHEST_GATE_R_SHOULDER_X,
        _CHEST_GATE_L_HIP_X, _CHEST_GATE_R_HIP_X,
        _CHEST_GATE_CM_PER_PX,
    )
    assert is_ok is False

    # zero scale
    is_ok2, msg2 = check_chest_arm_separation(
        mask, _CHEST_GATE_Y_SHOULDERS, _CHEST_GATE_Y_HIPS,
        _CHEST_GATE_L_SHOULDER_X, _CHEST_GATE_R_SHOULDER_X,
        _CHEST_GATE_L_HIP_X, _CHEST_GATE_R_HIP_X,
        0.0,
    )
    assert is_ok2 is False
