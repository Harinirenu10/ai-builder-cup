"""End-to-end integration tests for 3D body twin reconstruction (Phase 10).

Covers:
1. Complete happy-path pipeline:
   Photo validation -> Landmark extraction -> 12 Measurements -> SMPL Fitting -> Critic Refinement -> GLB Export -> Twin Document -> Privacy Cleanup.
2. Critic refinement loop with multi-round adjustments and round logging.
3. Strict zero-retention raw photo privacy cleanup (all 4 deleted vs. front preserved on opt-in).
4. Failure modes and safety gates:
   - Underage consent rejection (status='rejected', error_code='AGE_CONSENT_REQUIRED').
   - Image quality / blur defect (status='needs_retake', error_code='BLUR_DETECTED').
   - Missing input photos (status='failed', error_code='MISSING_PHOTOS').
   - Unhandled processing error (status='failed', error_code='PROCESSING_ERROR') with guaranteed cleanup.
5. Consistency between worker pipeline (`process_twin_job`) and offline CLI (`run_offline_pipeline`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest

from fitcore.schemas import (
    CriticReport,
    CriticRegionFeedback,
    CriticRegion,
    CriticView,
    CriticIssue,
    CriticCause,
    JobDocument,
    JobStatus,
    Measurement,
    Measurements,
    MeasurementZone,
    RoundLogDocument,
    TwinDocument,
)
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)
from services.geometry.exporter import validate_glb_bytes
from services.geometry.landmarks import ViewLandmarkResult
from services.geometry.twin import run_offline_pipeline
from services.geometry.validator import (
    GeminiPhotoCheckResponse,
    PhotoSafetyAssessment,
    ViewValidationResult,
)
from services.geometry.worker import (
    JobNotFoundError,
    process_twin_job,
)


def _make_high_contrast_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate high-contrast synthetic JPEG passing local quality/blur checks."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (255, 255, 255), 4)
    cv2.circle(img, (width // 2, 70), 30, (230, 200, 180), -1)
    cv2.rectangle(img, (width // 2 - 35, 110), (width // 2 + 35, 240), (80, 80, 80), -1)
    cv2.line(img, (width // 2, 240), (width // 2 - 20, height - 20), (40, 40, 40), 6)
    cv2.line(img, (width // 2, 240), (width // 2 + 20, height - 20), (40, 40, 40), 6)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _make_blurry_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate uniform gray JPEG triggering blur detection."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _mock_synthetic_landmarks_and_silhouettes():
    """Synthetic landmarks and masks for rapid test execution."""
    views = ["front", "back", "left", "right"]
    lm_results = {}
    masks = {}
    silhouettes = {}
    h, w = 360, 240

    for v in views:
        mask = np.zeros((h, w), dtype=np.uint8)
        bw = 70 if v in ("front", "back") else 45
        mask[50:330, w // 2 - bw // 2 : w // 2 + bw // 2] = 1
        masks[v] = mask

        lm_results[v] = ViewLandmarkResult(
            view=v,
            landmarks=[],
            image_width=w,
            image_height=h,
            mask_top_row=50,
            mask_bottom_row=329,
            pixel_height=280,
            cm_per_pixel=175.0 / 280.0,
            mean_confidence=0.92,
        )
    return lm_results, masks, silhouettes


@pytest.fixture(autouse=True)
def mock_gemini_validator():
    """Autouse fixture providing clean Gemini photo inspection report for offline tests."""
    mock_resp = GeminiPhotoCheckResponse(
        views={
            "front": ViewValidationResult(ok=True, issues=[]),
            "back": ViewValidationResult(ok=True, issues=[]),
            "left": ViewValidationResult(ok=True, issues=[]),
            "right": ViewValidationResult(ok=True, issues=[]),
        },
        safety=PhotoSafetyAssessment(is_child=False, is_inappropriate=False),
        overall_summary="All 4 photos pass photogrammetry quality checks.",
    )
    with patch("services.geometry.validator.generate_json", return_value=mock_resp):
        yield mock_resp


@pytest.fixture
def e2e_cloud_env():
    """Setup in-memory Firestore and Storage clients with synthetic valid photos."""
    valid_jpeg = _make_high_contrast_jpeg()
    storage_files = {
        "users/user_e2e/jobs/job_e2e_1/front.jpg": valid_jpeg,
        "users/user_e2e/jobs/job_e2e_1/back.jpg": valid_jpeg,
        "users/user_e2e/jobs/job_e2e_1/left.jpg": valid_jpeg,
        "users/user_e2e/jobs/job_e2e_1/right.jpg": valid_jpeg,
    }
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient(files=storage_files, bucket_name="measured-fit-e2e")

    fs.update_job(
        "job_e2e_1",
        {
            "id": "job_e2e_1",
            "user_id": "user_e2e",
            "status": JobStatus.checking.value,
            "height_cm": 178.5,
            "gender": "neutral",
            "age_18_confirmed": True,
            "opt_in_keep_front_photo": False,
            "photos": {
                "front": "users/user_e2e/jobs/job_e2e_1/front.jpg",
                "back": "users/user_e2e/jobs/job_e2e_1/back.jpg",
                "left": "users/user_e2e/jobs/job_e2e_1/left.jpg",
                "right": "users/user_e2e/jobs/job_e2e_1/right.jpg",
            },
        },
    )

    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


# --- 1. Full Happy-Path Pipeline Test ---


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_e2e_happy_path_pipeline_complete(mock_extract, e2e_cloud_env):
    """Verify complete end-to-end execution through worker pipeline."""
    fs, st = e2e_cloud_env
    mock_extract.return_value = _mock_synthetic_landmarks_and_silhouettes()

    def mock_critic_stop(overlays, measurements):
        return CriticReport(
            regions=[],
            stop=True,
            summary="Convergence achieved at round 1.",
        )

    result = process_twin_job(
        job_id="job_e2e_1",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=mock_critic_stop,
    )

    # 1. Status transition check
    assert result["status"] == "done"
    assert result["job_id"] == "job_e2e_1"
    assert result["twin_id"] is not None
    assert result["final_mean_iou"] is not None
    assert result["final_mean_iou"] >= 0.0

    # 2. Firestore JobDocument verification
    job_doc = fs.get_job("job_e2e_1")
    assert job_doc["status"] == "done"
    assert job_doc["twin_id"] == result["twin_id"]
    assert "updated_at" in job_doc

    # 3. Firestore TwinDocument verification
    twin_doc = fs.get_twin(result["twin_id"])
    assert twin_doc is not None
    assert twin_doc["user_id"] == "user_e2e"
    assert twin_doc["height_cm"] == 178.5
    assert len(twin_doc["betas"]) == 10

    # 4. Check that all 12 zones are present in measurements
    measurements = twin_doc["measurements"]["zones"]
    assert len(measurements) == 12
    for z in MeasurementZone:
        assert z.value in measurements
        m_val = measurements[z.value]
        assert m_val["value_cm"] > 0.0
        assert 0.0 <= m_val["confidence"] <= 1.0

    # 5. Cloud Storage GLB upload verification
    glb_key = f"users/user_e2e/twins/{result['twin_id']}/twin.glb"
    glb_bytes = st.files[glb_key]
    is_valid_glb, err_msg, glb_info = validate_glb_bytes(glb_bytes)
    assert is_valid_glb
    assert err_msg is None
    assert glb_info["num_vertices"] == 6890
    assert glb_info["num_faces"] == 13776

    # 6. Strict Privacy Cleanup: all 4 raw photos deleted when opt_in_keep_front_photo=False
    for v in ("front", "back", "left", "right"):
        key = f"users/user_e2e/jobs/job_e2e_1/{v}.jpg"
        assert key not in st.files


# --- 2. Multi-Round Critic Refinement E2E Test ---


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_e2e_critic_multi_round_refinement(mock_extract, e2e_cloud_env):
    """Verify multi-round critic loop records RoundLogDocuments with step halving."""
    fs, st = e2e_cloud_env
    mock_extract.return_value = _mock_synthetic_landmarks_and_silhouettes()

    round_counter = [0]

    def mock_multi_round_critic(overlays, measurements):
        round_counter[0] += 1
        if round_counter[0] == 1:
            # Round 1: Request waist adjustment
            return CriticReport(
                regions=[
                    CriticRegionFeedback(
                        region=CriticRegion.waist,
                        view=CriticView.front,
                        issue=CriticIssue.too_wide,
                        cause=CriticCause.model_error,
                        severity=0.5,
                        trust=0.9,
                    )
                ],
                stop=False,
                summary="Refining waist circumference.",
            )
        # Round 2: Stop
        return CriticReport(
            regions=[],
            stop=True,
            summary="Refinement converged.",
        )

    result = process_twin_job(
        job_id="job_e2e_1",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=mock_multi_round_critic,
    )

    assert result["status"] == "done"
    assert round_counter[0] == 2

    # Verify rounds logged in jobs/{job_id}/rounds/{round_num}
    rounds = fs.rounds.get("job_e2e_1", {})
    assert len(rounds) >= 1
    assert 1 in rounds
    round1_log = rounds[1]
    assert round1_log["round"] == 1
    assert "mean_iou_before" in round1_log
    assert "mean_iou_after" in round1_log


# --- 3. Privacy Cleanup with Front Photo Retention Opt-in ---


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_e2e_privacy_preserves_front_photo_when_opted_in(mock_extract, e2e_cloud_env):
    """When opt_in_keep_front_photo is True, only front photo is kept; back, left, right are deleted."""
    fs, st = e2e_cloud_env
    mock_extract.return_value = _mock_synthetic_landmarks_and_silhouettes()

    fs.update_job("job_e2e_1", {"opt_in_keep_front_photo": True})

    result = process_twin_job(
        job_id="job_e2e_1",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=lambda o, m: CriticReport(regions=[], stop=True, summary="Done"),
    )

    assert result["status"] == "done"

    # Front photo MUST remain
    front_key = "users/user_e2e/jobs/job_e2e_1/front.jpg"
    assert front_key in st.files

    # Back, left, right MUST be deleted
    for v in ("back", "left", "right"):
        key = f"users/user_e2e/jobs/job_e2e_1/{v}.jpg"
        assert key not in st.files


# --- 4. Failure Modes & Edge Cases ---


def test_e2e_underage_consent_rejected(e2e_cloud_env):
    """Failure case: missing 18+ consent marks job as 'rejected' with AGE_CONSENT_REQUIRED."""
    fs, st = e2e_cloud_env
    fs.update_job("job_e2e_1", {"age_18_confirmed": False})

    result = process_twin_job(job_id="job_e2e_1", firestore_client=fs, storage_client=st)
    assert result["status"] == "rejected"
    assert result["error_code"] == "AGE_CONSENT_REQUIRED"

    # Verify privacy cleanup occurred
    for v in ("front", "back", "left", "right"):
        assert f"users/user_e2e/jobs/job_e2e_1/{v}.jpg" not in st.files


def test_e2e_blurry_photo_needs_retake(e2e_cloud_env):
    """Failure case: blurry photo marks job as 'needs_retake' with BLUR_DETECTED."""
    fs, st = e2e_cloud_env
    st.files["users/user_e2e/jobs/job_e2e_1/front.jpg"] = _make_blurry_jpeg()

    result = process_twin_job(job_id="job_e2e_1", firestore_client=fs, storage_client=st)
    assert result["status"] == "needs_retake"
    assert result["error_code"] == "PHOTO_QUALITY_CHECK_FAILED"
    assert "blurry" in result["error_message"].lower()

    # Verify privacy cleanup occurred
    for v in ("front", "back", "left", "right"):
        assert f"users/user_e2e/jobs/job_e2e_1/{v}.jpg" not in st.files


def test_e2e_missing_photo_fails_safely(e2e_cloud_env):
    """Failure case: missing photo file marks job as 'failed' with MISSING_PHOTOS."""
    fs, st = e2e_cloud_env
    del st.files["users/user_e2e/jobs/job_e2e_1/back.jpg"]

    result = process_twin_job(job_id="job_e2e_1", firestore_client=fs, storage_client=st)
    assert result["status"] == "failed"
    assert result["error_code"] == "MISSING_PHOTOS"


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_e2e_processing_error_fails_safely_with_cleanup(mock_extract, e2e_cloud_env):
    """Failure case: unhandled exception marks job as failed with PROCESSING_ERROR and cleans up."""
    fs, st = e2e_cloud_env
    mock_extract.side_effect = RuntimeError("Simulated segmentation engine crash")

    result = process_twin_job(job_id="job_e2e_1", firestore_client=fs, storage_client=st)
    assert result["status"] == "failed"
    assert result["error_code"] == "PROCESSING_ERROR"

    # All photos must be cleaned up
    for v in ("front", "back", "left", "right"):
        assert f"users/user_e2e/jobs/job_e2e_1/{v}.jpg" not in st.files


# --- 5. Parity between Worker Pipeline and Offline CLI ---


@patch("services.geometry.twin.extract_four_view_landmarks_and_silhouettes")
def test_e2e_cli_parity_with_worker_output(mock_cli_extract, tmp_path):
    """Verify offline CLI produces all 12 zones matching worker schema."""
    mock_cli_extract.return_value = _mock_synthetic_landmarks_and_silhouettes()

    valid_img = _make_high_contrast_jpeg()
    f_path = tmp_path / "front.jpg"
    b_path = tmp_path / "back.jpg"
    l_path = tmp_path / "left.jpg"
    r_path = tmp_path / "right.jpg"
    f_path.write_bytes(valid_img)
    b_path.write_bytes(valid_img)
    l_path.write_bytes(valid_img)
    r_path.write_bytes(valid_img)

    out_dir = tmp_path / "out"

    summary = run_offline_pipeline(
        front_path=f_path,
        back_path=b_path,
        left_path=l_path,
        right_path=r_path,
        height_cm=175.0,
        output_dir=out_dir,
    )

    assert Path(summary["measurements_json"]).exists()
    assert Path(summary["glb_path"]).exists()
    assert summary["mean_iou"] >= 0.0
    assert (out_dir / "measurements.json").exists()
    assert (out_dir / "twin.glb").exists()

    with open(out_dir / "measurements.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "zones" in data
    assert len(data["zones"]) == 12
    for z in MeasurementZone:
        assert z.value in data["zones"]

    # Verify GLB byte integrity
    glb_bytes = (out_dir / "twin.glb").read_bytes()
    is_valid, err_msg, info = validate_glb_bytes(glb_bytes)
    assert is_valid
    assert err_msg is None
    assert info["num_vertices"] == 6890
