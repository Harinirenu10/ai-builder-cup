"""Unit tests for services/geometry/worker.py (Cloud Tasks Worker Pipeline).

Verifies:
1. End-to-end twin generation: status transitions checking -> measuring -> fitting -> refining -> done.
2. Firestore twin document creation with betas, measurements, and GLB storage URL.
3. Round logging in jobs/{job_id}/rounds/{round_num}.
4. Missing job raising JobNotFoundError (HTTP 404 in endpoint).
5. Missing photo handling marking job as failed with MISSING_PHOTOS.
6. Adult consent failure marking job as rejected (not failed) with AGE_CONSENT_REQUIRED.
7. Blurry photo quality failure marking job as needs_retake (not failed).
8. Unexpected internal error handling marking job as failed with PROCESSING_ERROR.
9. Privacy cleanup: deleting all 4 photos when opt_in_keep_front_photo=False.
10. Privacy cleanup: preserving front photo and deleting back/left/right when opt_in_keep_front_photo=True.
11. FastAPI testclient integration with Cloud Tasks headers.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fitcore.schemas import (
    CriticReport,
    JobDocument,
    JobStatus,
    Measurement,
    Measurements,
    MeasurementZone,
)
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)
from services.geometry.landmarks import PoseLandmarkPoint, ViewLandmarkResult
from services.geometry.main import app
from services.geometry.worker import (
    JobNotFoundError,
    process_twin_job,
)


def _make_valid_test_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate high-contrast synthetic JPEG passing local quality/blur checks."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (255, 255, 255), 4)
    cv2.circle(img, (width // 2, 70), 30, (230, 200, 180), -1)
    cv2.rectangle(img, (width // 2 - 35, 110), (width // 2 + 35, 240), (80, 80, 80), -1)
    cv2.line(img, (width // 2, 240), (width // 2 - 20, height - 20), (40, 40, 40), 6)
    cv2.line(img, (width // 2, 240), (width // 2 + 20, height - 20), (40, 40, 40), 6)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _make_blurry_test_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate uniform gray JPEG with low Laplacian variance (triggers blur)."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _mock_critic_stop(overlays: dict, measurements: Measurements) -> CriticReport:
    """Mock critic callback terminating the refinement loop immediately."""
    return CriticReport(
        regions=[],
        stop=True,
        summary="Synthetic test fit is satisfactory and meets visual alignment criteria.",
    )


def _create_mock_landmarks_and_silhouettes():
    """Lightweight synthetic landmarks and masks for rapid test execution."""
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


def _create_mock_landmarks_and_silhouettes_with_chest_geometry(arms_merged: bool):
    """Like _create_mock_landmarks_and_silhouettes, but with real shoulder/hip
    landmark points populated on the front view (the base fixture uses an
    empty landmarks list, which silently bypasses check_chest_arm_separation
    entirely -- see worker.py's `if l_sh and r_sh and l_hip and r_hip:` guard
    -- so it cannot exercise the chest arm-separation gate at all).

    When arms_merged=False, the front mask's chest row has a clear background
    gap between the torso and a separate arm blob (gate should pass). When
    True, the arm is painted directly connected to the torso with no gap
    (gate should fail with needs_retake), reproducing the real finding from
    this investigation.
    """
    views = ["front", "back", "left", "right"]
    lm_results = {}
    masks = {}
    silhouettes = {}
    h, w = 360, 240

    # Front-view landmark geometry: shoulders near y=70, hips near y=180,
    # torso columns roughly [90,150] (matching the front mask's 70px-wide
    # torso below, centered at w//2=120).
    y_shoulders, y_hips = 70.0, 180.0
    l_sh_x, r_sh_x = 150.0, 90.0
    l_hip_x, r_hip_x = 145.0, 95.0

    def _pt(index, name, px, py):
        return PoseLandmarkPoint(
            index=index, name=name, x=px / w, y=py / h, z=0.0,
            visibility=0.95, presence=0.95, pixel_x=px, pixel_y=py,
        )

    front_landmarks = [
        _pt(11, "LEFT_SHOULDER", l_sh_x, y_shoulders),
        _pt(12, "RIGHT_SHOULDER", r_sh_x, y_shoulders),
        _pt(23, "LEFT_HIP", l_hip_x, y_hips),
        _pt(24, "RIGHT_HIP", r_hip_x, y_hips),
    ]

    for v in views:
        mask = np.zeros((h, w), dtype=np.uint8)
        bw = 70 if v in ("front", "back") else 45
        mask[50:330, w // 2 - bw // 2 : w // 2 + bw // 2] = 1

        if v == "front":
            # Chest row ~= round(70 + 0.22*(180-70)) = 94.
            if arms_merged:
                mask[80:110, 40:200] = 1  # arm directly connects into the torso
            else:
                mask[80:110, 20:70] = 1    # left arm, separated by a real gap
                mask[80:110, 170:220] = 1  # right arm, separated by a real gap

        masks[v] = mask

        lm_results[v] = ViewLandmarkResult(
            view=v,
            landmarks=front_landmarks if v == "front" else [],
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
def mock_gemini_photo_check():
    """Autouse fixture providing clean Gemini photo inspection report for offline tests."""
    from services.geometry.validator import (
        GeminiPhotoCheckResponse,
        PhotoSafetyAssessment,
        ViewValidationResult,
    )
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
def mock_cloud_environment():
    """Setup in-memory Firestore and Storage clients with synthetic valid photos."""
    valid_jpeg = _make_valid_test_jpeg()
    storage_files = {
        "users/user_42/jobs/job_101/front.jpg": valid_jpeg,
        "users/user_42/jobs/job_101/back.jpg": valid_jpeg,
        "users/user_42/jobs/job_101/left.jpg": valid_jpeg,
        "users/user_42/jobs/job_101/right.jpg": valid_jpeg,
    }
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient(files=storage_files, bucket_name="measured-fit-test")

    # Seed valid job
    fs.update_job(
        "job_101",
        {
            "id": "job_101",
            "user_id": "user_42",
            "status": JobStatus.checking.value,
            "height_cm": 178.0,
            "gender": "neutral",
            "age_18_confirmed": True,
            "opt_in_keep_front_photo": False,
            "photos": {
                "front": "users/user_42/jobs/job_101/front.jpg",
                "back": "users/user_42/jobs/job_101/back.jpg",
                "left": "users/user_42/jobs/job_101/left.jpg",
                "right": "users/user_42/jobs/job_101/right.jpg",
            },
        },
    )

    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_success(mock_extract, mock_cloud_environment):
    """End-to-end success test verifying status transitions, twin creation, GLB export, and photo deletion."""
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    result = process_twin_job(
        job_id="job_101",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=_mock_critic_stop,
    )

    # 1. Returned outcome
    assert result["status"] == JobStatus.done.value
    assert "twin_id" in result
    twin_id = result["twin_id"]

    # 2. Firestore Job document
    job = fs.get_job("job_101")
    assert job is not None
    assert job["status"] == JobStatus.done.value
    assert job["twin_id"] == twin_id
    assert job["current_step"] == "completed"

    # 3. Firestore Twin document
    twin = fs.get_twin(twin_id)
    assert twin is not None
    assert twin["user_id"] == "user_42"
    assert twin["height_cm"] == 178.0
    assert len(twin["betas"]) == 10
    assert "chest" in twin["measurements"]["zones"]
    assert twin["glb_storage_url"].startswith("gs://measured-fit-test/")

    # 4. Refinement round logging
    assert "job_101" in fs.rounds
    assert 1 in fs.rounds["job_101"]
    round_log = fs.rounds["job_101"][1]
    assert round_log["round"] == 1
    assert "Synthetic test fit is satisfactory" in round_log["critic_notes"]

    # 5. GLB file uploaded to storage
    glb_key = f"users/user_42/twins/{twin_id}/twin.glb"
    assert glb_key in st.files
    assert st.files[glb_key].startswith(b"glTF")

    # 6. Privacy cleanup: opt_in_keep_front_photo was False, so ALL 4 photos deleted
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/back.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/left.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/right.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/front.jpg" not in st.files


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_reaches_done_when_critic_times_out_repeatedly(mock_extract, mock_cloud_environment):
    """Regression test for the real demo symptom (job stuck at 'refining' /
    85%, log: 'LLM [gemma4:12b] call failed permanently after 1 attempts: The
    read operation timed out'): exercises the REAL critic default path (no
    critic_llm_fn injected, unlike every other test in this file) with
    fitcore.llm.generate_json patched to always time out. The job must still
    reach JobStatus.done -- using the already-computed, already-valid SMPL fit
    -- not hang, not fail, and not fabricate a critic result. Answers this
    investigation's first question directly: yes, the job correctly reaches a
    terminal state afterward.
    """
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    with patch("fitcore.llm.generate_json", side_effect=TimeoutError("The read operation timed out")):
        result = process_twin_job(job_id="job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.done.value
    assert "twin_id" in result

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.done.value
    assert job["twin_id"] == result["twin_id"]

    twin = fs.get_twin(result["twin_id"])
    assert twin is not None and "chest" in twin["measurements"]["zones"]

    # Critic unavailability must be visible in the round log, not silently hidden.
    round_log = fs.rounds["job_101"][1]
    assert round_log["accepted"] is False
    assert "unavailable" in round_log["critic_notes"].lower()

    # Privacy cleanup must still run even though critic review never completed.
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_preserves_front_photo_when_opted_in(mock_extract, mock_cloud_environment):
    """When opt_in_keep_front_photo=True, front photo must be preserved while others are deleted."""
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    # Update job to opt in
    fs.update_job("job_101", {"opt_in_keep_front_photo": True})

    result = process_twin_job(
        job_id="job_101",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=_mock_critic_stop,
    )

    assert result["status"] == JobStatus.done.value

    # Front photo preserved
    assert "users/user_42/jobs/job_101/front.jpg" not in st.deleted_paths
    assert "users/user_42/jobs/job_101/front.jpg" in st.files

    # Back, left, right photos deleted
    assert "users/user_42/jobs/job_101/back.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/left.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/right.jpg" in st.deleted_paths
    assert "users/user_42/jobs/job_101/back.jpg" not in st.files


def test_process_twin_job_missing_job(mock_cloud_environment):
    """Missing job must raise JobNotFoundError."""
    fs, st = mock_cloud_environment
    with pytest.raises(JobNotFoundError):
        process_twin_job("job_nonexistent", firestore_client=fs, storage_client=st)


def test_process_twin_job_missing_photo_marks_failed(mock_cloud_environment):
    """Missing photo must mark job failed with MISSING_PHOTOS."""
    fs, st = mock_cloud_environment
    # Remove one required photo from storage
    del st.files["users/user_42/jobs/job_101/left.jpg"]

    result = process_twin_job("job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.failed.value
    assert result["error_code"] == "MISSING_PHOTOS"

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.failed.value
    assert job["error_code"] == "MISSING_PHOTOS"


# ----------------------------------------------------------------------------------
# Privacy regression: persisted job-record photo cleanup.
#
# mock_cloud_environment (above) seeds the job's 'photos' field with STORAGE PATH
# references, which _cleanup_storage_photos already handles correctly (confirmed by
# test_process_twin_job_success and test_process_twin_job_preserves_front_photo_when_opted_in
# above). It does not reproduce the actual bug: services/api/main.py's
# create_twin_job_endpoint stores the raw submitted photo data (a data: URL) directly
# as the job document's 'photos' field value, not a storage path. A real end-to-end run
# with real photos confirmed those raw bytes were still present on the job record after
# the job reached 'done', because _cleanup_storage_photos only ever deletes STORAGE
# blobs -- it never touches the Firestore job document's own 'photos' field. These
# tests use INLINE data: URLs (matching the real submission format) and inspect the
# persisted job record itself, not just storage, to guard against that gap.


@pytest.fixture
def mock_cloud_environment_inline_photos():
    """Like mock_cloud_environment, but the job's 'photos' field holds inline data:
    URLs -- the actual format services/api/main.py's create_twin_job_endpoint writes
    -- instead of storage path references."""
    valid_jpeg = _make_valid_test_jpeg()
    inline_photo = "data:image/jpeg;base64," + base64.b64encode(valid_jpeg).decode()

    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient(files={}, bucket_name="measured-fit-test")

    fs.update_job(
        "job_inline",
        {
            "id": "job_inline",
            "user_id": "user_42",
            "status": JobStatus.checking.value,
            "height_cm": 178.0,
            "gender": "neutral",
            "age_18_confirmed": True,
            "opt_in_keep_front_photo": False,
            "photos": {
                "front": inline_photo,
                "back": inline_photo,
                "left": inline_photo,
                "right": inline_photo,
            },
        },
    )

    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st, inline_photo


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_success_clears_inline_photos_from_job_record(
    mock_extract, mock_cloud_environment_inline_photos
):
    """On success with opt_in_keep_front_photo=False, the persisted job record's
    'photos' field must end up empty -- not just storage blobs deleted."""
    fs, st, _inline_photo = mock_cloud_environment_inline_photos
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    result = process_twin_job(
        job_id="job_inline",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=_mock_critic_stop,
    )

    assert result["status"] == JobStatus.done.value

    job = fs.get_job("job_inline")
    assert job["status"] == JobStatus.done.value  # job status itself is unaffected
    assert "twin_id" in job and job["twin_id"] == result["twin_id"]  # twin data preserved
    assert job["photos"] == {}, f"expected all raw photos cleared from job record, got: {job['photos']}"


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_success_keeps_only_front_inline_photo_when_opted_in(
    mock_extract, mock_cloud_environment_inline_photos
):
    """On success with opt_in_keep_front_photo=True, the persisted job record must
    retain ONLY the front photo's raw data; back/left/right must be cleared."""
    fs, st, inline_photo = mock_cloud_environment_inline_photos
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()
    fs.update_job("job_inline", {"opt_in_keep_front_photo": True})

    result = process_twin_job(
        job_id="job_inline",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=_mock_critic_stop,
    )

    assert result["status"] == JobStatus.done.value

    job = fs.get_job("job_inline")
    assert job["status"] == JobStatus.done.value
    assert job["photos"] == {"front": inline_photo}, (
        f"expected only front photo retained on job record, got: {job['photos']}"
    )


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_processing_error_clears_inline_photos_from_job_record(
    mock_extract, mock_cloud_environment_inline_photos
):
    """On an unexpected crash with opt_in_keep_front_photo=False, the persisted job
    record's 'photos' field must still end up empty -- cleanup on failure must reach
    the job record too, not just storage."""
    fs, st, _inline_photo = mock_cloud_environment_inline_photos
    mock_extract.side_effect = RuntimeError("MediaPipe unexpected internal error")

    result = process_twin_job("job_inline", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.failed.value
    assert result["error_code"] == "PROCESSING_ERROR"

    job = fs.get_job("job_inline")
    assert job["status"] == JobStatus.failed.value  # failure status preserved
    assert job["error_code"] == "PROCESSING_ERROR"
    assert job["photos"] == {}, f"expected all raw photos cleared from job record, got: {job['photos']}"


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_processing_error_keeps_only_front_inline_photo_when_opted_in(
    mock_extract, mock_cloud_environment_inline_photos
):
    """On an unexpected crash with opt_in_keep_front_photo=True, the persisted job
    record must retain ONLY the front photo; back/left/right must be cleared even on
    the failure path."""
    fs, st, inline_photo = mock_cloud_environment_inline_photos
    mock_extract.side_effect = RuntimeError("MediaPipe unexpected internal error")
    fs.update_job("job_inline", {"opt_in_keep_front_photo": True})

    result = process_twin_job("job_inline", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.failed.value

    job = fs.get_job("job_inline")
    assert job["status"] == JobStatus.failed.value
    assert job["error_code"] == "PROCESSING_ERROR"
    assert job["photos"] == {"front": inline_photo}, (
        f"expected only front photo retained on job record, got: {job['photos']}"
    )


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_validation_failure_clears_inline_photos_from_job_record(
    mock_extract, mock_cloud_environment_inline_photos
):
    """Early rejection paths (here: the chest arm-separation gate) must also clear
    the job record's raw photos, even when opted in -- matching
    _cleanup_storage_photos's existing opt_in_keep_front=False behavior at this same
    call site: a rejected capture's "front" photo is not worth retaining either."""
    fs, st, _inline_photo = mock_cloud_environment_inline_photos
    fs.update_job("job_inline", {"opt_in_keep_front_photo": True})
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes_with_chest_geometry(arms_merged=True)

    result = process_twin_job("job_inline", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.needs_retake.value
    assert result["error_code"] == "PHOTO_QUALITY_CHECK_FAILED"

    job = fs.get_job("job_inline")
    assert job["status"] == JobStatus.needs_retake.value
    assert job["photos"] == {}, f"expected all raw photos cleared from job record, got: {job['photos']}"


def test_process_twin_job_underage_consent_rejected(mock_cloud_environment):
    """Missing 18+ consent must mark job as rejected (not failed)."""
    fs, st = mock_cloud_environment
    fs.update_job("job_101", {"age_18_confirmed": False})

    result = process_twin_job("job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.rejected.value
    assert result["error_code"] == "AGE_CONSENT_REQUIRED"

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.rejected.value
    assert job["error_code"] == "AGE_CONSENT_REQUIRED"

    # All photos cleaned up on rejection
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths


def test_process_twin_job_blurry_photo_needs_retake(mock_cloud_environment):
    """Blurry photo failing quality checks must mark job as needs_retake (not failed)."""
    fs, st = mock_cloud_environment
    # Replace front photo with uniform gray image (low Laplacian variance)
    st.files["users/user_42/jobs/job_101/front.jpg"] = _make_blurry_test_jpeg()

    result = process_twin_job("job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.needs_retake.value

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.needs_retake.value

    # All photos cleaned up on retake request
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_chest_arms_merged_triggers_needs_retake(mock_extract, mock_cloud_environment):
    """When the front mask shows the arm directly connected to the torso at
    chest height (no background gap), the worker must stop at needs_retake
    with an arms_against_body-style instruction, BEFORE computing
    measurements or running SMPL fitting -- not silently proceed with a
    contaminated chest value. Reproduces the real failure mode found testing
    with a real consented photo and a synthetic SMPL ground-truth round trip.
    """
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes_with_chest_geometry(arms_merged=True)

    result = process_twin_job("job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.needs_retake.value
    assert result["error_code"] == "PHOTO_QUALITY_CHECK_FAILED"
    assert "arm" in result["error_message"].lower()

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.needs_retake.value
    assert job["current_step"] == "validation_failed"
    # Must not have progressed to fitting/done.
    assert job.get("twin_id") is None

    # Privacy cleanup still runs on this retake path.
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_chest_arms_separated_proceeds_normally(mock_extract, mock_cloud_environment):
    """When the front mask shows a genuine background gap between arm and
    torso at chest height, the gate must pass and the job must proceed all
    the way to completion as usual -- confirming the gate does not block
    well-posed photos."""
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes_with_chest_geometry(arms_merged=False)

    result = process_twin_job(
        "job_101",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=_mock_critic_stop,
    )

    assert result["status"] == JobStatus.done.value
    assert "twin_id" in result


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_processing_error_fails_safely(mock_extract, mock_cloud_environment):
    """Runtime processing exception must be caught and mark job failed with PROCESSING_ERROR."""
    fs, st = mock_cloud_environment
    mock_extract.side_effect = RuntimeError("MediaPipe unexpected internal error")

    result = process_twin_job("job_101", firestore_client=fs, storage_client=st)

    assert result["status"] == JobStatus.failed.value
    assert result["error_code"] == "PROCESSING_ERROR"
    assert "MediaPipe unexpected internal error" in result["error_message"]

    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.failed.value
    assert job["error_code"] == "PROCESSING_ERROR"

    # Photos cleaned up even on crash
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_fastapi_endpoint_post_twin_process(mock_extract, mock_cloud_environment):
    """Test POST /twin/process/{job_id} through FastAPI TestClient."""
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    with patch("services.geometry.worker.run_critic_refinement_loop") as mock_critic:
        from services.geometry.fitting import fit_smpl_shape
        dummy_fit = fit_smpl_shape(Measurements(zones={}), target_height_cm=178.0, max_steps=5)
        mock_critic.return_value = (dummy_fit, [], {})

        client = TestClient(app)
        response = client.post(
            "/twin/process/job_101",
            headers={
                "X-CloudTasks-QueueName": "fit-jobs-queue",
                "X-CloudTasks-TaskName": "task-test-8899",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert "twin_id" in data


def test_fastapi_endpoint_404_on_missing_job(mock_cloud_environment):
    """POST /twin/process/{job_id} on nonexistent job returns 404."""
    client = TestClient(app)
    response = client.post("/twin/process/job_does_not_exist")
    assert response.status_code == 404


@patch("services.geometry.worker.extract_four_view_landmarks_and_silhouettes")
def test_process_twin_job_reaches_done_when_critic_exceeds_overall_timeout(mock_extract, mock_cloud_environment):
    """Regression test: when the critic call or its retries exceed the strict
    overall timeout (e.g. repeated socket timeouts against an unresponsive
    Gemma/Ollama model), process_twin_job must not hang at 'refining' / 85%.
    It must safely fall back to the already-computed, valid 3D fit, continue to
    GLB export and upload, record critic unavailability in round logs, reach
    JobStatus.done, and clean up temporary storage photos.
    """
    fs, st = mock_cloud_environment
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()

    def slow_critic_simulating_timeout(overlays, meas):
        time.sleep(1.5)
        return CriticReport(regions=[], stop=True, summary="Late critic report.")

    t0 = time.time()
    result = process_twin_job(
        job_id="job_101",
        firestore_client=fs,
        storage_client=st,
        critic_llm_fn=slow_critic_simulating_timeout,
        critic_timeout_s=0.2,
    )
    elapsed = time.time() - t0

    # Ensure critic review was cut off by timeout rather than running the full 1.5s
    round_log = fs.rounds["job_101"][1]
    assert round_log["accepted"] is False
    assert round_log["critic_latency_ms"] < 1200, f"Critic latency {round_log['critic_latency_ms']}ms exceeded timeout limit"
    assert result["status"] == JobStatus.done.value
    assert "twin_id" in result
    assert "users/user_42/twins/" in result["glb_storage_url"]
    assert result["glb_storage_url"].endswith("/twin.glb")

    # Terminal status in Firestore
    job = fs.get_job("job_101")
    assert job["status"] == JobStatus.done.value
    assert job["current_step"] == "completed"
    assert job["twin_id"] == result["twin_id"]

    # Twin document successfully created with valid measurements
    twin = fs.get_twin(result["twin_id"])
    assert twin is not None
    assert "chest" in twin["measurements"]["zones"]
    assert twin["glb_storage_url"] == result["glb_storage_url"]

    # Round log records unavailability
    round_log = fs.rounds["job_101"][1]
    assert round_log["accepted"] is False
    assert "unavailable" in round_log["critic_notes"].lower()
    assert "timed out" in round_log["critic_notes"].lower()

    # Privacy cleanup completed
    assert "users/user_42/jobs/job_101/front.jpg" in st.deleted_paths
