"""Tests for services/api/main.py twin endpoints."""

from __future__ import annotations

import base64
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from fitcore.schemas import JobStatus
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)
from services.api.main import app

client = TestClient(app)


def _make_dummy_base64_jpeg() -> str:
    # Minimal JPEG header
    raw = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9])
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}"


@pytest.fixture
def mock_cloud():
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient()
    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


def auth(uid: str) -> dict:
    return {"Authorization": f"Bearer dev:{uid}"}


def test_healthz_endpoint():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "api"}


def test_create_twin_job_rejects_unconfirmed_age(mock_cloud):
    payload = {
        "user_id": "test_user_1",
        "height_cm": 175.0,
        "gender": "neutral",
        "age_18_confirmed": False,
        "opt_in_keep_front_photo": False,
        "photo_front": _make_dummy_base64_jpeg(),
        "photo_back": _make_dummy_base64_jpeg(),
        "photo_left": _make_dummy_base64_jpeg(),
        "photo_right": _make_dummy_base64_jpeg(),
    }
    resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))
    assert resp.status_code == 400
    assert "18+" in resp.json()["detail"]


def test_create_twin_job_rejects_missing_photo(mock_cloud):
    payload = {
        "user_id": "test_user_1",
        "height_cm": 175.0,
        "gender": "neutral",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": False,
        "photo_front": "",
        "photo_back": _make_dummy_base64_jpeg(),
        "photo_left": _make_dummy_base64_jpeg(),
        "photo_right": _make_dummy_base64_jpeg(),
    }
    resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))
    assert resp.status_code == 422
    assert "Missing or empty photo" in resp.json()["detail"]


def test_create_twin_job_success_with_mocked_worker(mock_cloud):
    """create_twin_job_endpoint now returns as soon as the job document exists (status
    "checking"), and the actual pipeline runs via FastAPI BackgroundTasks -- see
    services/api/main.py::_run_twin_job_in_background. TestClient runs background tasks before
    client.post() returns, so by the time this assertion runs, the (mocked) worker has already
    executed and the job document reflects its outcome -- this is what GET /api/twin/jobs/{id}
    polling would see in the real app."""
    fs, _ = mock_cloud
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "test_user_1",
        "height_cm": 178.5,
        "gender": "female",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": True,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }

    def fake_worker(job_id, firestore_client):
        # Mirrors the real worker's contract: it reports outcomes by writing the job document,
        # not merely by a return value, since nothing reads this function's return value anymore.
        firestore_client.update_job(job_id, {"status": JobStatus.done.value, "twin_id": "twin_mock456"})
        return {"status": JobStatus.done.value, "job_id": job_id, "twin_id": "twin_mock456", "final_mean_iou": 0.942}

    with patch("services.api.main.process_twin_job", side_effect=fake_worker):
        resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "checking"  # immediate response: accepted, not finished
    assert data["twin_id"] is None
    job_id = data["job_id"]
    assert job_id

    job = fs.get_job(job_id)
    assert job["status"] == "done"
    assert job["twin_id"] == "twin_mock456"


def test_create_twin_job_ignores_spoofed_user_id_and_uses_the_authenticated_caller(mock_cloud):
    """A signed-in caller cannot attribute a job to a different user_id by putting one in the
    request body -- the job is always created for the authenticated uid, here "real_caller",
    even though the body claims to be "someone_else"."""
    fs, _ = mock_cloud
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "someone_else",
        "height_cm": 170.0,
        "gender": "neutral",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": False,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }
    mock_result = {"status": JobStatus.done.value, "job_id": "job_spoof_test", "twin_id": None}

    with patch("services.api.main.process_twin_job", return_value=mock_result):
        resp = client.post("/api/twin/jobs", json=payload, headers=auth("real_caller"))
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert fs.get_job(job_id)["user_id"] == "real_caller"


def test_create_twin_job_returns_immediately_even_if_worker_raises(mock_cloud):
    """The worker's own failure must never surface as an HTTP 500 to the client -- the whole
    point of this fix is that create_twin_job_endpoint returns as soon as the job document
    exists, before the (possibly slow, possibly failing) background task even runs. Previously
    process_twin_job ran inline and its exception was caught by the endpoint and turned into a
    500; now that exception happens off the request/response cycle entirely, and is recorded onto
    the job document instead (see _run_twin_job_in_background's defensive backstop)."""
    fs, _ = mock_cloud
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "test_user_1",
        "height_cm": 170.0,
        "gender": "neutral",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": False,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }

    def fake_worker_that_raises(job_id, firestore_client):
        raise RuntimeError("simulated worker crash")

    with patch("services.api.main.process_twin_job", side_effect=fake_worker_that_raises):
        resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "checking"
    job_id = data["job_id"]

    job = fs.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "PROCESSING_ERROR"
    assert "simulated worker crash" in job["error_message"]

    # The polling endpoint a real client would use sees the same clear failure.
    polled = client.get(f"/api/twin/jobs/{job_id}", headers=auth("test_user_1")).json()
    assert polled["status"] == "failed"


def test_background_worker_skips_jobs_not_in_checking_status(mock_cloud):
    """_run_twin_job_in_background must not re-run process_twin_job for a job that is no longer
    at its initial "checking" status -- guards against the same job_id being processed more than
    once (e.g. a stray duplicate schedule), which could otherwise redundantly re-run the whole
    pipeline or clobber an already-recorded result."""
    fs, _ = mock_cloud
    from services.api.main import _run_twin_job_in_background

    fs.update_job("job_already_done", {"id": "job_already_done", "status": JobStatus.done.value, "user_id": "alice"})

    with patch("services.api.main.process_twin_job") as mock_worker:
        _run_twin_job_in_background("job_already_done", fs)
        mock_worker.assert_not_called()


def test_create_twin_job_requires_authentication():
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "test_user_1",
        "height_cm": 170.0,
        "gender": "neutral",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": False,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }
    resp = client.post("/api/twin/jobs", json=payload)  # no Authorization header
    assert resp.status_code == 401


def test_get_twin_job_requires_authentication(mock_cloud):
    resp = client.get("/api/twin/jobs/job_nonexistent")  # no Authorization header
    assert resp.status_code == 401


def test_get_twin_job_not_found(mock_cloud):
    resp = client.get("/api/twin/jobs/job_nonexistent", headers=auth("anyone"))
    assert resp.status_code == 404


def test_get_twin_job_owner_can_access_their_own_job(mock_cloud):
    fs, _ = mock_cloud
    fs.update_job("job_owned", {"id": "job_owned", "user_id": "alice", "status": "done"})
    resp = client.get("/api/twin/jobs/job_owned", headers=auth("alice"))
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"


def test_get_twin_job_cross_user_access_is_forbidden(mock_cloud):
    fs, _ = mock_cloud
    fs.update_job("job_owned_by_alice", {"id": "job_owned_by_alice", "user_id": "alice", "status": "done"})
    resp = client.get("/api/twin/jobs/job_owned_by_alice", headers=auth("bob"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "forbidden"


def test_get_twin_document_requires_authentication(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent")  # no Authorization header
    assert resp.status_code == 401


def test_get_twin_document_not_found(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent", headers=auth("anyone"))
    assert resp.status_code == 404


def test_get_twin_document_owner_can_access_their_own_twin(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin("twin_owned", {"id": "twin_owned", "user_id": "alice", "height_cm": 165.0})
    resp = client.get("/api/twin/twin_owned", headers=auth("alice"))
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"


def test_get_twin_document_cross_user_access_is_forbidden_and_does_not_leak_data(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin(
        "twin_secret",
        {"id": "twin_secret", "user_id": "alice", "height_cm": 165.0, "betas": [0.42] * 10},
    )
    resp = client.get("/api/twin/twin_secret", headers=auth("bob"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "forbidden"
    # the 403 body itself must not leak any of alice's private data
    assert "0.42" not in resp.text
    assert "165" not in resp.text


def test_create_twin_job_returns_final_iou_from_persisted_twin_document(mock_cloud):
    """final_iou (from /v1/me/twin, which is what the frontend reads once the job is done) must
    reflect the twin document's actual final_mean_iou value (the realistic path: worker.py
    persists it on the twin document), not silently come back null due to a key-name mismatch.
    This moved here from the job-creation response: create_twin_job_endpoint now only returns an
    immediate accepted/"checking" acknowledgement (see test_create_twin_job_success_with_mocked_worker)
    -- the actual twin result is only available once the background job finishes, via polling."""
    fs, _ = mock_cloud
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "test_user_1",
        "height_cm": 170.0,
        "gender": "neutral",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": False,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }

    fs.create_twin("twin_iou_test", {"id": "twin_iou_test", "user_id": "test_user_1", "final_mean_iou": 0.873})

    def fake_worker(job_id, firestore_client):
        firestore_client.update_job(job_id, {"status": JobStatus.done.value, "twin_id": "twin_iou_test"})

    with patch("services.api.main.process_twin_job", side_effect=fake_worker):
        resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "checking"

    twin = client.get("/v1/me/twin", headers=auth("test_user_1")).json()
    assert twin["final_iou"] == 0.873


def test_create_twin_job_returns_streamable_glb_url_not_raw_gs_uri(mock_cloud):
    """glb_url (from /v1/me/twin once the job is done) must be a path the browser can fetch
    directly (/api/twin/{id}/glb), not the raw gs:// storage URI a browser cannot load. Moved
    here from the job-creation response for the same reason as the final_iou test above."""
    fs, st = mock_cloud
    dummy_img = _make_dummy_base64_jpeg()
    payload = {
        "user_id": "test_user_1",
        "height_cm": 178.5,
        "gender": "female",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": True,
        "photo_front": dummy_img,
        "photo_back": dummy_img,
        "photo_left": dummy_img,
        "photo_right": dummy_img,
    }

    glb_storage_url = st.upload_bytes("users/u1/twins/twin_mock456/twin.glb", b"fake-glb-bytes")
    fs.create_twin("twin_mock456", {"id": "twin_mock456", "user_id": "test_user_1", "glb_storage_url": glb_storage_url})

    def fake_worker(job_id, firestore_client):
        firestore_client.update_job(job_id, {"status": JobStatus.done.value, "twin_id": "twin_mock456"})

    with patch("services.api.main.process_twin_job", side_effect=fake_worker):
        resp = client.post("/api/twin/jobs", json=payload, headers=auth("test_user_1"))
    assert resp.status_code == 200

    twin = client.get("/v1/me/twin", headers=auth("test_user_1")).json()
    assert twin["glb_url"] == "/api/twin/twin_mock456/glb"


def test_get_twin_glb_requires_authentication(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent/glb")  # no Authorization header
    assert resp.status_code == 401


def test_get_twin_glb_streams_bytes_with_correct_content_type(mock_cloud):
    fs, st = mock_cloud
    glb_storage_url = st.upload_bytes("users/u1/twins/twin_abc/twin.glb", b"fake-glb-bytes")
    fs.create_twin("twin_abc", {"id": "twin_abc", "user_id": "alice", "glb_storage_url": glb_storage_url})

    resp = client.get("/api/twin/twin_abc/glb", headers=auth("alice"))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content == b"fake-glb-bytes"


def test_get_twin_glb_cross_user_access_is_forbidden(mock_cloud):
    fs, st = mock_cloud
    glb_storage_url = st.upload_bytes("users/u1/twins/twin_abc2/twin.glb", b"fake-glb-bytes")
    fs.create_twin("twin_abc2", {"id": "twin_abc2", "user_id": "alice", "glb_storage_url": glb_storage_url})

    resp = client.get("/api/twin/twin_abc2/glb", headers=auth("bob"))
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "forbidden"
    assert resp.content != b"fake-glb-bytes"


def test_get_twin_glb_not_found_for_missing_twin(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent/glb", headers=auth("anyone"))
    assert resp.status_code == 404


def test_get_twin_glb_not_found_when_no_glb_asset(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin("twin_no_glb", {"id": "twin_no_glb", "user_id": "alice"})
    resp = client.get("/api/twin/twin_no_glb/glb", headers=auth("alice"))
    assert resp.status_code == 404


def test_get_twin_glb_returns_404_when_storage_object_missing(mock_cloud):
    """Regression: when a twin references a glb_storage_url but the backing file is missing
    from storage (e.g. stale probe or deleted file), GET /api/twin/{id}/glb must return 404
    (not 500 internal server error)."""
    fs, _ = mock_cloud
    fs.create_twin(
        "twin_missing_glb",
        {
            "id": "twin_missing_glb",
            "user_id": "alice",
            "glb_storage_url": "gs://local-dev-bucket/users/alice/twins/twin_missing_glb/twin.glb",
        },
    )
    resp = client.get("/api/twin/twin_missing_glb/glb", headers=auth("alice"))
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_me_twin_preserves_twin_data_when_glb_file_missing(mock_cloud):
    """Regression: if a twin's backing GLB file is missing from storage, the twin record
    itself (measurements, zones, etc.) must remain accessible via /v1/me/twin without breaking."""
    fs, _ = mock_cloud
    fs.create_twin(
        "twin_probe_without_file",
        {
            "id": "twin_probe_without_file",
            "user_id": "alice",
            "glb_storage_url": "gs://local-dev-bucket/users/alice/twins/twin_probe_without_file/twin.glb",
            "measurements": {"zones": {"chest": {"value_cm": 95.0, "confidence": 0.9, "source": "smpl_fit"}}},
            "created_at": "2026-10-18T10:00:00Z",
        },
    )
    resp = client.get("/v1/me/twin", headers=auth("alice"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["twin_id"] == "twin_probe_without_file"
    assert data["measurements"]["zones"]["chest"]["value_cm"] == 95.0
    assert data["glb_url"] == "/api/twin/twin_probe_without_file/glb"

    # Attempting to stream the missing GLB returns 404, not 500
    glb_resp = client.get(data["glb_url"], headers=auth("alice"))
    assert glb_resp.status_code == 404
