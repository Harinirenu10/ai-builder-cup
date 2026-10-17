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
    resp = client.post("/api/twin/jobs", json=payload)
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
    resp = client.post("/api/twin/jobs", json=payload)
    assert resp.status_code == 422
    assert "Missing or empty photo" in resp.json()["detail"]


def test_create_twin_job_success_with_mocked_worker(mock_cloud):
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

    # NOTE: process_twin_job (services/geometry/worker.py) returns the key
    # "final_mean_iou", not "final_iou" -- this mock previously used the wrong
    # key name, which coincidentally matched a since-fixed bug in main.py that
    # also read the wrong key, so the two bugs canceled out and this test
    # passed without actually exercising real data flow.
    mock_result = {
        "status": JobStatus.done.value,
        "job_id": "job_mock123",
        "twin_id": "twin_mock456",
        "final_mean_iou": 0.942,
    }

    with patch("services.api.main.process_twin_job", return_value=mock_result):
        resp = client.post("/api/twin/jobs", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["twin_id"] == "twin_mock456"
        assert data["final_iou"] == 0.942
        assert "job_id" in data


def test_get_twin_job_not_found(mock_cloud):
    resp = client.get("/api/twin/jobs/job_nonexistent")
    assert resp.status_code == 404


def test_get_twin_document_not_found(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent")
    assert resp.status_code == 404


def test_create_twin_job_returns_final_iou_from_persisted_twin_document(mock_cloud):
    """final_iou in the job response must reflect the twin document's actual
    final_mean_iou value (the realistic path: worker.py persists it on the twin
    document), not silently come back null due to a key-name mismatch."""
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

    fs.create_twin("twin_iou_test", {"id": "twin_iou_test", "final_mean_iou": 0.873})

    mock_result = {
        "status": JobStatus.done.value,
        "job_id": "job_iou_test",
        "twin_id": "twin_iou_test",
    }

    with patch("services.api.main.process_twin_job", return_value=mock_result):
        resp = client.post("/api/twin/jobs", json=payload)
        assert resp.status_code == 200
        assert resp.json()["final_iou"] == 0.873


def test_create_twin_job_returns_streamable_glb_url_not_raw_gs_uri(mock_cloud):
    """glb_url in the job response must be a path the browser can fetch directly
    (/api/twin/{id}/glb), not the raw gs:// storage URI a browser cannot load."""
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
    fs.create_twin("twin_mock456", {"id": "twin_mock456", "glb_storage_url": glb_storage_url})

    mock_result = {
        "status": JobStatus.done.value,
        "job_id": "job_mock123",
        "twin_id": "twin_mock456",
        "final_iou": 0.942,
    }

    with patch("services.api.main.process_twin_job", return_value=mock_result):
        resp = client.post("/api/twin/jobs", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["glb_url"] == "/api/twin/twin_mock456/glb"


def test_get_twin_glb_streams_bytes_with_correct_content_type(mock_cloud):
    fs, st = mock_cloud
    glb_storage_url = st.upload_bytes("users/u1/twins/twin_abc/twin.glb", b"fake-glb-bytes")
    fs.create_twin("twin_abc", {"id": "twin_abc", "glb_storage_url": glb_storage_url})

    resp = client.get("/api/twin/twin_abc/glb")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content == b"fake-glb-bytes"


def test_get_twin_glb_not_found_for_missing_twin(mock_cloud):
    resp = client.get("/api/twin/twin_nonexistent/glb")
    assert resp.status_code == 404


def test_get_twin_glb_not_found_when_no_glb_asset(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin("twin_no_glb", {"id": "twin_no_glb"})
    resp = client.get("/api/twin/twin_no_glb/glb")
    assert resp.status_code == 404
