"""Tests for services/api/apilib/fit_mesh.py (Prompt 5, Phase 2:
Body-Mesh Fit Heatmap and Pose-Deltas API).

Verifies: authentication required, twin ownership enforced, standing-pose
compatibility, seated-pose strain (from the existing Fit Engine), missing
pose data reported as unavailable (never fabricated), GLB retrieval
round-trip, and expiry of temporary mesh assets.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.geometry.cloud_client import InMemoryFirestoreClient, InMemoryStorageClient, set_cloud_clients

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_cloud():
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient()
    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


def _auth(uid: str) -> dict:
    return {"Authorization": f"Bearer dev:{uid}"}


def _twin_measurements() -> dict:
    zones = {
        "waist": {"value_cm": 82, "confidence": 0.9, "source": "test"},
        "hip": {"value_cm": 98, "confidence": 0.9, "source": "test"},
        "thigh": {"value_cm": 56, "confidence": 0.9, "source": "test"},
        "inseam": {"value_cm": 78, "confidence": 0.9, "source": "test"},
        "back_rise": {"value_cm": 26, "confidence": 0.9, "source": "test"},
    }
    return {"zones": zones}


def _make_twin(fs, twin_id="twin_1", user_id="alice", gender="neutral"):
    fs.create_twin(twin_id, {
        "id": twin_id, "user_id": user_id, "gender": gender,
        "height_cm": 175.0, "betas": [0.0] * 10, "measurements": _twin_measurements(),
    })
    return twin_id


def _trousers_product(size="M", fit_type="regular", **row_overrides):
    row = {"waist": 83, "hip": 112, "thigh": 77, "inseam": 79, "back_rise": 27}
    row.update(row_overrides)
    return {
        "id": "p1", "brand": "b", "title": "t", "category": "trousers", "fit_type": fit_type,
        "fabric": [{"fiber": "cotton", "percent": 100}],
        "size_chart": {"sizes": {size: row}, "chart_type": "garment"},
    }


# --------------------------------------------------------------------------- POST /fit/mesh


def test_fit_mesh_requires_auth(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs)
    resp = client.post("/fit/mesh", json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M"})
    assert resp.status_code == 401


def test_fit_mesh_rejects_other_users_twin(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.post(
        "/fit/mesh",
        json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M"},
        headers=_auth("bob"),
    )
    assert resp.status_code == 403


def test_fit_mesh_standing_pose_returns_zone_report_and_glb_url(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.post(
        "/fit/mesh",
        json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M", "pose": "standing"},
        headers=_auth("alice"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["pose"] == "standing"
    assert data["zone_report"]["waist"]["verdict"] == "tight"
    assert data["zone_report"]["waist"]["status"] == "colored"
    assert data["zone_report"]["hip"]["verdict"] == "ideal"
    assert data["zone_report"]["inseam"]["status"] == "unsupported_no_vertex_mapping"
    assert data["glb_url"].startswith("/fit/mesh/") and data["glb_url"].endswith("/glb")


def test_fit_mesh_seated_pose_reflects_existing_fit_engine_strain(mock_cloud):
    """Reuses the proven scenario from
    packages/fitcore/tests/fit/test_engine.py::
    test_pose_slim_trouser_zero_thigh_ease_strains_when_seated.
    """
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    body = {
        "twin_id": "twin_1",
        "product": _trousers_product(waist=86, hip=106, thigh=56, inseam=79, back_rise=27, fit_type="slim"),
        "size": "M",
        "pose": "seated",
        "pose_deltas": {"poses": {"seated": {"minutes": 120, "delta_cm": {"thigh": 3.0}}}},
    }
    resp = client.post("/fit/mesh", json=body, headers=_auth("alice"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["zone_report"]["thigh"]["strain"] is True


def test_fit_mesh_missing_pose_data_reported_unavailable_not_fabricated(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    body = {
        "twin_id": "twin_1",
        "product": _trousers_product(),
        "size": "M",
        "pose": "seated",
        # Only thigh has a declared seated delta; waist has none.
        "pose_deltas": {"poses": {"seated": {"minutes": 60, "delta_cm": {"thigh": 2.0}}}},
    }
    resp = client.post("/fit/mesh", json=body, headers=_auth("alice"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["zone_report"]["waist"]["status"] == "pose_data_unavailable"
    assert data["zone_report"]["waist"]["strain"] is None


def test_fit_mesh_twin_without_betas_is_422(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin("twin_empty", {"id": "twin_empty", "user_id": "alice", "gender": "neutral", "measurements": _twin_measurements()})
    resp = client.post(
        "/fit/mesh", json={"twin_id": "twin_empty", "product": _trousers_product(), "size": "M"}, headers=_auth("alice"),
    )
    assert resp.status_code == 422


def test_fit_mesh_unknown_twin_is_404(mock_cloud):
    resp = client.post(
        "/fit/mesh", json={"twin_id": "twin_ghost", "product": _trousers_product(), "size": "M"}, headers=_auth("alice"),
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------- GET /fit/mesh/{id}/glb


def test_fit_mesh_glb_roundtrip(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    create_resp = client.post(
        "/fit/mesh", json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M"}, headers=_auth("alice"),
    )
    assert create_resp.status_code == 200
    glb_url = create_resp.json()["glb_url"]

    glb_resp = client.get(glb_url, headers=_auth("alice"))
    assert glb_resp.status_code == 200
    assert glb_resp.headers["content-type"] == "model/gltf-binary"
    assert glb_resp.content[:4] == b"glTF"


def test_fit_mesh_glb_requires_auth(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    create_resp = client.post(
        "/fit/mesh", json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M"}, headers=_auth("alice"),
    )
    glb_url = create_resp.json()["glb_url"]
    resp = client.get(glb_url)
    assert resp.status_code == 401


def test_fit_mesh_glb_rejects_other_user(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    create_resp = client.post(
        "/fit/mesh", json={"twin_id": "twin_1", "product": _trousers_product(), "size": "M"}, headers=_auth("alice"),
    )
    glb_url = create_resp.json()["glb_url"]
    resp = client.get(glb_url, headers=_auth("bob"))
    assert resp.status_code == 403


def test_fit_mesh_glb_expired_is_410(mock_cloud):
    fs, st = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    expired_mesh_id = f"twin_1.faketoken.{int(time.time()) - 10}"
    st.upload_bytes(f"fit_meshes/{expired_mesh_id}.glb", b"stale-bytes", content_type="model/gltf-binary")

    resp = client.get(f"/fit/mesh/{expired_mesh_id}/glb", headers=_auth("alice"))
    assert resp.status_code == 410


def test_fit_mesh_glb_unknown_mesh_id_is_404(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    future_mesh_id = f"twin_1.faketoken.{int(time.time()) + 900}"
    resp = client.get(f"/fit/mesh/{future_mesh_id}/glb", headers=_auth("alice"))
    assert resp.status_code == 404


# --------------------------------------------------------------------------- GET /twin/{id}/pose-deltas


def test_pose_deltas_requires_auth(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.get("/twin/twin_1/pose-deltas")
    assert resp.status_code == 401


def test_pose_deltas_rejects_other_user(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.get("/twin/twin_1/pose-deltas", headers=_auth("bob"))
    assert resp.status_code == 403


def test_pose_deltas_reports_length_zones_and_flags_unmodeled_zones(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.get("/twin/twin_1/pose-deltas?pose=seated", headers=_auth("alice"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["pose"] == "seated"
    for zone in ("torso_length", "back_rise", "inseam", "garment_length"):
        assert zone in data["measurements"]
        assert "delta_cm" in data["measurements"][zone]
    for zone in ("chest", "waist", "hip", "neck", "thigh"):
        assert zone in data["unmodeled_zones"]
        assert zone not in data["measurements"]


def test_pose_deltas_standing_vs_standing_is_zero_delta(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.get("/twin/twin_1/pose-deltas?pose=standing", headers=_auth("alice"))
    assert resp.status_code == 200
    data = resp.json()
    for zone_data in data["measurements"].values():
        assert zone_data["delta_cm"] == pytest.approx(0.0, abs=1e-6)


def test_pose_deltas_invalid_pose_is_422(mock_cloud):
    fs, _ = mock_cloud
    _make_twin(fs, twin_id="twin_1", user_id="alice")
    resp = client.get("/twin/twin_1/pose-deltas?pose=crouching", headers=_auth("alice"))
    assert resp.status_code == 422
