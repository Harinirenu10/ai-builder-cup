"""Tests for GET /v1/me/twin customer endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apilib.stores import get_store
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)
from services.api.main import app

client = TestClient(app)


def _auth(uid: str) -> dict[str, str]:
    return {"Authorization": f"Bearer dev:{uid}"}


@pytest.fixture
def mock_cloud():
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient()
    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


def test_me_twin_requires_auth():
    resp = client.get("/v1/me/twin")
    assert resp.status_code == 401
    assert "Sign in first" in str(resp.json())


def test_me_twin_returns_404_when_no_twin(mock_cloud):
    resp = client.get("/v1/me/twin", headers=_auth("user_without_twin"))
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "no_twin"


def test_me_twin_returns_store_seeded_twin():
    store = get_store()
    uid = "demo-shopper"
    store.set(
        "twins",
        uid,
        {
            "owner_id": uid,
            "measurements": {
                "zones": {
                    "chest": {"value_cm": 100.0, "confidence": 0.95, "source": "simulated"},
                    "waist": {"value_cm": 82.0, "confidence": 0.92, "source": "simulated"},
                }
            },
        },
    )

    resp = client.get("/v1/me/twin", headers=_auth(uid))
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == uid
    assert "chest" in data["measurements"]["zones"]
    assert data["measurements"]["zones"]["chest"]["value_cm"] == 100.0


def test_me_twin_returns_firestore_twin(mock_cloud):
    fs, st = mock_cloud
    uid = "shopper_real"
    twin_id = "twin_shopper_real_01"

    fs.create_twin(
        twin_id,
        {
            "id": twin_id,
            "user_id": uid,
            "status": "done",
            "glb_storage_url": "users/shopper_real/twins/twin_01/twin.glb",
            "final_mean_iou": 0.945,
            "created_at": "2026-10-17T20:00:00Z",
            "measurements": {
                "zones": {
                    "chest": {"value_cm": 98.5, "confidence": 0.96, "source": "smpl"},
                    "waist": {"value_cm": 80.0, "confidence": 0.94, "source": "smpl"},
                }
            },
        },
    )

    resp = client.get("/v1/me/twin", headers=_auth(uid))
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == uid
    assert data["twin_id"] == twin_id
    assert data["glb_url"] == f"/api/twin/{twin_id}/glb"
    assert data["final_iou"] == 0.945
    assert data["measurements"]["zones"]["chest"]["value_cm"] == 98.5


def test_me_twin_does_not_leak_other_user_twin(mock_cloud):
    fs, _ = mock_cloud
    fs.create_twin(
        "twin_other_user",
        {
            "id": "twin_other_user",
            "user_id": "other_user",
            "status": "done",
            "measurements": {
                "zones": {
                    "chest": {"value_cm": 110.0, "confidence": 0.90, "source": "smpl"},
                }
            },
        },
    )

    # Calling as "new_user" should get 404, never "other_user"'s twin
    resp = client.get("/v1/me/twin", headers=_auth("new_user"))
    assert resp.status_code == 404


def test_progress_requires_auth():
    resp = client.get("/v1/me/progress")
    assert resp.status_code == 401
    resp = client.put("/v1/me/progress", json={"active_tab": "capture"})
    assert resp.status_code == 401


def test_progress_lifecycle_and_user_isolation():
    import uuid
    suffix = uuid.uuid4().hex[:8]
    alice = f"alice_progress_{suffix}"
    bob = f"bob_progress_{suffix}"

    # Alice gets default progress
    r = client.get("/v1/me/progress", headers=_auth(alice))
    assert r.status_code == 200
    assert r.json()["user_id"] == alice
    assert r.json()["active_tab"] == "twin"

    # Alice saves progress
    payload = {
        "active_tab": "capture",
        "height_cm": "178.5",
        "gender": "male",
        "age_18_confirmed": True,
        "opt_in_keep_front_photo": True,
        "captured_views": ["front", "back"],
        "last_job_id": "job_123456",
    }
    r = client.put("/v1/me/progress", json=payload, headers=_auth(alice))
    assert r.status_code == 200
    saved = r.json()
    assert saved["user_id"] == alice
    assert saved["height_cm"] == "178.5"
    assert saved["captured_views"] == ["front", "back"]
    assert saved["last_job_id"] == "job_123456"

    # Re-reading Alice returns saved progress
    r2 = client.get("/v1/me/progress", headers=_auth(alice))
    assert r2.status_code == 200
    assert r2.json()["height_cm"] == "178.5"

    # Bob's progress is isolated from Alice's
    rb = client.get("/v1/me/progress", headers=_auth(bob))
    assert rb.status_code == 200
    assert rb.json()["user_id"] == bob
    assert rb.json()["height_cm"] == "175.0"  # default, untouched by Alice

