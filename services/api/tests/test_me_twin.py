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



# ----------------------------------------------------------- POST /v1/me/twin/{twin_id}/save
# The pipeline already persists every successful twin; this endpoint records the owner's
# explicit intent to keep it ("Save My 3D Twin" in the customer app) and is what drives the
# Saved/Save button state the UI shows. It must be idempotent (no duplicate saves, stable
# saved_at) and owner-only, like every other twin endpoint.


def _firestore_twin(uid: str, twin_id: str, **extra) -> dict:
    doc = {
        "id": twin_id,
        "user_id": uid,
        "glb_storage_url": f"users/{uid}/twins/{twin_id}/twin.glb",
        "final_mean_iou": 0.42,
        "created_at": "2026-10-18T12:00:00+00:00",
        "measurements": {"zones": {"chest": {"value_cm": 98.5, "confidence": 0.9, "source": "smpl_fit"}}},
    }
    doc.update(extra)
    return doc


def test_save_twin_marks_it_saved_and_is_reflected_by_me_twin(mock_cloud):
    fs, _ = mock_cloud
    uid = "saver_1"
    twin_id = "twin_to_save"
    fs.create_twin(twin_id, _firestore_twin(uid, twin_id))

    # Freshly generated twins start out not-yet-saved, so the UI shows a Save button.
    before = client.get("/v1/me/twin", headers=_auth(uid))
    assert before.status_code == 200
    assert before.json()["saved"] is False
    assert before.json()["saved_at"] is None

    resp = client.post(f"/v1/me/twin/{twin_id}/save", headers=_auth(uid))
    assert resp.status_code == 200
    body = resp.json()
    assert body["twin_id"] == twin_id
    assert body["saved"] is True
    assert body["already_saved"] is False
    assert body["saved_at"]

    after = client.get("/v1/me/twin", headers=_auth(uid))
    assert after.status_code == 200
    assert after.json()["saved"] is True
    assert after.json()["saved_at"] == body["saved_at"]
    # Saving must not disturb the twin's actual data.
    assert after.json()["measurements"]["zones"]["chest"]["value_cm"] == 98.5
    assert after.json()["glb_url"] == f"/api/twin/{twin_id}/glb"


def test_save_twin_is_idempotent_and_avoids_duplicate_saves(mock_cloud):
    fs, _ = mock_cloud
    uid = "saver_2"
    twin_id = "twin_double_click"
    fs.create_twin(twin_id, _firestore_twin(uid, twin_id))

    first = client.post(f"/v1/me/twin/{twin_id}/save", headers=_auth(uid))
    assert first.status_code == 200
    assert first.json()["already_saved"] is False
    original_saved_at = first.json()["saved_at"]

    # A double-click / retried request must not create a second save or move the timestamp.
    second = client.post(f"/v1/me/twin/{twin_id}/save", headers=_auth(uid))
    assert second.status_code == 200
    assert second.json()["saved"] is True
    assert second.json()["already_saved"] is True
    assert second.json()["saved_at"] == original_saved_at
    assert fs.get_twin(twin_id)["saved_at"] == original_saved_at


def test_save_twin_requires_auth_and_ownership(mock_cloud):
    fs, _ = mock_cloud
    owner = "saver_owner"
    twin_id = "twin_owned"
    fs.create_twin(twin_id, _firestore_twin(owner, twin_id))

    anon = client.post(f"/v1/me/twin/{twin_id}/save")
    assert anon.status_code == 401

    intruder = client.post(f"/v1/me/twin/{twin_id}/save", headers=_auth("someone_else"))
    assert intruder.status_code == 403
    assert intruder.json()["detail"]["code"] == "forbidden"
    # The twin must remain untouched by the rejected attempt.
    assert fs.get_twin(twin_id).get("saved") in (None, False)

    missing = client.post("/v1/me/twin/twin_does_not_exist/save", headers=_auth(owner))
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "twin_not_found"
