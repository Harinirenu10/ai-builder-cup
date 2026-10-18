"""Integration tests: /v1/fit-query against a REAL Body Twin from the Prompt 1 geometry
pipeline (Firestore twins/{twin_id}, services/geometry/cloud_client.py), not the Store's
seeded twins/{uid} stand-in used elsewhere in this test suite.

Covers the fix in apilib/fitquery.py::evaluate_for_uid / _real_twin_for_uid: the Store is
tried first (so every pre-existing seeded-twin test keeps working unchanged), and Firestore
is used only as a fallback when the uid has no Store twin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.geometry.cloud_client import InMemoryFirestoreClient, set_cloud_clients
from tests.conftest import KEY, ORIGIN, user

ROOT = Path(__file__).resolve().parents[3]
TWIN_ZONES = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]


def _query(client, product, token, *, key=KEY, origin=ORIGIN):
    headers = {"X-Merchant-Key": key}
    if origin:
        headers["Origin"] = origin
    return client.post("/v1/fit-query", json={"product": product, "consent_token": token}, headers=headers)


def _consent_for(client, uid: str) -> str:
    r = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user(uid))
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _real_twin(user_id: str, created_at: str, chest_cm: float = 96.0) -> dict:
    zones = {z: dict(v) for z, v in TWIN_ZONES.items()}
    zones["chest"]["value_cm"] = chest_cm
    return {
        "id": f"twin_{user_id}_{created_at}",
        "user_id": user_id,
        "gender": "neutral",
        "height_cm": 170.0,
        "betas": [],
        "measurements": {"zones": zones},
        "created_at": created_at,
    }


@pytest.fixture
def fresh_firestore():
    """A real (Prompt 1 convention) Firestore twin store, isolated per test."""
    fs = InMemoryFirestoreClient()
    set_cloud_clients(firestore_client=fs, storage_client=None)
    yield fs
    set_cloud_clients(firestore_client=None, storage_client=None)


def test_fit_query_uses_real_twin_when_no_store_twin_exists(client, product, fresh_firestore):
    """Acceptance test: create a real twin through the API-layer storage path, then call
    /v1/fit-query for the same authenticated user and confirm it is actually used."""
    fresh_firestore.create_twin("twin_carol_1", _real_twin("carol", "2026-01-01T00:00:00+00:00"))
    token = _consent_for(client, "carol")

    resp = _query(client, product, token)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Same product + the same chest=96 measurement as the Store-seeded "alice" twin ->
    # the same recommendation, proving evaluate() actually ran against real data, not a stub.
    assert body["recommended_size"] == "L"
    assert body["zone_verdicts"]


def test_fit_query_missing_twin_still_404s_no_twin(client, product, fresh_firestore):
    """A uid with no twin in either store must still get the documented no_twin error, not a crash."""
    token = _consent_for(client, "nobody_has_a_twin")
    resp = _query(client, product, token)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_twin"


def test_fit_query_uses_the_latest_of_multiple_real_twins(client, product, fresh_firestore):
    """A user who recaptured their twin has more than one twins/{twin_id} document; the
    most recently created one must be the one used, not an arbitrary or the oldest one."""
    fresh_firestore.create_twin("twin_dana_old", _real_twin("dana", "2025-01-01T00:00:00+00:00", chest_cm=70.0))
    fresh_firestore.create_twin("twin_dana_new", _real_twin("dana", "2026-06-01T00:00:00+00:00", chest_cm=96.0))
    token = _consent_for(client, "dana")

    resp = _query(client, product, token)
    assert resp.status_code == 200, resp.text
    # chest=96 (the newer twin) recommends "L" on this product chart; chest=70 (the older
    # twin) is far too small for any size on it and would not recommend "L".
    assert resp.json()["recommended_size"] == "L"


def test_fit_query_cross_user_access_is_not_possible(client, product, fresh_firestore):
    """A real twin created for one user must never be served for a different uid's fit-query,
    even though the lookup now reaches all the way into Firestore."""
    fresh_firestore.create_twin("twin_carol_1", _real_twin("carol", "2026-01-01T00:00:00+00:00"))
    other_uid_token = _consent_for(client, "someone_else_entirely")

    resp = _query(client, product, other_uid_token)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_twin"


def test_fit_query_store_twin_takes_precedence_over_real_twin(client, product, store, fresh_firestore):
    """Backward compatibility: when a uid has a Store-seeded (demo/test) twin AND a real
    Firestore twin, the existing Store lookup still wins -- unchanged behavior for every
    pre-existing seeded-twin test and demo flow."""
    # "alice" already has a Store twin (chest=96, see conftest.py's `store` fixture).
    # Give her a real Firestore twin with a wildly different chest so precedence is unambiguous.
    fresh_firestore.create_twin("twin_alice_real", _real_twin("alice", "2026-01-01T00:00:00+00:00", chest_cm=10.0))
    r = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("alice"))
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    resp = _query(client, product, token)
    assert resp.status_code == 200, resp.text
    # Same result as the existing Store-only test (test_fit_query.py) -> the Store twin
    # was used, not the real twin with its implausible chest=10 value.
    assert resp.json()["recommended_size"] == "L"
