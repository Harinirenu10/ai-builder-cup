"""Tests for the demo virtual try-on endpoints (services/api/apilib/tryon.py).

Verifies:
1. The catalog lists exactly the one predefined garment, with the
   visualization-only disclaimer attached.
2. The GLB endpoint is owner-only and signed-in-only, reusing the same
   ownership rules as the existing twin GLB endpoint.
3. An unknown garment id is a 404, never a silently substituted garment.
4. A twin with no fitted shape parameters is a 422, not a blank mesh.
5. The happy path streams a real GLB holding BOTH the body and the garment,
   and the second request for the same twin is served from the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.api.main import app
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)

GARMENT_ID = "demo-threads-classic-crew-tee"

client = TestClient(app)


def auth(uid: str) -> dict:
    return {"Authorization": f"Bearer dev:{uid}"}


@pytest.fixture
def firestore():
    fs = InMemoryFirestoreClient()
    set_cloud_clients(firestore_client=fs, storage_client=InMemoryStorageClient())
    return fs


@pytest.fixture
def owned_twin(firestore):
    twin_id = "twin_tryon_test"
    firestore.create_twin(
        twin_id,
        {
            "twin_id": twin_id,
            "user_id": "alice",
            "gender": "neutral",
            "betas": [0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    return twin_id


def _url(twin_id: str, garment_id: str = GARMENT_ID) -> str:
    return f"/api/twin/{twin_id}/tryon/{garment_id}/glb"


def test_catalog_lists_the_tee_as_a_top():
    resp = client.get("/api/tryon/garments")
    assert resp.status_code == 200
    garments = {g["garment_id"]: g for g in resp.json()["garments"]}
    assert GARMENT_ID in garments
    assert garments[GARMENT_ID]["kind"] == "top"
    # The response itself says it is not a fit answer.
    assert "no fit verdict" in garments[GARMENT_ID]["note"]


def test_glb_requires_sign_in(owned_twin):
    assert client.get(_url(owned_twin)).status_code == 401


def test_glb_rejects_another_users_twin(owned_twin):
    resp = client.get(_url(owned_twin), headers=auth("mallory"))
    assert resp.status_code == 403


def test_unknown_twin_is_404(firestore):
    resp = client.get(_url("twin_does_not_exist"), headers=auth("alice"))
    assert resp.status_code == 404


def test_unknown_garment_is_404_not_a_substitute(owned_twin):
    resp = client.get(_url(owned_twin, "some-retailer-jacket"), headers=auth("alice"))
    assert resp.status_code == 404
    assert GARMENT_ID in resp.text


def test_twin_without_betas_is_422(firestore):
    firestore.create_twin("twin_no_betas", {"twin_id": "twin_no_betas", "user_id": "alice", "betas": []})
    resp = client.get(_url("twin_no_betas"), headers=auth("alice"))
    assert resp.status_code == 422


def test_streams_one_glb_holding_body_and_garment(owned_twin):
    resp = client.get(_url(owned_twin), headers=auth("alice"))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content[:4] == b"glTF"

    scene = trimesh.load(trimesh.util.wrap_as_stream(resp.content), file_type="glb")
    assert set(scene.geometry.keys()) == {"body", "demo_tee"}
    body = scene.geometry["body"]
    garment = scene.geometry["demo_tee"]
    assert body.vertices.shape == (6890, 3)
    assert len(garment.faces) > 5000
    # The garment sits on the body's upper half, not beside it.
    assert float(garment.vertices[:, 1].min()) > 0.4 * float(body.vertices[:, 1].max())

    # Second request for the same twin is byte-identical (served from cache).
    again = client.get(_url(owned_twin), headers=auth("alice"))
    assert again.status_code == 200
    assert again.content == resp.content


def test_unprefixed_path_also_works(owned_twin):
    """The router is mounted twice so the Firebase Hosting /api/** rewrite and a
    direct Cloud Run call both reach it."""
    resp = client.get(f"/twin/{owned_twin}/tryon/{GARMENT_ID}/glb", headers=auth("alice"))
    assert resp.status_code == 200
    assert resp.content[:4] == b"glTF"


# --------------------------------------------------------------------------- bottoms

TROUSERS_ID = "demo-threads-fitted-trousers"


def test_catalog_lists_both_demo_garments_with_their_kind():
    resp = client.get("/api/tryon/garments")
    assert resp.status_code == 200
    garments = {g["garment_id"]: g for g in resp.json()["garments"]}
    assert set(garments) == {GARMENT_ID, TROUSERS_ID}
    assert garments[GARMENT_ID]["kind"] == "top"
    assert garments[TROUSERS_ID]["kind"] == "bottom"
    # Every entry carries the visualization-only disclaimer.
    assert all("no fit verdict" in g["note"] for g in garments.values())
    # Language rule: the copy describes the garment, never the body.
    blurb = garments[TROUSERS_ID]["description"].lower()
    assert "trousers" in blurb
    assert not any(word in blurb for word in ("slim", "flattering", "weight", "figure"))


def test_trousers_glb_requires_sign_in_and_ownership(owned_twin):
    url = _url(owned_twin, TROUSERS_ID)
    assert client.get(url).status_code == 401
    assert client.get(url, headers=auth("mallory")).status_code == 403


def test_trousers_twin_without_betas_is_422(firestore):
    firestore.create_twin("twin_no_betas_b", {"twin_id": "twin_no_betas_b", "user_id": "alice", "betas": []})
    resp = client.get(_url("twin_no_betas_b", TROUSERS_ID), headers=auth("alice"))
    assert resp.status_code == 422


def test_streams_one_glb_holding_body_and_trousers(owned_twin):
    resp = client.get(_url(owned_twin, TROUSERS_ID), headers=auth("alice"))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content[:4] == b"glTF"

    scene = trimesh.load(trimesh.util.wrap_as_stream(resp.content), file_type="glb")
    assert set(scene.geometry.keys()) == {"body", "demo_trousers"}
    body = scene.geometry["body"]
    garment = scene.geometry["demo_trousers"]
    assert body.vertices.shape == (6890, 3)
    assert len(garment.faces) > 3000
    # The trousers sit on the body's lower half, not beside it or on the chest.
    stature = float(body.vertices[:, 1].max())
    assert float(garment.vertices[:, 1].max()) < 0.75 * stature
    assert float(garment.vertices[:, 1].min()) < 0.2 * stature


def test_each_garment_is_cached_separately(owned_twin):
    """Switching garments in the UI must not serve the previous one from cache."""
    tee = client.get(_url(owned_twin, GARMENT_ID), headers=auth("alice"))
    trousers = client.get(_url(owned_twin, TROUSERS_ID), headers=auth("alice"))
    assert tee.status_code == 200 and trousers.status_code == 200
    assert tee.content != trousers.content

    # And each one still comes back byte-identical on a repeat request.
    assert client.get(_url(owned_twin, GARMENT_ID), headers=auth("alice")).content == tee.content
    assert client.get(_url(owned_twin, TROUSERS_ID), headers=auth("alice")).content == trousers.content
