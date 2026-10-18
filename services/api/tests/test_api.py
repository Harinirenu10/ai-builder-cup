import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("api_main", ROOT / "services" / "api" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return TestClient(mod.app)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for v in ("LLM_BACKEND", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "GEMINI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AUTH_MODE", "dev")  # /v1/fit, /v1/biz/verdict and /v1/tailor-card require current_uid


@pytest.fixture
def store(monkeypatch):
    """A fresh Store per test: /v1/fit, /v1/biz/verdict and /v1/tailor-card now look the caller's
    own twin up here (apilib/fitquery.py::resolve_twin_measurements_for_uid) rather than trusting
    a `twin` field in the request body."""
    import apilib.stores as stores_module
    s = stores_module.MemoryStore()
    monkeypatch.setattr(stores_module, "_store", s)
    return s


TWIN_ZONES = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]


def seed_twin(store, uid: str, zones: dict | None = None) -> None:
    store.set("twins", uid, {"owner_id": uid, "measurements": {"zones": zones or TWIN_ZONES}})


@pytest.fixture
def alice(store):
    """alice has a twin matching samples/twin.json (chest=96 -> recommends "L" on the kurta chart)."""
    seed_twin(store, "alice")
    return store


@pytest.fixture
def payload():
    return {"product": json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8"))}


def auth(uid: str = "alice") -> dict:
    return {"Authorization": f"Bearer dev:{uid}"}


# ----------------------------------------------------------------------------- legitimate customer flows


def test_fit_returns_full_result(client, payload, alice):
    r = client.post("/v1/fit", json=payload, headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["recommended_size"] == "L" and body["explanation"] is None


def test_fit_with_reviews_and_poses(client, payload, alice):
    payload["review_signals"] = [{"zone": "chest", "direction": "smaller", "strength": "slight"}]
    payload["pose_deltas"] = {"poses": {"seated": {"minutes": 60, "delta_cm": {"hip": 2.0}}}}
    r = client.post("/v1/fit", json=payload, headers=auth())
    assert r.status_code == 200 and r.json()["review_notes"]


def test_biz_verdict_has_no_measurements(client, payload, alice):
    r = client.post("/v1/biz/verdict", json=payload, headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_size"] == "L" and body["zones"]
    text = json.dumps(body)
    for forbidden in ("ease", "value_cm", "margin", "shrink", "target_", "_cm"):
        assert forbidden not in text
    # no body measurement numbers leak
    for v in TWIN_ZONES.values():
        assert f'"{v["value_cm"]}"' not in text


def test_missing_size_chart_is_422(client, payload, alice):
    payload["product"]["size_chart"] = None
    assert client.post("/v1/fit", json=payload, headers=auth()).status_code == 422


def test_tailor_card_english(client, payload, alice):
    r = client.post("/v1/tailor-card?languages=en", json=payload, headers=auth())
    assert r.status_code == 200 and r.json()["size"] == "L" and set(r.json()["lines"]) == {"en"}


def test_tailor_card_conflict_when_no_size(client, payload, store):
    seed_twin(store, "alice", {**TWIN_ZONES, "chest": {"value_cm": 200, "confidence": 0.9, "source": "x"}})
    assert client.post("/v1/tailor-card?languages=en", json=payload, headers=auth()).status_code == 409


# ----------------------------------------------------------------------------- security audit regression tests
#
# Security-focused audit findings (see eval/results.md): /v1/fit, /v1/biz/verdict and
# /v1/tailor-card previously had no authentication at all, and separately accepted a `twin` field
# directly in the request body -- so any signed-in caller could submit (or simply already know)
# someone else's measurements and get a result for them. Fixed by (1) requiring a signed-in
# caller and (2) removing `twin` from the request entirely: the server now always looks up the
# authenticated uid's own twin (apilib/fitquery.py::resolve_twin_measurements_for_uid), the same
# Store-then-Firestore pattern /v1/fit-query already used. FitRequest no longer HAS a `twin`
# field, so there is no longer any field to put someone else's measurements into.


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_unauthenticated_request_is_rejected(client, payload, alice, path):
    r = client.post(path, json=payload)  # no Authorization header at all
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "auth_required"


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_garbage_bearer_token_is_rejected(client, payload, alice, path):
    r = client.post(path, json=payload, headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_missing_bearer_scheme_is_rejected(client, payload, alice, path):
    """A raw token with no "Bearer " prefix must not be accepted."""
    r = client.post(path, json=payload, headers={"Authorization": "dev:alice"})
    assert r.status_code == 401


def test_dev_auth_mode_is_only_honored_when_auth_mode_env_var_is_exactly_dev(client, payload, alice, monkeypatch):
    """AUTH_MODE=dev is meant for local/dev/test only (never set it in a real deployment -- see
    this turn's separate report on preventing that). There is no code-level guard against it being
    set in production; this test documents that gap, it does not close it. It does confirm the
    narrower, already-safe behavior: the dev shortcut requires the env var to be exactly "dev"."""
    monkeypatch.setenv("AUTH_MODE", "production")  # anything other than the literal string "dev"
    r = client.post("/v1/fit", json=payload, headers=auth())
    assert r.status_code == 401  # "Bearer dev:alice" is rejected once AUTH_MODE is not "dev"


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_user_can_access_their_own_authorized_twin(client, payload, alice, path):
    r = client.post(path, json=payload, headers=auth("alice"))
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_user_cannot_retrieve_another_users_twin_data(client, payload, store, path):
    """bob has a much smaller body than alice's (chest=70 instead of 96); if bob could somehow see
    alice's result instead of his own, he would get alice's "L" (200) every time. Instead bob's
    own, much smaller body gets a different outcome -- either a different recommended size, or (for
    /v1/tailor-card, which rejects a no-fitting-size result outright) a 409 -- proving the
    computation ran against BOB's twin, not alice's, even though both twins exist in the same
    store and alice's `samples/twin.json`-based data is seeded first."""
    seed_twin(store, "alice")
    seed_twin(store, "bob", {**TWIN_ZONES, "chest": {"value_cm": 70.0, "confidence": 0.9, "source": "x"}})
    alice_resp = client.post(path, json=payload, headers=auth("alice"))
    bob_resp = client.post(path, json=payload, headers=auth("bob"))
    assert alice_resp.status_code == 200, alice_resp.text
    assert bob_resp.status_code in (200, 409), bob_resp.text
    assert (alice_resp.status_code, alice_resp.json() if alice_resp.status_code == 200 else None) != (
        bob_resp.status_code, bob_resp.json() if bob_resp.status_code == 200 else None
    )
    if path == "/v1/biz/verdict":
        assert alice_resp.json()["recommended_size"] == "L"
        assert bob_resp.json()["recommended_size"] != "L"
    elif path == "/v1/tailor-card":
        assert bob_resp.status_code == 409  # bob's chest=70 fits no size on this chart
        assert alice_resp.json()["size"] == "L"


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_submitting_a_twin_field_in_the_body_has_no_effect(client, payload, alice, path):
    """FitRequest has no `twin` field; pydantic silently drops unknown fields, so even a client
    that still tries to send someone else's measurements in the body cannot influence the result --
    only alice's own, server-side-looked-up twin (chest=96, recommends "L") is ever used."""
    rigged = {**payload, "twin": {"zones": {"chest": {"value_cm": 999.0, "confidence": 0.9, "source": "x"}}}}
    r = client.post(path, json=rigged, headers=auth("alice"))
    assert r.status_code == 200, r.text
    if path == "/v1/biz/verdict":
        assert r.json()["recommended_size"] == "L"  # alice's real twin, not the rigged chest=999


@pytest.mark.parametrize("path", ["/v1/fit", "/v1/biz/verdict", "/v1/tailor-card"])
def test_missing_twin_produces_a_safe_409_not_a_crash(client, payload, store, path):
    """A signed-in caller with no twin anywhere (not seeded, no real Firestore twin) gets the
    documented no_twin error, matching /v1/fit-query's own convention -- not a 500 or a stack trace."""
    r = client.post(path, json=payload, headers=auth("nobody-has-a-twin"))
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "no_twin"