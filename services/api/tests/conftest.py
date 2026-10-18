import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parents[1]
ROOT = API_DIR.parents[1]
sys.path.insert(0, str(API_DIR))

# services/api/main.py validates its auth config (apilib.auth.validate_security_config) at import
# time, and several test files import it directly at module/collection time -- before any fixture,
# even an autouse one, gets a chance to run. setdefault (not monkeypatch, which only applies once a
# fixture executes) guarantees that import always sees a safe config, for every file in this
# directory, regardless of which fixture (if any) a given test file uses. Individual tests still
# override these per-test via monkeypatch (see `_env` below) for their own assertions.
os.environ.setdefault("AUTH_MODE", "dev")
os.environ.setdefault("CONSENT_JWT_SECRET", "test-secret-" + "x" * 24)

# The geometry/API dev-fallback cloud clients are disk-backed by default (see
# services/geometry/local_persistence.py), so any test that reaches get_firestore_client()/
# get_storage_client() without injecting its own client would otherwise write real files into
# services/geometry/.data. Tests must stay hermetic and ephemeral: pin them to the pure in-RAM
# store, which is exactly the behavior they were written against. Set at module level for the
# same reason as the auth config above -- it must apply before any import-time client creation.
os.environ.setdefault("GEOMETRY_STORE", "memory")

ORIGIN = "https://shop.example"
KEY = "pk_test_demo"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in ("LLM_BACKEND", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("CONSENT_JWT_SECRET", "test-secret-" + "x" * 24)
    monkeypatch.setenv("API_CORS_ORIGINS", "https://app.example")
    monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)


@pytest.fixture
def app_module():
    spec = importlib.util.spec_from_file_location("api_main_conftest", API_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store():
    from apilib.stores import MemoryStore

    s = MemoryStore()
    s.set("merchants", "m_demo", {"name": "Demo Threads Co.", "publishable_key": KEY, "allowed_origins": [ORIGIN]})
    s.set("merchants", "m_other", {"name": "Other Shop", "publishable_key": "pk_test_other", "allowed_origins": ["https://other.example"]})
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    s.set("twins", "alice", {"owner_id": "alice", "measurements": {"zones": twin["zones"]}})
    s.set("twins", "bob", {"owner_id": "bob", "measurements": {"zones": {**twin["zones"], "chest": {"value_cm": 112, "confidence": 0.9, "source": "simulated"}}}})
    return s


@pytest.fixture
def client(app_module, store, monkeypatch):
    import apilib.stores as stores
    from apilib.auth import RateLimiter, get_limiter

    monkeypatch.setattr(stores, "_store", store)          # the CORS middleware reads the global store
    app_module.app.dependency_overrides[stores.get_store] = lambda: store
    limiter = RateLimiter()
    app_module.app.dependency_overrides[get_limiter] = lambda: limiter
    return TestClient(app_module.app)


def user(uid: str) -> dict:
    return {"Authorization": f"Bearer dev:{uid}"}


@pytest.fixture
def product():
    p = json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8"))
    return {"size_chart": p["size_chart"], "category": p["category"], "fit_type": p["fit_type"], "fabric": p["fabric"],
            "stretch_class": p["stretch_class"]}


@pytest.fixture
def consent(client):
    """A fresh consent for alice at the demo merchant: returns the token response."""
    r = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("alice"))
    assert r.status_code == 200, r.text
    return r.json()
