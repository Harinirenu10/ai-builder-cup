"""Regression tests for readerlib.cache.default_store()'s Firestore -> MemoryStore fallback.

Context: services/reader/main.py's module-level get_cache() calls ReadCache() with no
store argument, which calls default_store(). When GOOGLE_CLOUD_PROJECT is set (as it
would be in most deployed or developer-configured environments) but no usable GCP
credentials are available locally (missing Application Default Credentials, unreachable
Firestore, etc.), the un-fixed code let FirestoreStore()'s exception propagate all the
way up through POST /read as an unhandled 500. services/geometry/cloud_client.py already
has the right pattern for this (try the live client, catch broadly, log, fall back to an
in-memory client) -- default_store() now mirrors it, falling back to MemoryStore()
specifically, per the fix.

The suite's own autouse _env fixture (conftest.py) sets READER_CACHE=memory for every
test, which never exercises this code path at all -- these tests explicitly override that
to exercise the real GOOGLE_CLOUD_PROJECT + auto-mode path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readerlib.cache import FirestoreStore, MemoryStore, default_store


class _FakeCollection:
    def document(self, key):
        raise AssertionError("not exercised in these tests")


class _FakeFirestoreClient:
    """Stands in for google.cloud.firestore.Client without needing real credentials."""

    def __init__(self, project=None):
        self.project = project

    def collection(self, name):
        return _FakeCollection()


def test_firestore_initializes_successfully_and_remains_selected(monkeypatch):
    """When Firestore construction succeeds, default_store() returns a FirestoreStore,
    not a fallback -- the live path must still be preferred when it actually works."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("READER_CACHE", "auto")
    monkeypatch.setattr("google.cloud.firestore.Client", _FakeFirestoreClient)

    store = default_store()

    assert isinstance(store, FirestoreStore)


def test_firestore_initialization_failure_falls_back_to_memory_store(monkeypatch, caplog):
    """When Firestore construction fails (e.g. missing Application Default Credentials,
    the exact failure reproduced live against this environment), default_store() must
    fall back to MemoryStore() instead of letting the exception propagate, and must log
    the real error rather than swallowing it silently."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("READER_CACHE", "auto")

    def _raise_credentials_error(project=None):
        raise RuntimeError("Your default credentials were not found.")

    monkeypatch.setattr("google.cloud.firestore.Client", _raise_credentials_error)

    import logging
    with caplog.at_level(logging.WARNING, logger="reader.cache"):
        store = default_store()

    assert isinstance(store, MemoryStore)
    assert any("default credentials" in rec.message.lower() for rec in caplog.records), (
        "the real Firestore failure must be logged, not swallowed silently"
    )


def test_explicit_firestore_mode_still_raises_on_failure(monkeypatch):
    """Preserve existing behavior: READER_CACHE=firestore is an explicit request for the
    live store, so a construction failure must still raise, not silently degrade."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("READER_CACHE", "firestore")

    def _raise_credentials_error(project=None):
        raise RuntimeError("Your default credentials were not found.")

    monkeypatch.setattr("google.cloud.firestore.Client", _raise_credentials_error)

    with pytest.raises(RuntimeError, match="default credentials"):
        default_store()


def test_post_read_proceeds_locally_without_gcp_credentials(monkeypatch):
    """The confirmed live blocker: POST /read must not crash when GOOGLE_CLOUD_PROJECT is
    set but Firestore credentials are unavailable. Drives the real app's get_cache() (no
    _cache monkeypatch, unlike the other endpoint tests in test_pipeline.py), so this
    actually exercises default_store() through the live HTTP endpoint, not a bypass of it.
    """
    import importlib.util

    from fastapi.testclient import TestClient

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("READER_CACHE", "auto")

    def _raise_credentials_error(project=None):
        raise RuntimeError("Your default credentials were not found.")

    monkeypatch.setattr("google.cloud.firestore.Client", _raise_credentials_error)

    spec = importlib.util.spec_from_file_location(
        "reader_main_cache_fallback", Path(__file__).resolve().parents[1] / "main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = TestClient(mod.app)

    # A URL this harness can resolve without a real external fetch or LLM call: READER_
    # ALLOW_PRIVATE=1 is set by the autouse _env fixture but no local server is running
    # here, so this exercises get_cache() -> default_store() during request handling and
    # must come back as a handled "error" response, not an unhandled 500 from the cache
    # layer itself.
    r = client.post("/read", json={"url": "http://127.0.0.1:1/unreachable"})

    assert r.status_code in (200, 502), (
        f"expected a handled response, not a crash from cache initialization; got {r.status_code}: {r.text}"
    )
    assert isinstance(mod._cache.store, MemoryStore), "get_cache() should have fallen back to MemoryStore"
