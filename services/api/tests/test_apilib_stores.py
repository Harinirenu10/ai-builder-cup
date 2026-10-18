"""Regression tests for apilib/stores.py::default_store()'s Firestore/FileStore selection.

Covers the fix: in "auto" mode, a Firestore initialization failure (e.g. GOOGLE_CLOUD_PROJECT is
set but no ADC credentials exist, the normal state of a dev machine that hasn't run
`gcloud auth application-default login`) must log a warning and fall back to FileStore instead of
crashing every request with an uncaught 500 -- matching the established pattern in
services/geometry/cloud_client.py::get_firestore_client(). API_STORE=firestore (explicitly
requesting the live store) must still fail loudly, not silently degrade.
"""

import logging

import pytest

import apilib.stores as stores_module
from apilib.stores import FileStore, MemoryStore, default_store


class _FakeFirestoreStoreOK:
    """Stands in for a successful google.cloud.firestore.Client() connection."""

    def __init__(self) -> None:
        self.initialized = True


class _FakeFirestoreStoreMissingCreds:
    """Stands in for google.auth.exceptions.DefaultCredentialsError -- NOT an ImportError."""

    def __init__(self) -> None:
        raise RuntimeError("Your default credentials were not found. Run gcloud auth application-default login.")


@pytest.fixture(autouse=True)
def _isolate_store_env(monkeypatch):
    """default_store() reads these directly; keep each test independent of the real environment."""
    for var in ("API_STORE", "GOOGLE_CLOUD_PROJECT", "API_DATA_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_auto_mode_uses_firestore_when_it_initializes_successfully(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setattr(stores_module, "FirestoreStore", _FakeFirestoreStoreOK)

    store = default_store()

    assert isinstance(store, _FakeFirestoreStoreOK)


def test_auto_mode_falls_back_to_filestore_when_credentials_are_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("API_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(stores_module, "FirestoreStore", _FakeFirestoreStoreMissingCreds)

    with caplog.at_level(logging.WARNING, logger="apilib.stores"):
        store = default_store()

    assert isinstance(store, FileStore)
    assert store.path == tmp_path / "store.json"
    assert any("FileStore" in r.message and "firestore" in r.message.lower() for r in caplog.records)
    # the fallback store must actually be usable, not a half-broken object
    store.set("merchants", "m1", {"name": "Demo"})
    assert store.get("merchants", "m1")["name"] == "Demo"


def test_explicit_firestore_mode_raises_instead_of_falling_back(monkeypatch, caplog):
    monkeypatch.setenv("API_STORE", "firestore")
    monkeypatch.setattr(stores_module, "FirestoreStore", _FakeFirestoreStoreMissingCreds)

    with caplog.at_level(logging.WARNING, logger="apilib.stores"):
        with pytest.raises(RuntimeError, match="credentials"):
            default_store()

    # strict mode: no silent degrade, no misleading "falling back" warning either
    assert not any("FileStore" in r.message for r in caplog.records)


def test_explicit_firestore_mode_succeeds_when_credentials_are_available(monkeypatch):
    monkeypatch.setenv("API_STORE", "firestore")
    monkeypatch.setattr(stores_module, "FirestoreStore", _FakeFirestoreStoreOK)

    store = default_store()

    assert isinstance(store, _FakeFirestoreStoreOK)


def test_memory_mode_bypasses_firestore_entirely(monkeypatch):
    monkeypatch.setenv("API_STORE", "memory")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setattr(stores_module, "FirestoreStore", _FakeFirestoreStoreMissingCreds)

    store = default_store()

    assert isinstance(store, MemoryStore)


def test_auto_mode_without_google_cloud_project_uses_filestore_directly(monkeypatch, tmp_path):
    monkeypatch.setenv("API_DATA_DIR", str(tmp_path))

    store = default_store()

    assert isinstance(store, FileStore)
    assert store.path == tmp_path / "store.json"
