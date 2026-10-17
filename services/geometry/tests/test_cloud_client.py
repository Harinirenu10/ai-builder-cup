"""Tests for services/geometry/cloud_client.py fallback client caching.

Regression coverage for a bug where get_firestore_client()/get_storage_client()
constructed a brand-new, empty InMemoryFirestoreClient/InMemoryStorageClient on
every call when no live GCP credentials were configured. That meant a job or
twin written by one request was invisible to any later request (status polling,
twin lookup, GLB download) within the same running process.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    get_firestore_client,
    get_storage_client,
    set_cloud_clients,
)


@pytest.fixture(autouse=True)
def _reset_cloud_client_globals():
    """Ensure the fallback-cache globals start and end each test cleared."""
    set_cloud_clients(firestore_client=None, storage_client=None)
    yield
    set_cloud_clients(firestore_client=None, storage_client=None)


def test_get_firestore_client_fallback_is_cached_process_wide(monkeypatch):
    """Without live GCP env vars, repeated calls must return the SAME in-memory
    instance (not a fresh, empty one) so writes from one call are visible to
    later calls."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    fs1 = get_firestore_client()
    assert isinstance(fs1, InMemoryFirestoreClient)
    fs1.update_job("job_123", {"status": "done"})

    fs2 = get_firestore_client()
    assert fs2 is fs1
    assert fs2.get_job("job_123") == {"id": "job_123", "status": "done"}


def test_get_storage_client_fallback_is_cached_process_wide(monkeypatch):
    """Same guarantee as above, for the Cloud Storage fallback client."""
    monkeypatch.delenv("FIREBASE_STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    st1 = get_storage_client()
    assert isinstance(st1, InMemoryStorageClient)
    url = st1.upload_bytes("users/u1/twins/t1/twin.glb", b"glb-bytes")

    st2 = get_storage_client()
    assert st2 is st1
    assert st2.download_bytes(url) == b"glb-bytes"


def test_explicit_set_cloud_clients_still_takes_precedence(monkeypatch):
    """Explicit injection (used by tests and, in future, DI) must override the
    fallback cache rather than being shadowed by it."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    injected = InMemoryFirestoreClient()
    set_cloud_clients(firestore_client=injected)
    assert get_firestore_client() is injected
