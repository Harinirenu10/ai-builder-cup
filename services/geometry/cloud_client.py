"""Cloud Storage and Firestore client wrappers for Measured Fit Geometry Service.

Supports both production Google Cloud clients (Firestore, Cloud Storage)
and in-memory / mock implementations for offline development and testing.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fitcore.schemas import RoundLogDocument, TwinDocument

logger = logging.getLogger("geometry.cloud_client")


@runtime_checkable
class FirestoreClientProtocol(Protocol):
    """Protocol for Firestore job and twin document operations."""

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch job document by id from jobs/{job_id}."""
        ...

    def update_job(self, job_id: str, updates: dict[str, Any]) -> None:
        """Update job fields at jobs/{job_id}."""
        ...

    def create_round_log(self, job_id: str, round_num: int, log_data: dict[str, Any] | RoundLogDocument) -> None:
        """Store refinement round document at jobs/{job_id}/rounds/{round_num}."""
        ...

    def create_twin(self, twin_id: str, twin_data: dict[str, Any] | TwinDocument) -> None:
        """Store fitted twin document at twins/{twin_id}."""
        ...

    def get_twin(self, twin_id: str) -> dict[str, Any] | None:
        """Fetch twin document by id from twins/{twin_id}."""
        ...


@runtime_checkable
class StorageClientProtocol(Protocol):
    """Protocol for Cloud Storage photo and asset operations."""

    def download_bytes(self, uri_or_path: str) -> bytes:
        """Download raw bytes from gs:// URI, relative storage path, data URL, base64, or local path."""
        ...

    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to Cloud Storage bucket and return storage URI."""
        ...

    def delete_file(self, uri_or_path: str) -> bool:
        """Delete file from storage bucket."""
        ...


class InMemoryFirestoreClient:
    """In-memory Firestore client for offline testing and local simulation."""

    def __init__(self, initial_jobs: dict[str, dict[str, Any]] | None = None) -> None:
        self.jobs: dict[str, dict[str, Any]] = initial_jobs or {}
        self.rounds: dict[str, dict[int, dict[str, Any]]] = {}
        self.twins: dict[str, dict[str, Any]] = {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, updates: dict[str, Any]) -> None:
        if job_id not in self.jobs:
            self.jobs[job_id] = {"id": job_id}
        self.jobs[job_id].update(updates)

    def create_round_log(self, job_id: str, round_num: int, log_data: dict[str, Any] | RoundLogDocument) -> None:
        if job_id not in self.rounds:
            self.rounds[job_id] = {}
        data = log_data.model_dump() if isinstance(log_data, RoundLogDocument) else dict(log_data)
        self.rounds[job_id][round_num] = data

    def create_twin(self, twin_id: str, twin_data: dict[str, Any] | TwinDocument) -> None:
        data = twin_data.model_dump() if isinstance(twin_data, TwinDocument) else dict(twin_data)
        self.twins[twin_id] = data

    def get_twin(self, twin_id: str) -> dict[str, Any] | None:
        return self.twins.get(twin_id)


class InMemoryStorageClient:
    """In-memory Cloud Storage client for testing photo lifecycle and privacy cleanup."""

    def __init__(self, files: dict[str, bytes] | None = None, bucket_name: str = "test-bucket") -> None:
        self.bucket_name = bucket_name
        self.files: dict[str, bytes] = files or {}
        self.deleted_paths: list[str] = []

    def _normalize_key(self, uri_or_path: str) -> str:
        prefix = f"gs://{self.bucket_name}/"
        if uri_or_path.startswith(prefix):
            return uri_or_path[len(prefix):]
        if uri_or_path.startswith("gs://"):
            parts = uri_or_path[5:].split("/", 1)
            return parts[1] if len(parts) > 1 else parts[0]
        return uri_or_path.lstrip("/")

    def download_bytes(self, uri_or_path: str) -> bytes:
        # 1. Base64 data URL
        if uri_or_path.startswith("data:image/") and ";base64," in uri_or_path:
            _, b64_str = uri_or_path.split(";base64,", 1)
            return base64.b64decode(b64_str)

        # 2. Key in in-memory storage
        norm_key = self._normalize_key(uri_or_path)
        if norm_key in self.files:
            return self.files[norm_key]
        if uri_or_path in self.files:
            return self.files[uri_or_path]

        # 3. Local file path check
        local_p = Path(uri_or_path)
        if local_p.exists() and local_p.is_file():
            return local_p.read_bytes()

        raise FileNotFoundError(f"Storage object not found: {uri_or_path}")

    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        norm_key = self._normalize_key(destination_path)
        self.files[norm_key] = data
        return f"gs://{self.bucket_name}/{norm_key}"

    def delete_file(self, uri_or_path: str) -> bool:
        norm_key = self._normalize_key(uri_or_path)
        self.deleted_paths.append(uri_or_path)
        deleted = False
        if norm_key in self.files:
            del self.files[norm_key]
            deleted = True
        if uri_or_path in self.files:
            del self.files[uri_or_path]
            deleted = True

        # Also remove if it was a local file
        local_p = Path(uri_or_path)
        if local_p.exists() and local_p.is_file():
            try:
                local_p.unlink()
                deleted = True
            except Exception:
                pass
        return deleted


class GoogleCloudFirestoreClient:
    """Production Google Cloud Firestore client implementation."""

    def __init__(self, project_id: str | None = None) -> None:
        from google.cloud import firestore
        proj = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("FIREBASE_PROJECT_ID")
        self.db = firestore.Client(project=proj)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        doc = self.db.collection("jobs").document(job_id).get()
        return doc.to_dict() if doc.exists else None

    def update_job(self, job_id: str, updates: dict[str, Any]) -> None:
        self.db.collection("jobs").document(job_id).set(updates, merge=True)

    def create_round_log(self, job_id: str, round_num: int, log_data: dict[str, Any] | RoundLogDocument) -> None:
        data = log_data.model_dump() if isinstance(log_data, RoundLogDocument) else dict(log_data)
        self.db.collection("jobs").document(job_id).collection("rounds").document(str(round_num)).set(data)

    def create_twin(self, twin_id: str, twin_data: dict[str, Any] | TwinDocument) -> None:
        data = twin_data.model_dump() if isinstance(twin_data, TwinDocument) else dict(twin_data)
        self.db.collection("twins").document(twin_id).set(data)

    def get_twin(self, twin_id: str) -> dict[str, Any] | None:
        doc = self.db.collection("twins").document(twin_id).get()
        return doc.to_dict() if doc.exists else None


class GoogleCloudStorageClient:
    """Production Google Cloud Storage client implementation."""

    def __init__(self, bucket_name: str | None = None) -> None:
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket_name = bucket_name or os.environ.get("FIREBASE_STORAGE_BUCKET") or "measured-fit-default"
        self.bucket = self.client.bucket(self.bucket_name)

    def _parse_bucket_and_blob(self, uri_or_path: str) -> tuple[Any, str]:
        if uri_or_path.startswith("gs://"):
            without_scheme = uri_or_path[5:]
            parts = without_scheme.split("/", 1)
            b_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            return self.client.bucket(b_name), blob_name
        return self.bucket, uri_or_path.lstrip("/")

    def download_bytes(self, uri_or_path: str) -> bytes:
        # Base64 data URL
        if uri_or_path.startswith("data:image/") and ";base64," in uri_or_path:
            _, b64_str = uri_or_path.split(";base64,", 1)
            return base64.b64decode(b64_str)

        bucket, blob_name = self._parse_bucket_and_blob(uri_or_path)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()

    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        clean_path = destination_path.lstrip("/")
        blob = self.bucket.blob(clean_path)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{clean_path}"

    def delete_file(self, uri_or_path: str) -> bool:
        try:
            bucket, blob_name = self._parse_bucket_and_blob(uri_or_path)
            blob = bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as exc:
            logger.warning("Failed to delete storage file %s: %s", uri_or_path, exc)
            return False


# Global provider getters with support for injection in testing
_global_firestore_client: FirestoreClientProtocol | None = None
_global_storage_client: StorageClientProtocol | None = None


def set_cloud_clients(
    firestore_client: FirestoreClientProtocol | None = None,
    storage_client: StorageClientProtocol | None = None,
) -> None:
    """Set global test/mock cloud client instances."""
    global _global_firestore_client, _global_storage_client
    _global_firestore_client = firestore_client
    _global_storage_client = storage_client


def get_firestore_client() -> FirestoreClientProtocol:
    """Get active Firestore client, falling back to in-memory if GCP is not configured.

    The in-memory fallback is cached process-wide (same pattern as the live GCP client
    would naturally provide) so that a job created in one request is still visible to
    later requests (status polling, twin lookup) within the same running process.
    """
    global _global_firestore_client
    if _global_firestore_client is not None:
        return _global_firestore_client

    # Check for live GCP credentials or environment
    if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("FIREBASE_PROJECT_ID") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            _global_firestore_client = GoogleCloudFirestoreClient()
            return _global_firestore_client
        except Exception as exc:
            logger.warning("Could not initialize live Firestore client: %s. Using InMemoryFirestoreClient.", exc)

    _global_firestore_client = InMemoryFirestoreClient()
    return _global_firestore_client


def get_storage_client() -> StorageClientProtocol:
    """Get active Storage client, falling back to in-memory if GCP is not configured.

    The in-memory fallback is cached process-wide so uploaded photos/GLB assets remain
    reachable by later requests within the same running process (see get_firestore_client).
    """
    global _global_storage_client
    if _global_storage_client is not None:
        return _global_storage_client

    if os.environ.get("FIREBASE_STORAGE_BUCKET") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            _global_storage_client = GoogleCloudStorageClient()
            return _global_storage_client
        except Exception as exc:
            logger.warning("Could not initialize live Storage client: %s. Using InMemoryStorageClient.", exc)

    _global_storage_client = InMemoryStorageClient()
    return _global_storage_client
