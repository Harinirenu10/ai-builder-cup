"""Disk-backed dev persistence for the twin pipeline's Firestore/Storage clients.

Mirrors the exact JSON-file pattern already established by services/api/apilib/stores.py's
FileStore and services/reader/readerlib/cache.py's FileStore: local dev gets a real,
restart-surviving store with zero new infrastructure, while a configured GCP project still
gets the real GoogleCloudFirestoreClient/GoogleCloudStorageClient (see cloud_client.py's
get_firestore_client/get_storage_client, which pick between them).

Without this, InMemoryFirestoreClient/InMemoryStorageClient (cloud_client.py) are pure
in-process dicts: every twin, job, and uploaded GLB vanishes the moment the API/geometry
process restarts. FileBackedFirestoreClient/FileBackedStorageClient add disk persistence on
top of those same classes (reusing their in-memory dict logic, just loading on start and
saving on every write), so a twin created before a restart is still there after one.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from services.geometry.cloud_client import InMemoryFirestoreClient, InMemoryStorageClient

logger = logging.getLogger("geometry.local_persistence")

_FIT_PREFS_SEP = "\x1f"  # unit separator -- never appears in a real user_id/category


class FileBackedFirestoreClient(InMemoryFirestoreClient):
    """InMemoryFirestoreClient that persists every collection to one JSON file on disk.

    Every read and every write re-reads that file first, the same way
    services/api/apilib/stores.py::FileStore does. This is what keeps a long-running process
    (the uvicorn dev server) consistent with anything else touching the same store -- a CLI run,
    a seed script, a cleanup, a second worker. Without the re-read, this client would rewrite the
    whole file from the snapshot it loaded at construction and silently undo those writes: a twin
    deleted out-of-process came back on disk the next time the running server saved a capture,
    and then shadowed the real twins in get_latest_twin_for_user while pointing at a blob that no
    longer existed.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._stamp: tuple[int, int] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _current_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        # Re-parsing on every call would be wasteful: a consented front photo (the user's
        # opt_in_keep_front_photo) stays on its job document as inline base64, so this file runs
        # to megabytes and parsing it measured ~100ms per operation. Stat it instead and only
        # parse when it actually changed, which for a single dev server is almost never. The
        # tradeoff: an external write landing on the same byte size within the same mtime
        # nanosecond tick would be missed, which no realistic editor or process produces.
        stamp = self._current_stamp()
        if stamp is not None and stamp == self._stamp:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read Firestore dev store at %s (%s); starting empty.", self.path, exc)
            return
        self._stamp = stamp
        self.jobs = raw.get("jobs", {})
        self.rounds = {
            job_id: {int(round_num): data for round_num, data in rounds.items()}
            for job_id, rounds in raw.get("rounds", {}).items()
        }
        self.twins = raw.get("twins", {})
        self.garments = raw.get("garments", {})
        self.fit_prefs = {
            tuple(key.split(_FIT_PREFS_SEP, 1)): value
            for key, value in raw.get("fit_prefs", {}).items()
        }
        self.products = raw.get("products", {})
        self.orders = raw.get("orders", {})
        self.progress = raw.get("progress", {})

    def _save(self) -> None:
        payload = {
            "jobs": self.jobs,
            "rounds": {
                job_id: {str(round_num): data for round_num, data in rounds.items()}
                for job_id, rounds in self.rounds.items()
            },
            "twins": self.twins,
            "garments": self.garments,
            "fit_prefs": {
                f"{uid}{_FIT_PREFS_SEP}{category}": value
                for (uid, category), value in self.fit_prefs.items()
            },
            "products": self.products,
            "orders": self.orders,
            "progress": self.progress,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        # Our own write is already reflected in memory; record its stamp so the next operation
        # does not re-parse the file we just produced.
        self._stamp = self._current_stamp()

    # ---- writes: re-read first so the change merges onto current disk state, then persist

    def update_job(self, job_id: str, updates: dict[str, Any]) -> None:
        self._load()
        super().update_job(job_id, updates)
        self._save()

    def create_round_log(self, job_id: str, round_num: int, log_data: Any) -> None:
        self._load()
        super().create_round_log(job_id, round_num, log_data)
        self._save()

    def create_twin(self, twin_id: str, twin_data: Any) -> None:
        self._load()
        super().create_twin(twin_id, twin_data)
        self._save()

    def update_twin(self, twin_id: str, updates: dict[str, Any]) -> None:
        self._load()
        super().update_twin(twin_id, updates)
        self._save()

    def create_garment(self, garment_id: str, garment_data: dict[str, Any]) -> None:
        self._load()
        super().create_garment(garment_id, garment_data)
        self._save()

    def set_fit_prefs(self, user_id: str, category: str, prefs_data: dict[str, Any]) -> None:
        self._load()
        super().set_fit_prefs(user_id, category, prefs_data)
        self._save()

    def create_product(self, product_id: str, product_data: dict[str, Any]) -> None:
        self._load()
        super().create_product(product_id, product_data)
        self._save()

    def create_order(self, order_id: str, order_data: Any) -> None:
        self._load()
        super().create_order(order_id, order_data)
        self._save()

    def set_user_progress(self, user_id: str, progress_data: dict[str, Any]) -> None:
        self._load()
        super().set_user_progress(user_id, progress_data)
        self._save()

    # ---- reads: re-read first so another process's writes/deletions are visible immediately

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_job(job_id)

    def get_twin(self, twin_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_twin(twin_id)

    def get_latest_twin_for_user(self, user_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_latest_twin_for_user(user_id)

    def get_garment(self, garment_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_garment(garment_id)

    def get_fit_prefs(self, user_id: str, category: str) -> dict[str, Any] | None:
        self._load()
        return super().get_fit_prefs(user_id, category)

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_product(product_id)

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_order(order_id)

    def get_user_progress(self, user_id: str) -> dict[str, Any] | None:
        self._load()
        return super().get_user_progress(user_id)


class FileBackedStorageClient(InMemoryStorageClient):
    """InMemoryStorageClient that writes/reads/deletes blobs as real files on local disk.

    Unlike InMemoryFirestoreClient's single JSON file, storage objects (raw photos, GLB
    meshes) are written as individual files under `base_dir`, one per storage key, so large
    GLB bytes are never loaded into a long-lived in-process dict at all -- only touched on
    actual upload/download/delete.
    """

    def __init__(self, base_dir: Path, bucket_name: str = "local-dev-bucket") -> None:
        super().__init__(bucket_name=bucket_name)
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, norm_key: str) -> Path:
        # norm_key is always produced by _normalize_key from server-generated storage paths
        # (e.g. "users/{uid}/twins/{twin_id}/twin.glb"), never raw user input, but resolve and
        # re-check containment anyway as defense in depth against an unexpected ".." segment.
        candidate = (self.base_dir / norm_key).resolve()
        if self.base_dir.resolve() not in candidate.parents and candidate != self.base_dir.resolve():
            raise ValueError(f"Refusing to write storage object outside base_dir: {norm_key}")
        return candidate

    def download_bytes(self, uri_or_path: str) -> bytes:
        if uri_or_path.startswith("data:image/") and ";base64," in uri_or_path:
            return super().download_bytes(uri_or_path)

        norm_key = self._normalize_key(uri_or_path)
        blob_path = self._blob_path(norm_key)
        if blob_path.is_file():
            return blob_path.read_bytes()

        return super().download_bytes(uri_or_path)

    def upload_bytes(self, destination_path: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        norm_key = self._normalize_key(destination_path)
        blob_path = self._blob_path(norm_key)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(data)
        return f"gs://{self.bucket_name}/{norm_key}"

    def delete_file(self, uri_or_path: str) -> bool:
        norm_key = self._normalize_key(uri_or_path)
        deleted = False
        try:
            blob_path = self._blob_path(norm_key)
            if blob_path.is_file():
                blob_path.unlink()
                deleted = True
        except ValueError:
            pass
        if super().delete_file(uri_or_path):
            deleted = True
        return deleted


def default_dev_data_dir() -> Path:
    return Path(os.environ.get("GEOMETRY_DATA_DIR", str(Path(__file__).resolve().parent / ".data")))


def build_dev_firestore_client() -> InMemoryFirestoreClient:
    """The geometry/API dev-fallback Firestore client: file-backed unless GEOMETRY_STORE=memory."""
    if os.environ.get("GEOMETRY_STORE", "auto").lower() == "memory":
        return InMemoryFirestoreClient()
    return FileBackedFirestoreClient(default_dev_data_dir() / "firestore.json")


def build_dev_storage_client() -> InMemoryStorageClient:
    """The geometry/API dev-fallback Storage client: file-backed unless GEOMETRY_STORE=memory."""
    if os.environ.get("GEOMETRY_STORE", "auto").lower() == "memory":
        return InMemoryStorageClient()
    return FileBackedStorageClient(default_dev_data_dir() / "blobs")
