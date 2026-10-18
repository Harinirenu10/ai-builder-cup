"""Document stores for merchants, consents, twins, fit_queries and gifts.

Firestore in the cloud (API_STORE=firestore, or auto when GOOGLE_CLOUD_PROJECT is set), JSON files for local
development (services/api/.data), memory for tests. Same four operations everywhere.
"""

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("apilib.stores")


class Store(Protocol):
    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None: ...
    def set(self, coll: str, doc_id: str, doc: dict[str, Any]) -> None: ...
    def update(self, coll: str, doc_id: str, fields: dict[str, Any]) -> None: ...
    def find(self, coll: str, **equals: Any) -> list[dict[str, Any]]: ...
    def add(self, coll: str, doc: dict[str, Any]) -> str: ...


def new_id() -> str:
    return uuid.uuid4().hex


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None:
        doc = self.data.get(coll, {}).get(doc_id)
        return json.loads(json.dumps(doc)) if doc is not None else None

    def set(self, coll: str, doc_id: str, doc: dict[str, Any]) -> None:
        with self._lock:
            self.data.setdefault(coll, {})[doc_id] = json.loads(json.dumps({**doc, "id": doc_id}))

    def update(self, coll: str, doc_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            cur = self.data.setdefault(coll, {}).get(doc_id)
            if cur is None:
                raise KeyError(f"{coll}/{doc_id}")
            cur.update(json.loads(json.dumps(fields)))

    def find(self, coll: str, **equals: Any) -> list[dict[str, Any]]:
        rows = [d for d in self.data.get(coll, {}).values() if all(d.get(k) == v for k, v in equals.items())]
        return json.loads(json.dumps(rows))

    def add(self, coll: str, doc: dict[str, Any]) -> str:
        doc_id = new_id()
        self.set(coll, doc_id, doc)
        return doc_id


class FileStore(MemoryStore):
    """MemoryStore that persists every change to one JSON file, so a seed script and the server share data."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data), encoding="utf-8")

    def get(self, coll, doc_id):
        self._load()
        return super().get(coll, doc_id)

    def set(self, coll, doc_id, doc):
        self._load()
        super().set(coll, doc_id, doc)
        self._save()

    def update(self, coll, doc_id, fields):
        self._load()
        super().update(coll, doc_id, fields)
        self._save()

    def find(self, coll, **equals):
        self._load()
        return super().find(coll, **equals)


class FirestoreStore:
    def __init__(self) -> None:
        from google.cloud import firestore  # lazy: only needed in the cloud

        self._db = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))

    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None:
        snap = self._db.collection(coll).document(doc_id).get()
        return {**snap.to_dict(), "id": doc_id} if snap.exists else None

    def set(self, coll: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._db.collection(coll).document(doc_id).set(doc)

    def update(self, coll: str, doc_id: str, fields: dict[str, Any]) -> None:
        self._db.collection(coll).document(doc_id).update(fields)

    def find(self, coll: str, **equals: Any) -> list[dict[str, Any]]:
        q = self._db.collection(coll)
        for k, v in equals.items():
            q = q.where(k, "==", v)
        return [{**s.to_dict(), "id": s.id} for s in q.stream()]

    def add(self, coll: str, doc: dict[str, Any]) -> str:
        ref = self._db.collection(coll).document()
        ref.set(doc)
        return ref.id


_store: Store | None = None


def default_store() -> Store:
    mode = os.environ.get("API_STORE", "auto").lower()
    if mode == "memory":
        return MemoryStore()
    if mode == "firestore" or (mode == "auto" and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        try:
            return FirestoreStore()
        except Exception as exc:
            # API_STORE=firestore is an explicit request for the live store: fail loudly rather
            # than silently degrading (same strict/auto split as services/geometry/cloud_client.py
            # and services/reader/readerlib/cache.py's own default_store functions). "auto" with
            # GOOGLE_CLOUD_PROJECT set but no usable credentials (missing ADC, unreachable
            # Firestore, etc.) falls back to FileStore instead of crashing every request.
            if mode == "firestore":
                raise
            logger.warning(
                "Could not initialize live Firestore store (%s); using FileStore for this process.", exc
            )
    return FileStore(Path(os.environ.get("API_DATA_DIR", Path(__file__).resolve().parent.parent / ".data")) / "store.json")


def get_store() -> Store:
    """FastAPI dependency; tests replace it with app.dependency_overrides."""
    global _store
    if _store is None:
        _store = default_store()
    return _store
