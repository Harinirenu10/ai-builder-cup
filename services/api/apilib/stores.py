"""Document stores for merchants, consents, twins, fit_queries and gifts.

Firestore in the cloud (API_STORE=firestore, or auto when GOOGLE_CLOUD_PROJECT is set), JSON files for local
development (services/api/.data), memory for tests. Same four operations everywhere.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

TTL_FIELD = "expire_at"  # ISO string in every store; FirestoreStore writes it as a Timestamp so a Firestore TTL policy can delete the document


class Store(Protocol):
    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None: ...
    def set(self, coll: str, doc_id: str, doc: dict[str, Any]) -> None: ...
    def update(self, coll: str, doc_id: str, fields: dict[str, Any]) -> None: ...
    def find(self, coll: str, **equals: Any) -> list[dict[str, Any]]: ...
    def add(self, coll: str, doc: dict[str, Any]) -> str: ...
    def set_many(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None: ...


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

    def set_many(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        """Write (collection, id, document) rows together."""
        with self._lock:
            for coll, doc_id, doc in rows:
                self.data.setdefault(coll, {})[doc_id] = json.loads(json.dumps({**doc, "id": doc_id}))


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

    def set_many(self, rows):
        self._load()
        super().set_many(rows)
        self._save()


def _to_firestore(doc: dict[str, Any]) -> dict[str, Any]:
    if isinstance(doc.get(TTL_FIELD), str):
        doc = {**doc, TTL_FIELD: datetime.fromisoformat(doc[TTL_FIELD]).astimezone(timezone.utc)}
    return doc


def _from_firestore(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in doc.items()}


class FirestoreStore:
    def __init__(self) -> None:
        from google.cloud import firestore  # lazy: only needed in the cloud

        self._db = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))

    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None:
        snap = self._db.collection(coll).document(doc_id).get()
        return {**_from_firestore(snap.to_dict()), "id": doc_id} if snap.exists else None

    def set(self, coll: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._db.collection(coll).document(doc_id).set(_to_firestore(doc))

    def update(self, coll: str, doc_id: str, fields: dict[str, Any]) -> None:
        self._db.collection(coll).document(doc_id).update(fields)

    def find(self, coll: str, **equals: Any) -> list[dict[str, Any]]:
        q = self._db.collection(coll)
        for k, v in equals.items():
            q = q.where(k, "==", v)
        return [{**_from_firestore(s.to_dict()), "id": s.id} for s in q.stream()]

    def add(self, coll: str, doc: dict[str, Any]) -> str:
        ref = self._db.collection(coll).document()
        ref.set(_to_firestore(doc))
        return ref.id

    def set_many(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        """Batched writes, 400 per commit (the limit is 500)."""
        for i in range(0, len(rows), 400):
            batch = self._db.batch()
            for coll, doc_id, doc in rows[i:i + 400]:
                batch.set(self._db.collection(coll).document(doc_id), _to_firestore(doc))
            batch.commit()


_store: Store | None = None


def default_store() -> Store:
    mode = os.environ.get("API_STORE", "auto").lower()
    if mode == "memory":
        return MemoryStore()
    if mode == "firestore" or (mode == "auto" and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        try:
            return FirestoreStore()
        except ImportError:
            if mode == "firestore":
                raise
    return FileStore(Path(os.environ.get("API_DATA_DIR", Path(__file__).resolve().parent.parent / ".data")) / "store.json")


def get_store() -> Store:
    """FastAPI dependency; tests replace it with app.dependency_overrides."""
    global _store
    if _store is None:
        _store = default_store()
    return _store
