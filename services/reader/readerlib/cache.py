"""Cache of read results by sha256(url) in products/{hash}, fresh for 7 days.

Firestore when READER_CACHE=firestore (or auto with GOOGLE_CLOUD_PROJECT set and the client installed),
otherwise JSON files under READER_CACHE_DIR (default services/reader/.cache). No raw page text or
screenshots are stored.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

TTL = timedelta(days=7)
COLLECTION = "products"


def url_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


class Store(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, doc: dict[str, Any]) -> None: ...


class FileStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        f = self.root / f"{key}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    def put(self, key: str, doc: dict[str, Any]) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


class MemoryStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.docs.get(key)

    def put(self, key: str, doc: dict[str, Any]) -> None:
        self.docs[key] = doc


class FirestoreStore:
    def __init__(self) -> None:
        from google.cloud import firestore  # lazy: only needed in the cloud

        self._col = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT")).collection(COLLECTION)

    def get(self, key: str) -> dict[str, Any] | None:
        snap = self._col.document(key).get()
        return snap.to_dict() if snap.exists else None

    def put(self, key: str, doc: dict[str, Any]) -> None:
        self._col.document(key).set(doc)


def default_store() -> Store:
    mode = os.environ.get("READER_CACHE", "auto").lower()
    if mode == "memory":
        return MemoryStore()
    if mode == "firestore" or (mode == "auto" and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        try:
            return FirestoreStore()
        except ImportError:
            if mode == "firestore":
                raise
    return FileStore(Path(os.environ.get("READER_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache")))


class ReadCache:
    def __init__(self, store: Store | None = None, ttl: timedelta = TTL):
        self.store = store or default_store()
        self.ttl = ttl

    def get(self, url: str, now: datetime | None = None) -> dict[str, Any] | None:
        doc = self.store.get(url_key(url))
        if not doc:
            return None
        fetched = datetime.fromisoformat(doc["fetched_at"])
        if (now or datetime.now(timezone.utc)) - fetched > self.ttl:
            return None
        return doc

    def put(self, url: str, doc: dict[str, Any], now: datetime | None = None) -> None:
        self.store.put(url_key(url), {**doc, "url": url, "fetched_at": (now or datetime.now(timezone.utc)).isoformat()})
