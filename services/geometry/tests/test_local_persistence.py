"""Regression tests for restart-surviving twin persistence (services/geometry/local_persistence.py).

Before this, the dev fallback clients (InMemoryFirestoreClient/InMemoryStorageClient) were pure
in-process dicts: a twin and its GLB existed only inside the running API process, so restarting
the backend silently lost every saved twin and the 3D Twin Viewer fell back to "No 3D Twin Yet"
even though the capture had succeeded. These tests cover the two flows that regressed:

- "page refresh": repeated reads against the same live client return the same twin.
- "backend restart": brand-new client instances over the same data dir (what a fresh process
  gets) still return the twin document, its measurements, and its GLB bytes.

They also lock down that disk persistence did NOT weaken the zero-retention photo guarantee:
deleting a photo must physically remove it from disk, and stripping the job record's inline
photos must not leave them behind in the on-disk JSON.
"""

from __future__ import annotations

import json

import pytest

from fitcore.schemas import Measurement, Measurements, MeasurementZone, TwinDocument
from services.geometry.local_persistence import (
    FileBackedFirestoreClient,
    FileBackedStorageClient,
    build_dev_firestore_client,
    build_dev_storage_client,
)

GLB_BYTES = b"glTF\x02\x00\x00\x00" + b"\x00" * 64


def _twin_doc(twin_id: str = "twin_abc123", user_id: str = "user_1") -> TwinDocument:
    return TwinDocument(
        id=twin_id,
        user_id=user_id,
        gender="neutral",
        height_cm=172.0,
        betas=[0.1] * 10,
        measurements=Measurements(
            zones={
                MeasurementZone.chest: Measurement(value_cm=98.5, confidence=0.9, source="smpl_fit"),
                MeasurementZone.waist: Measurement(value_cm=81.2, confidence=0.9, source="smpl_fit"),
            }
        ),
        glb_storage_url="gs://local-dev-bucket/users/user_1/twins/twin_abc123/twin.glb",
        final_mean_iou=0.42,
        created_at="2026-10-18T12:00:00+00:00",
    )


def test_twin_and_glb_survive_simulated_backend_restart(tmp_path):
    """A twin saved before a restart must still be readable by a brand-new process's clients."""
    fs = FileBackedFirestoreClient(tmp_path / "firestore.json")
    st = FileBackedStorageClient(tmp_path / "blobs")

    glb_url = st.upload_bytes(
        destination_path="users/user_1/twins/twin_abc123/twin.glb",
        data=GLB_BYTES,
        content_type="model/gltf-binary",
    )
    fs.create_twin("twin_abc123", _twin_doc())

    # Simulate a backend restart: the old process's in-memory state is gone entirely; a fresh
    # process constructs new client objects over the same data dir.
    fs_after = FileBackedFirestoreClient(tmp_path / "firestore.json")
    st_after = FileBackedStorageClient(tmp_path / "blobs")

    restored = fs_after.get_twin("twin_abc123")
    assert restored is not None, "twin document did not survive the restart"
    assert restored["user_id"] == "user_1"
    assert restored["glb_storage_url"] == glb_url
    assert restored["measurements"]["zones"]["chest"]["value_cm"] == 98.5
    assert restored["final_mean_iou"] == 0.42

    assert st_after.download_bytes(glb_url) == GLB_BYTES, "GLB bytes did not survive the restart"


def test_get_latest_twin_for_user_survives_restart_and_isolates_owners(tmp_path):
    """The /v1/me/twin lookup path must work after a restart, and only ever return your own twin."""
    fs = FileBackedFirestoreClient(tmp_path / "firestore.json")
    fs.create_twin("twin_mine_old", _twin_doc("twin_mine_old", "user_1"))
    newer = _twin_doc("twin_mine_new", "user_1")
    newer.created_at = "2026-10-18T13:00:00+00:00"
    fs.create_twin("twin_mine_new", newer)
    fs.create_twin("twin_theirs", _twin_doc("twin_theirs", "user_2"))

    fs_after = FileBackedFirestoreClient(tmp_path / "firestore.json")

    mine = fs_after.get_latest_twin_for_user("user_1")
    assert mine is not None and mine["id"] == "twin_mine_new"
    theirs = fs_after.get_latest_twin_for_user("user_2")
    assert theirs is not None and theirs["id"] == "twin_theirs"
    assert fs_after.get_latest_twin_for_user("user_nobody") is None


def test_page_refresh_repeated_reads_are_stable(tmp_path):
    """A page refresh re-issues GET /v1/me/twin against the same live process: same twin back."""
    fs = FileBackedFirestoreClient(tmp_path / "firestore.json")
    fs.create_twin("twin_abc123", _twin_doc())

    first = fs.get_latest_twin_for_user("user_1")
    second = fs.get_latest_twin_for_user("user_1")
    third = fs.get_twin("twin_abc123")

    assert first is not None and second is not None and third is not None
    assert first["id"] == second["id"] == third["id"] == "twin_abc123"


def test_saved_flag_persists_across_restart(tmp_path):
    """The explicit Save action (POST /v1/me/twin/{id}/save -> update_twin) must outlive a restart."""
    fs = FileBackedFirestoreClient(tmp_path / "firestore.json")
    fs.create_twin("twin_abc123", _twin_doc())
    assert fs.get_twin("twin_abc123")["saved"] is False

    fs.update_twin("twin_abc123", {"saved": True, "saved_at": "2026-10-18T14:00:00+00:00"})

    fs_after = FileBackedFirestoreClient(tmp_path / "firestore.json")
    restored = fs_after.get_twin("twin_abc123")
    assert restored["saved"] is True
    assert restored["saved_at"] == "2026-10-18T14:00:00+00:00"
    # Unrelated fields must not be clobbered by the merge-style update.
    assert restored["measurements"]["zones"]["chest"]["value_cm"] == 98.5


def test_job_documents_survive_restart(tmp_path):
    """Job status must also persist, so polling a job across a restart doesn't 404."""
    fs = FileBackedFirestoreClient(tmp_path / "firestore.json")
    fs.update_job("job_1", {"id": "job_1", "user_id": "user_1", "status": "done", "twin_id": "twin_abc123"})
    fs.create_round_log("job_1", 1, {"round": 1, "accepted": True, "mean_iou_after": 0.42})

    fs_after = FileBackedFirestoreClient(tmp_path / "firestore.json")
    job = fs_after.get_job("job_1")
    assert job is not None and job["status"] == "done" and job["twin_id"] == "twin_abc123"
    # round_num keys round-trip through JSON (string) back to int, as the in-memory client uses.
    assert fs_after.rounds["job_1"][1]["accepted"] is True


def test_photo_delete_physically_removes_the_file_from_disk(tmp_path):
    """Zero-retention: privacy cleanup must unlink the real file, not just drop a dict entry."""
    st = FileBackedStorageClient(tmp_path / "blobs")
    photo_url = st.upload_bytes("users/user_1/jobs/job_1/front.jpg", b"rawphotobytes", "image/jpeg")
    on_disk = tmp_path / "blobs" / "users/user_1/jobs/job_1/front.jpg"
    assert on_disk.is_file()

    assert st.delete_file(photo_url) is True
    assert not on_disk.exists(), "raw photo still on disk after privacy cleanup"

    # And it must stay gone after a restart, not reappear from a stale index.
    st_after = FileBackedStorageClient(tmp_path / "blobs")
    with pytest.raises(FileNotFoundError):
        st_after.download_bytes(photo_url)


def test_stripping_job_photos_does_not_leave_them_in_the_on_disk_json(tmp_path):
    """_cleanup_job_record_photos' update_job({"photos": {}}) must rewrite the file without them."""
    path = tmp_path / "firestore.json"
    fs = FileBackedFirestoreClient(path)
    fs.update_job("job_1", {"id": "job_1", "user_id": "user_1", "photos": {"front": "data:image/jpeg;base64,SECRETBYTES"}})
    assert "SECRETBYTES" in path.read_text(encoding="utf-8")

    fs.update_job("job_1", {"photos": {}})

    assert "SECRETBYTES" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"]["job_1"]["photos"] == {}
    fs_after = FileBackedFirestoreClient(path)
    assert fs_after.get_job("job_1")["photos"] == {}


def test_storage_client_refuses_path_traversal(tmp_path):
    """A storage key must never escape the data dir."""
    st = FileBackedStorageClient(tmp_path / "blobs")
    with pytest.raises(ValueError):
        st.upload_bytes("../../escaped.glb", b"nope")


def test_build_dev_clients_respect_geometry_store_memory(monkeypatch, tmp_path):
    """GEOMETRY_STORE=memory keeps the old pure-RAM behavior for tests that want it."""
    monkeypatch.setenv("GEOMETRY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOMETRY_STORE", "memory")
    assert not isinstance(build_dev_firestore_client(), FileBackedFirestoreClient)
    assert not isinstance(build_dev_storage_client(), FileBackedStorageClient)

    monkeypatch.setenv("GEOMETRY_STORE", "auto")
    assert isinstance(build_dev_firestore_client(), FileBackedFirestoreClient)
    assert isinstance(build_dev_storage_client(), FileBackedStorageClient)


def test_a_long_lived_client_does_not_clobber_another_process_write(tmp_path):
    """A write by a long-running server must merge onto current disk state, not overwrite it.

    The dev store is one JSON file rewritten in full on every write. A long-lived client (the
    uvicorn process) that only read the file once at construction would rewrite it from its own
    stale snapshot, silently undoing anything another process wrote or deleted in the meantime --
    a CLI run (python -m geometry.twin), a seed script, a cleanup, a second worker. Confirmed
    live: a twin deleted out-of-process reappeared on disk after the running server saved a new
    capture, and because its created_at was later than the real twins', it then shadowed them in
    get_latest_twin_for_user and served a glb_storage_url whose blob no longer existed.
    """
    path = tmp_path / "firestore.json"
    server = FileBackedFirestoreClient(path)
    server.create_twin("twin_real", _twin_doc("twin_real", "user_1"))

    # Another process writes a new twin and deletes an old one, through its own client.
    other = FileBackedFirestoreClient(path)
    other.create_twin("twin_from_cli", _twin_doc("twin_from_cli", "user_1"))
    other.twins.pop("twin_real", None)
    other._save()

    # The long-lived server now writes something unrelated.
    server.update_job("job_1", {"id": "job_1", "user_id": "user_1", "status": "done"})

    on_disk = json.loads(path.read_text(encoding="utf-8"))["twins"]
    assert "twin_from_cli" in on_disk, "the other process's twin was clobbered by the server's write"
    assert "twin_real" not in on_disk, "the other process's deletion was undone by the server's write"


def test_reads_see_another_process_write(tmp_path):
    """A long-lived client must not keep serving a twin another process already deleted."""
    path = tmp_path / "firestore.json"
    server = FileBackedFirestoreClient(path)
    server.create_twin("twin_stale", _twin_doc("twin_stale", "user_1"))
    assert server.get_twin("twin_stale") is not None

    cleanup = FileBackedFirestoreClient(path)
    cleanup.twins.pop("twin_stale", None)
    cleanup._save()

    assert server.get_twin("twin_stale") is None, "server still serves a twin deleted on disk"
    assert server.get_latest_twin_for_user("user_1") is None
