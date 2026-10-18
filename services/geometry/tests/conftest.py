import os

# The geometry dev-fallback cloud clients are disk-backed by default (see
# services/geometry/local_persistence.py), so any test that reaches get_firestore_client()/
# get_storage_client() without injecting its own client would otherwise write real files into
# services/geometry/.data. Tests must stay hermetic and ephemeral: pin them to the pure in-RAM
# store, which is exactly the behavior they were written against. test_local_persistence.py
# exercises the disk-backed classes directly against tmp_path instead.
os.environ.setdefault("GEOMETRY_STORE", "memory")
