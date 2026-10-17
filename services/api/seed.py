"""Seed the local API store (services/api/.data) for the mock storefront demo. No cloud needed.

    python services/api/seed.py                      # merchant "Demo Threads Co." + a simulated twin for user "demo-shopper"
    python services/api/seed.py --origin http://localhost:8081 --origin http://127.0.0.1:8081

Then run the API with AUTH_MODE=dev (see services/api/README.md). The twin is simulated data from samples/twin.json.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from apilib.stores import default_store  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_ORIGINS = ["http://localhost:8081", "http://127.0.0.1:8081", "http://localhost:5000"]


def seed(origins: list[str], uid: str = "demo-shopper") -> None:
    store = default_store()
    store.set("merchants", "m_demo_threads", {"name": "Demo Threads Co.", "publishable_key": "pk_demo_threads", "allowed_origins": origins})
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    store.set("twins", uid, {"owner_id": uid, "simulated": True, "measurements": {"zones": twin["zones"]}})
    print(f"Seeded merchant pk_demo_threads (origins: {', '.join(origins)}) and a simulated twin for user '{uid}'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", action="append", help="allowed store origin, repeatable")
    ap.add_argument("--uid", default="demo-shopper")
    a = ap.parse_args()
    seed(a.origin or DEFAULT_ORIGINS, a.uid)
