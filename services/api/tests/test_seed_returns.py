import importlib.util
import sys

import pytest

from apilib.stores import MemoryStore
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "seed"))


@pytest.fixture(scope="module")
def seeded():
    spec = importlib.util.spec_from_file_location("returns_demo", ROOT / "seed" / "returns_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    store = MemoryStore()
    out = mod.seed_returns(store, ["http://localhost:8081"])
    return store, out


def test_four_returns_one_per_routing_branch(seeded):
    store, out = seeded
    by = {r["return"]: r for r in out["returns"]}
    assert by["demo-return-1-label"]["route"] == "auto_exchange" and by["demo-return-1-label"]["status"] == "approved_exchange"
    assert by["demo-return-2-model-miss"]["route"] == "auto_exchange_next_size" and by["demo-return-2-model-miss"]["status"] == "approved_exchange"
    assert by["demo-return-3-wrong-size"]["route"] == "human_review" and by["demo-return-3-wrong-size"]["status"] == "pending_review"
    assert by["demo-return-4-policy"]["route"] == "merchant_policy" and by["demo-return-4-policy"]["status"] == "approved_refund"
    assert by["demo-return-1-label"]["guarantee"] and by["demo-return-2-model-miss"]["guarantee"] and not by["demo-return-4-policy"]["guarantee"]


def test_flags_and_logs_match_the_branches(seeded):
    store, _ = seeded
    r1, r2 = store.get("returns", "demo-return-1-label"), store.get("returns", "demo-return-2-model-miss")
    assert r1["routing"]["seller_label_error"] and not r1["routing"]["model_miss"]
    assert r2["routing"]["model_miss"] and not r2["routing"]["seller_label_error"]
    assert all(store.get("returns", i)["decision_log"] for i in ("demo-return-1-label", "demo-return-2-model-miss", "demo-return-3-wrong-size", "demo-return-4-policy"))


def test_seed_is_idempotent_and_simulated(seeded):
    store, _ = seeded
    spec = importlib.util.spec_from_file_location("returns_demo2", ROOT / "seed" / "returns_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.seed_returns(store, ["http://localhost:8081"])
    assert len(store.find("returns")) == 4 and len(store.find("orders")) == 6
    assert all(u["source"] == "simulated" for u in store.find("unboxing_measurements"))
    assert store.get("merchants", "m_demo_threads")["staff_uids"] == ["demo-staff"]
