"""Demo sandbox: seeding, POST /v1/demo/session, and every judge-path flow against a seeded in-memory store."""

import re
import sys
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import ROOT, user

sys.path.insert(0, str(ROOT / "seed"))
import seed as seed_mod  # noqa: E402

ORIGIN = "https://shop.example"
BANNED = re.compile(r"\b(weight|fat|slimming|flatter\w*|flaw\w*|overweight|heavy|skinny)\b", re.I)


@pytest.fixture
def seeded(store, tmp_path):
    seed_mod.seed_all(store, [ORIGIN], "https://shop.example/store", tmp_path / "glb")
    return store


@pytest.fixture
def session(client, seeded):
    r = client.post("/v1/demo/session")
    assert r.status_code == 200, r.text
    return r.json()


def test_seed_counts(seeded, tmp_path):
    assert len(seeded.find("wardrobe", uid="demo-persona")) == 20
    ev = seeded.find("events", uid="demo-persona")
    assert len(ev) == 5 and all(e["circle"] for e in ev)
    assert {"Client meeting", "Wedding reception", "Sangeet"} <= {e["title"] for e in ev}
    assert len(seeded.find("orders", uid="demo-persona")) == 3
    rets = [r for o in seeded.find("orders", uid="demo-persona") for r in seeded.find("returns", order_id=o["id"])]
    assert len(rets) == 4 and {r["routing"]["route"] for r in rets} == {"auto_exchange", "auto_exchange_next_size", "human_review", "merchant_policy"}
    assert len(seeded.find("gifts", demo_template=True)) == 1
    assert len(seeded.find("fits", uid="demo-persona")) == 6
    assert {p.name for p in (tmp_path / "glb").iterdir()} == {"twin.glb", *(f"heat-{s}.glb" for s in ("kurta", "shirt", "trousers", "tshirt", "jacket", "blouse"))}


def test_glb_files_are_valid(seeded, tmp_path):
    import glb

    for f in (tmp_path / "glb").iterdir():
        doc = glb.read_glb_header(f.read_bytes())
        assert doc["accessors"][0]["count"] > 1000


def test_session_is_isolated_and_expiring(client, seeded, session):
    uid = session["uid"]
    assert uid.startswith("demo-") and session["custom_token"] == f"dev:{uid}" and session["auth_mode"] == "dev"
    assert seeded.get("twins", uid)["owner_id"] == uid
    assert len(seeded.find("wardrobe", uid=uid)) == 20 and len(seeded.find("orders", uid=uid)) == 3
    exp = datetime.fromisoformat(seeded.get("twins", uid)["expire_at"])
    assert timedelta(hours=23) < exp - datetime.now(timezone.utc) < timedelta(hours=25)
    second = client.post("/v1/demo/session").json()
    assert second["uid"] != uid and not [d for d in seeded.find("orders", uid=second["uid"]) if d["id"].startswith(uid)]
    m = seeded.get("merchants", session["merchant_id"])
    assert m["staff_uids"] == [uid] and "publishable_key" not in m
    assert seeded.get("merchants", "m_demo_threads")["publishable_key"] == "pk_demo_threads"
    # the persona template is untouched
    assert len(seeded.find("orders", uid="demo-persona")) == 3


def test_session_rate_limited(client, seeded):
    codes = [client.post("/v1/demo/session").status_code for _ in range(8)]
    assert codes[:6] == [200] * 6 and 429 in codes[6:]


def test_session_without_seed_fails_clearly(client):
    r = client.post("/v1/demo/session")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "demo_not_seeded"


def test_twin_view_is_owner_only(client, session):
    h = user(session["uid"])
    t = client.get("/v1/me/twin", headers=h).json()
    assert t["simulated"] is True and t["glb_url"].endswith("twin.glb") and t["zones"]["chest"]["value_cm"] > 0
    assert client.get("/v1/me/twin").status_code == 401
    other = client.get("/v1/me/twin", headers=user("demo-stranger"))
    assert other.status_code == 404


def test_product_check_and_store_view_has_no_measurements(client, session):
    h = user(session["uid"])
    prods = client.get("/v1/me/products", headers=h).json()
    assert len(prods) == 6
    fit = client.get("/v1/me/fits/dtc-shirt", headers=h).json()
    assert fit["recommended_size"] and fit["zones"] and fit["heatmap_glb_url"].endswith("heat-shirt.glb")
    sv = str(fit["store_view"])
    assert "ease" not in sv and "value_cm" not in sv
    blouse = client.get("/v1/me/fits/dtc-blouse", headers=h).json()
    assert blouse["recommended_size"] is None and blouse["shown_size"]    # an honest "no size fits" case still renders
    assert client.get("/v1/me/fits/dtc-shirt", headers=user("demo-stranger")).status_code == 404


def test_occasion_test_for_every_event(client, session):
    h = user(session["uid"])
    events = client.get("/v1/me/events", headers=h).json()
    assert len(events) == 5
    tomorrow = (datetime.now(timezone(timedelta(hours=5, minutes=30))).date() + timedelta(days=1)).isoformat()
    by_title = {e["title"]: e for e in events}
    assert by_title["Client meeting"]["starts_at"].startswith(f"{tomorrow}T10:00")
    assert by_title["Wedding reception"]["starts_at"].startswith(f"{tomorrow}T19:00")
    assert datetime.fromisoformat(by_title["Sangeet"]["starts_at"]).weekday() == 5
    for e in events:
        r = client.post("/v1/me/occasion-test", json={"event_id": e["id"]}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert {p["slot"] for p in body["outfit"]} >= {"top", "bottom", "shoes"} and not body["missing_slots"]
        assert not BANNED.search(str(body)), body
    assert client.post("/v1/me/occasion-test", json={"event_id": "nope"}, headers=h).status_code == 404


def test_orders_unboxing_and_returns(client, session):
    orders = client.get("/v1/me/orders", headers=user(session["uid"])).json()
    items = [i for o in orders for i in o["items"]]
    assert len(orders) == 3 and len(items) == 4 and all(i["unboxing"] and i["return"] for i in items)


def test_gift_flow(client, session):
    h = user(session["uid"])
    g = client.get(f"/v1/gifts/{session['gift_id']}", headers=h).json()
    assert g["status"] == "pending" and g["can_approve"]
    assert client.post(f"/v1/gifts/{session['gift_id']}/approve", headers=h).json()["status"] == "approved"
    view = client.get(f"/v1/me/gifts/{session['gift_id']}/giver-view", headers=h).json()
    assert view["status"] == "approved" and view["recommended_size"] and set(view) == {"id", "status", "recommended_size", "confidence"}


def test_widget_query_through_consent(client, session):
    h = user(session["uid"])
    c = client.post("/v1/consents", json={"merchant": "pk_demo_threads", "origin": ORIGIN}, headers=h)
    assert c.status_code == 200, c.text
    prod = seed_mod.PRODUCTS["shirt"]
    r = client.post("/v1/fit-query", json={"product": prod, "consent_token": c.json()["token"]}, headers={"X-Merchant-Key": "pk_demo_threads", "Origin": ORIGIN})
    assert r.status_code == 200 and r.json()["recommended_size"]
    assert "value_cm" not in r.text and "ease" not in r.text


def test_returns_desk_is_per_sandbox(client, session):
    h = {**user(session["uid"]), "X-Merchant-Id": session["merchant_id"]}
    rows = client.get("/v1/biz/returns", headers=h).json()
    assert len(rows) == 4
    other = client.post("/v1/demo/session").json()
    assert {r["id"] for r in rows}.isdisjoint({r["id"] for r in client.get("/v1/biz/returns", headers={**user(other["uid"])}).json()})
    assert client.get("/v1/biz/kpis", headers=h).status_code == 200


def test_chart_studio_has_seeded_charts(client, session):
    charts = client.get("/v1/biz/charts", headers=user(session["uid"])).json()
    assert len(charts) >= 3 and all(c["live"] is False for c in charts)
