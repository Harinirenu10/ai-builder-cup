"""Orders with a fit guarantee, nudge events, returns routing through the API, and the returns desk."""

import json
import re

import pytest

from apilib.orders import guarantee_applies
from apilib.returns import get_summarizer
from fitcore.fit.config import get_config
from tests.conftest import KEY, ORIGIN, ROOT, user

STORE_PRODUCTS = json.loads((ROOT / "web" / "store" / "products.json").read_text(encoding="utf-8"))
HEADERS = {"X-Merchant-Key": KEY, "Origin": ORIGIN}
STAFF = "staff-1"


@pytest.fixture
def shirt():
    return STORE_PRODUCTS["shirt"]


@pytest.fixture
def merchant(store):
    store.update("merchants", "m_demo", {"staff_uids": [STAFF], "return_policy": "review"})
    store.set("merchants", "m_other", {"name": "Other Shop", "publishable_key": "pk_test_other", "allowed_origins": ["https://other.example"], "staff_uids": ["other-staff"]})
    return store.get("merchants", "m_demo")


def recommended(client, product, consent) -> tuple[str, float]:
    r = client.post("/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers=HEADERS).json()
    return r["recommended_size"], r["confidence"]


def order(client, product, size, token=None, sku="dtc-shirt", extra=None):
    item = {"sku": sku, "title": "Oxford Button-Down Shirt", "size": size, "product": product, **(extra or {})}
    return client.post("/v1/orders", json={"items": [item], "consent_token": token}, headers=HEADERS)


def sizes_of(product):
    return list(product["size_chart"]["sizes"])


def other_size(product, rec):
    return next(s for s in sizes_of(product) if s != rec)


# ----------------------------------------------------------------------------- guarantee


@pytest.mark.parametrize("bought,rec,conf,expected", [("M", "M", 0.9, True), ("M", "M", 0.7, True), ("M", "M", 0.69, False), ("L", "M", 0.95, False), ("M", None, None, False), ("M", "M", None, False)])
def test_guarantee_rule(bought, rec, conf, expected):
    assert get_config().guarantee.min_confidence == 0.7
    assert guarantee_applies(bought, rec, conf) is expected


def test_order_with_recommended_size_is_guaranteed(client, shirt, consent, store, merchant):
    rec, conf = recommended(client, shirt, consent)
    body = order(client, shirt, rec, consent["token"]).json()
    assert body["consent"] == "ok" and body["items"][0] == {"sku": "dtc-shirt", "size": rec, "guarantee": True, "recommended_size": rec, "confidence": conf, "note": None}
    doc = store.get("orders", body["order_id"])
    assert doc["uid"] == "alice" and doc["merchant_id"] == "m_demo"
    assert doc["items"][0] == {"sku": "dtc-shirt", "title": "Oxford Button-Down Shirt", "qty": 1, "size": rec, "recommended_size_at_purchase": rec,
                               "confidence_at_purchase": conf, "guarantee": True}
    assert "created_at" in doc and doc["product_snapshots"]["dtc-shirt"]["size_chart"]["sizes"]


def test_another_size_is_not_guaranteed_and_client_claims_are_ignored(client, shirt, consent):
    rec, _ = recommended(client, shirt, consent)
    wrong = other_size(shirt, rec)
    body = order(client, shirt, wrong, consent["token"], extra={"guarantee": True, "recommended_size_at_purchase": wrong, "confidence_at_purchase": 0.99}).json()
    item = body["items"][0]
    assert item["guarantee"] is False and item["recommended_size"] == rec


@pytest.mark.parametrize("make_token,state", [(lambda c: None, "none"), (lambda c: "x" * 40, "invalid")])
def test_checkout_works_without_valid_consent_but_has_no_guarantee(client, shirt, consent, make_token, state):
    r = order(client, shirt, "M", make_token(consent))
    assert r.status_code == 200 and r.json()["consent"] == state and r.json()["items"][0]["guarantee"] is False


def test_revoked_consent_checks_out_without_guarantee(client, shirt, consent, store):
    client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice"))
    body = order(client, shirt, "M", consent["token"]).json()
    assert body["consent"] == "revoked" and body["items"][0]["guarantee"] is False
    assert store.get("orders", body["order_id"])["uid"] is None


def test_shopper_without_a_twin_checks_out_with_a_note(client, shirt, store):
    token = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("carol")).json()["token"]
    item = order(client, shirt, "M", token).json()["items"][0]
    assert item["guarantee"] is False and item["note"] == "no_twin"


def test_order_checks_key_and_origin(client, shirt):
    assert order(client, shirt, "M").status_code == 200
    assert client.post("/v1/orders", json={"items": [{"sku": "a", "size": "M", "product": shirt}]}, headers={"X-Merchant-Key": KEY, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/v1/orders", json={"items": [{"sku": "a", "size": "M", "product": shirt}]}).status_code == 401
    assert client.post("/v1/orders", json={"items": []}, headers=HEADERS).status_code == 422


# ----------------------------------------------------------------------------- events


def test_events_are_logged_without_user_or_body_data(client, store):
    for t in ("nudge_shown", "nudge_accepted", "nudge_dismissed"):
        assert client.post("/v1/store-events", json={"type": t, "product_ref": "dtc-shirt"}, headers=HEADERS).status_code == 204
    rows = store.find("events")
    assert {r["type"] for r in rows} == {"nudge_shown", "nudge_accepted", "nudge_dismissed"}
    assert all(set(r) == {"id", "merchant_id", "type", "product_ref", "ts"} for r in rows)


def test_events_reject_bad_origin_type_and_key(client):
    assert client.post("/v1/store-events", json={"type": "nudge_shown", "product_ref": "x"}, headers={"X-Merchant-Key": KEY, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/v1/store-events", json={"type": "made_up", "product_ref": "x"}, headers=HEADERS).status_code == 422
    assert client.post("/v1/store-events", json={"type": "nudge_shown", "product_ref": "x"}, headers={"Origin": ORIGIN}).status_code == 401


# ----------------------------------------------------------------------------- returns through the API


def add_unboxing(store, order_id, sku, size_label, row, shift=0.0, uid="ub1"):
    store.set("unboxing_measurements", uid, {"order_id": order_id, "sku": sku, "size_label": size_label, "measured_cm": {z: v + shift for z, v in row.items()},
                                            "overlay_image_url": "/store/img/unboxing/x.svg", "source": "simulated"})
    return uid


def row(product, size):
    return {z: v for z, v in product["size_chart"]["sizes"][size].items() if z in ("chest", "garment_length", "sleeve_length")}


def file_return(client, order_id, reason, ub=None, item_index=0, note=None):
    body = {"order_id": order_id, "item_index": item_index, "reason": reason, "unboxing_measurement_id": ub, "note": note}
    return client.post("/v1/returns", json=body, headers=HEADERS)


@pytest.fixture
def guaranteed_order(client, shirt, consent, merchant):
    rec, _ = recommended(client, shirt, consent)
    return order(client, shirt, rec, consent["token"]).json()["order_id"], rec


def test_branch1_label_mismatch_auto_exchange_and_flag(client, shirt, store, guaranteed_order):
    oid, rec = guaranteed_order
    sizes = sizes_of(shirt)
    smaller = sizes[max(sizes.index(rec) - 1, 0)] if sizes.index(rec) else sizes[1]
    ub = add_unboxing(store, oid, "dtc-shirt", rec, row(shirt, smaller))
    r = file_return(client, oid, "too_small", ub).json()
    assert r["status"] == "approved_exchange" and "approved" in r["message"]
    doc = store.get("returns", r["return_id"])
    assert doc["routing"]["route"] == "auto_exchange" and doc["routing"]["seller_label_error"] is True
    assert doc["decision_log"] and doc["decision_log"][0]["rule"] == "guarantee"


def test_branch2_matches_label_exchanges_to_next_size_and_logs_model_miss(client, shirt, store, guaranteed_order):
    oid, rec = guaranteed_order
    ub = add_unboxing(store, oid, "dtc-shirt", rec, row(shirt, rec))
    r = file_return(client, oid, "too_small", ub).json()
    sizes = sizes_of(shirt)
    expected = sizes[sizes.index(rec) + 1] if sizes.index(rec) + 1 < len(sizes) else None
    doc = store.get("returns", r["return_id"])
    assert doc["routing"]["model_miss"] is True
    assert (r["status"], r["exchange_size"]) == (("approved_exchange", expected) if expected else ("pending_review", None))


def test_branch3_wrong_size_received_matching_label_goes_to_a_person(client, shirt, store, guaranteed_order):
    oid, rec = guaranteed_order
    ub = add_unboxing(store, oid, "dtc-shirt", rec, row(shirt, rec))
    r = file_return(client, oid, "wrong_size_received", ub).json()
    assert r["status"] == "pending_review" and "A person" in r["message"] and "declined" in r["message"]
    assert store.get("returns", r["return_id"])["routing"]["route"] == "human_review"


def test_branch4_no_guarantee_uses_merchant_policy(client, shirt, consent, store, merchant):
    rec, _ = recommended(client, shirt, consent)
    oid = order(client, shirt, other_size(shirt, rec), consent["token"]).json()["order_id"]
    assert file_return(client, oid, "too_big").json()["status"] == "pending_review"       # default policy: review
    store.update("merchants", "m_demo", {"return_policy": "refund"})
    oid2 = order(client, shirt, other_size(shirt, rec), consent["token"], sku="dtc-shirt-2").json()["order_id"]
    r = file_return(client, oid2, "too_big").json()
    assert r["status"] == "approved_refund" and "refund" in r["message"]


def test_return_validation_and_ownership(client, shirt, store, consent, guaranteed_order):
    oid, rec = guaranteed_order
    assert file_return(client, "nope-nope-nope", "too_small").status_code == 404
    assert file_return(client, oid, "too_small", item_index=5).status_code == 422
    assert file_return(client, oid, "not_a_reason").status_code == 422
    other = client.post("/v1/returns", json={"order_id": oid, "item_index": 0, "reason": "too_small"}, headers={"X-Merchant-Key": "pk_test_other", "Origin": "https://other.example"})
    assert other.status_code == 404                                                      # another merchant cannot see this order
    other_order = order(client, shirt, rec, consent["token"], sku="dtc-other").json()["order_id"]
    ub = add_unboxing(store, other_order, "dtc-other", rec, row(shirt, rec))
    assert file_return(client, oid, "too_small", ub).status_code == 422                  # unboxing belongs to a different order
    assert file_return(client, oid, "too_small").status_code == 200
    assert file_return(client, oid, "too_small").status_code == 409                      # one return per item


# ----------------------------------------------------------------------------- desk


@pytest.fixture
def seeded(client, shirt, store, consent, merchant):
    rec, _ = recommended(client, shirt, consent)
    oid = order(client, shirt, rec, consent["token"]).json()["order_id"]
    sizes = sizes_of(shirt)
    other = sizes[0] if rec != sizes[0] else sizes[1]
    ub = add_unboxing(store, oid, "dtc-shirt", rec, row(shirt, other))
    rid = file_return(client, oid, "too_small", ub, note="Felt tight").json()["return_id"]
    return {"oid": oid, "rid": rid, "rec": rec}


def desk(client, method, path, uid=STAFF, **kw):
    return getattr(client, method)(path, headers=user(uid), **kw)


def test_desk_requires_a_staff_account(client, seeded):
    assert client.get("/v1/biz/returns").status_code == 401
    assert desk(client, "get", "/v1/biz/returns", uid="alice").status_code == 403
    assert desk(client, "get", "/v1/biz/returns", uid="other-staff").json() == []          # another merchant's staff see nothing of ours
    assert desk(client, "get", f"/v1/biz/returns/{seeded['rid']}", uid="other-staff").status_code == 404
    assert desk(client, "get", "/v1/biz/kpis", uid="alice").status_code == 403


def test_evidence_card_holds_spec_measured_twin_verdict_and_routing_reason(client, seeded, shirt):
    card = desk(client, "get", f"/v1/biz/returns/{seeded['rid']}").json()
    assert card["product_title"] == "Oxford Button-Down Shirt" and card["ordered_size"] == seeded["rec"] and card["guarantee"] is True
    assert card["twin_verdict"]["recommended_size"] == seeded["rec"] and card["twin_verdict"]["confidence"] >= 0.7
    ev = card["evidence"]
    assert ev["overlay_image_url"] and ev["source"] == "simulated" and card["simulated"] is True
    assert ev["spec_cm"]["chest"] == shirt["size_chart"]["sizes"][seeded["rec"]]["chest"]
    assert ev["measured_cm"]["chest"] != ev["spec_cm"]["chest"] and ev["diff_cm"]["chest"] == pytest.approx(ev["measured_cm"]["chest"] - ev["spec_cm"]["chest"], abs=0.1)
    assert "Rule 1" in card["routing_reason"] and card["routing"]["seller_label_error"] is True and card["note"] == "Felt tight"
    assert desk(client, "get", "/v1/biz/returns").json()[0]["id"] == seeded["rid"]


def test_desk_never_exposes_body_measurements(client, seeded, store):
    twin = store.get("twins", "alice")["measurements"]["zones"]
    texts = [desk(client, "get", "/v1/biz/returns").text, desk(client, "get", f"/v1/biz/returns/{seeded['rid']}").text, desk(client, "get", "/v1/biz/kpis").text]
    desk(client, "post", f"/v1/biz/returns/{seeded['rid']}/summary")
    texts.append(desk(client, "get", f"/v1/biz/returns/{seeded['rid']}").text)
    blob = " ".join(texts)
    for key in ("shape", "beta", "twin_measurements", "torso_length", "back_rise", "neck"):
        assert f'"{key}"' not in blob, key
    assert "owner_id" not in blob and "alice" not in blob
    # twin values that are not also garment chart values never appear (shirt chart values are garment, so compare the non-colliding ones)
    garment_values = {v for r in STORE_PRODUCTS["shirt"]["size_chart"]["sizes"].values() for v in r.values()}
    for z in twin.values():
        if z["value_cm"] not in garment_values:
            assert not re.search(rf"(?<![\d.]){re.escape(str(z['value_cm']))}(?![\d.])", blob), z


def test_decisions_approve_exchange_review_append_to_the_log_and_there_is_no_reject(client, seeded):
    rid = seeded["rid"]
    for action, status in (("review", "in_review"), ("approve", "approved_refund"), ("exchange", "approved_exchange")):
        card = desk(client, "post", f"/v1/biz/returns/{rid}/decision", json={"action": action, "note": "checked"}).json()
        assert card["status"] == status
        last = card["decision_log"][-1]
        assert last["rule"] == "human_decision" and last["actor"] == STAFF and last["action"] == action and "checked" in last["text"]
    assert desk(client, "post", f"/v1/biz/returns/{rid}/decision", json={"action": "reject"}).status_code == 422
    assert desk(client, "post", f"/v1/biz/returns/{rid}/decision", json={"action": "deny"}).status_code == 422
    assert desk(client, "post", f"/v1/biz/returns/{rid}/decision", json={"action": "approve"}, uid="alice").status_code == 403
    assert len(desk(client, "get", f"/v1/biz/returns/{rid}").json()["decision_log"]) >= 5       # the rule log is kept, decisions are added


def test_summary_is_generated_once_cached_and_regenerated_on_request(client, seeded, app_module):
    calls = []

    def fake(facts):
        calls.append(facts)
        return "The M shirt was reported too small and the measurement differs from the spec.", "model"

    app_module.app.dependency_overrides[get_summarizer] = lambda: fake
    rid = seeded["rid"]
    first = desk(client, "post", f"/v1/biz/returns/{rid}/summary").json()
    assert first["summary"]["source"] == "model" and len(calls) == 1
    desk(client, "post", f"/v1/biz/returns/{rid}/summary")
    assert len(calls) == 1
    desk(client, "post", f"/v1/biz/returns/{rid}/summary?refresh=true")
    assert len(calls) == 2
    facts = calls[0]
    assert facts["shopper_left_a_note"] is True and "Felt tight" not in json.dumps(facts)      # the shopper's own words never reach the model


def test_summary_falls_back_to_the_template_when_no_model_is_configured(client, seeded):
    s = desk(client, "post", f"/v1/biz/returns/{seeded['rid']}/summary").json()["summary"]
    assert s["source"] == "template" and s["text"]
    assert not re.search(r"fraud|lying|liar|scam|cheat", s["text"], re.I)


def test_kpis(client, shirt, consent, store, merchant, seeded):
    rec = seeded["rec"]
    order(client, shirt, other_size(shirt, rec), consent["token"], sku="dtc-plain")        # an order without guarantee
    for t in ("nudge_shown", "nudge_shown", "nudge_accepted"):
        client.post("/v1/store-events", json={"type": t, "product_ref": "dtc-shirt"}, headers=HEADERS)
    k = desk(client, "get", "/v1/biz/kpis").json()
    assert (k["orders"], k["guaranteed_orders"], k["guaranteed_share"]) == (2, 1, 0.5)
    assert (k["returns"], k["exchanges"], k["refunds"], k["open_returns"]) == (1, 1, 0, 0)
    assert (k["nudges_shown"], k["nudges_accepted"], k["nudges_accept_rate"]) == (2, 1, 0.5)
    assert k["seller_label_errors"] == 1 and k["seller_label_error_skus"] == ["dtc-shirt"]
    desk(client, "post", f"/v1/biz/returns/{seeded['rid']}/decision", json={"action": "approve"})
    k2 = desk(client, "get", "/v1/biz/kpis").json()
    assert (k2["exchanges"], k2["refunds"]) == (0, 1)


def test_kpis_with_no_data_do_not_divide_by_zero(client, merchant):
    k = desk(client, "get", "/v1/biz/kpis").json()
    assert k["guaranteed_share"] == 0.0 and k["nudges_accept_rate"] == 0.0 and k["returns"] == 0
