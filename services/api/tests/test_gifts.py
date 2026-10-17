from datetime import datetime, timedelta, timezone

from tests.conftest import user


def create(client, product, uid="alice"):
    r = client.post("/v1/gifts", json={"product": product, "title": "Blue kurta", "note": "Happy birthday"}, headers=user(uid))
    assert r.status_code == 200, r.text
    return r.json()


def test_gift_flow_giver_sees_only_size_and_confidence(client, product):
    gift = create(client, product, "carol")
    assert gift["recipient_path"] == f"/app/gift/{gift['id']}"
    assert client.get(f"/v1/gifts/{gift['id']}", headers=user("carol")).json() == {"id": gift["id"], "status": "pending", "recommended_size": None, "confidence": None}

    view = client.get(f"/v1/gifts/{gift['id']}", headers=user("alice")).json()
    assert view["can_approve"] is True and view["title"] == "Blue kurta" and "measurements" in view["shares"]
    assert not any(k in view for k in ("recommended_size", "confidence"))     # the recipient sees what is asked, not a verdict yet

    done = client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice"))
    assert done.status_code == 200 and done.json()["status"] == "approved"

    giver = client.get(f"/v1/gifts/{gift['id']}", headers=user("carol")).json()
    assert set(giver) == {"id", "status", "recommended_size", "confidence"}
    assert giver["status"] == "approved" and giver["recommended_size"] == "L" and 0 < giver["confidence"] <= 1


def test_approve_works_only_once(client, product):
    gift = create(client, product, "carol")
    assert client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice")).status_code == 200
    again = client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice"))
    assert again.status_code == 409 and again.json()["detail"]["code"] == "gift_used"
    other = client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("bob"))
    assert other.status_code == 409
    assert client.get(f"/v1/gifts/{gift['id']}", headers=user("bob")).status_code == 403


def test_giver_cannot_approve_own_request(client, product):
    gift = create(client, product, "alice")
    assert client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice")).status_code == 403


def test_recipient_without_twin_gets_409_and_gift_stays_open(client, product):
    gift = create(client, product, "carol")
    assert client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("dave")).json()["detail"]["code"] == "no_twin"
    assert client.get(f"/v1/gifts/{gift['id']}", headers=user("carol")).json()["status"] == "pending"


def test_decline(client, product):
    gift = create(client, product, "carol")
    assert client.post(f"/v1/gifts/{gift['id']}/decline", headers=user("alice")).json()["status"] == "declined"
    giver = client.get(f"/v1/gifts/{gift['id']}", headers=user("carol")).json()
    assert giver["status"] == "declined" and giver["recommended_size"] is None
    assert client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice")).status_code == 409


def test_expired_gift(client, product, store):
    gift = create(client, product, "carol")
    store.update("gifts", gift["id"], {"expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()})
    assert client.get(f"/v1/gifts/{gift['id']}", headers=user("carol")).json()["status"] == "expired"
    assert client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice")).status_code == 410


def test_gift_requires_sign_in_and_unknown_id_is_404(client, product):
    assert client.post("/v1/gifts", json={"product": product}).status_code == 401
    assert client.get("/v1/gifts/nope", headers=user("alice")).status_code == 404


def test_gift_paths_also_work_without_prefix(client, product):
    r = client.post("/gifts", json={"product": product}, headers=user("carol"))
    assert r.status_code == 200
    assert client.get(f"/gifts/{r.json()['id']}", headers=user("carol")).status_code == 200


def test_gift_record_keeps_no_body_data(client, product, store):
    gift = create(client, product, "carol")
    client.post(f"/v1/gifts/{gift['id']}/approve", headers=user("alice"))
    doc = store.get("gifts", gift["id"])
    assert "value_cm" not in str(doc) and "chest" not in {k for k in doc}
    assert doc["recommended_size"] == "L"
