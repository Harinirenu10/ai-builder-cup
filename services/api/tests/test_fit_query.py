from datetime import datetime, timedelta, timezone

import jwt
import pytest

from apilib.auth import issue_consent_token, verify_consent_token
from tests.conftest import KEY, ORIGIN, user


def query(client, product, token, *, key=KEY, origin=ORIGIN):
    headers = {"X-Merchant-Key": key}
    if origin:
        headers["Origin"] = origin
    return client.post("/v1/fit-query", json={"product": product, "consent_token": token}, headers=headers)


# ----------------------------------------------------------------------------- happy path


def test_fit_query_returns_size_confidence_and_verdicts(client, product, consent):
    r = query(client, product, consent["token"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommended_size"] == "L" and 0 < body["confidence"] <= 1
    assert body["zone_verdicts"] and all(set(z) == {"zone", "verdict"} for z in body["zone_verdicts"])
    assert body["guarantee_eligible"] is True and body["alteration_hint"] is None


def test_alteration_hint_is_a_garment_side_phrase(client, product, consent, store):
    product["size_chart"]["sizes"]["L"]["sleeve_length"] = 63.5      # sleeves 5.5 cm over the twin's 58
    product["size_chart"]["sizes"]["XL"]["sleeve_length"] = 66
    body = query(client, product, consent["token"]).json()
    assert body["alteration_hint"] and "sleeves may need" in body["alteration_hint"] and "shortening" in body["alteration_hint"]


def test_call_without_origin_is_allowed_for_server_to_server(client, product, consent):
    assert query(client, product, consent["token"], origin=None).status_code == 200


def test_same_consent_is_reused_not_duplicated(client, store, consent):
    again = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("alice")).json()
    assert again["consent_id"] == consent["consent_id"] and len(store.find("consents", uid="alice")) == 1


# ----------------------------------------------------------------------------- origin and keys


def test_wrong_origin_is_rejected_on_query(client, product, consent):
    r = query(client, product, consent["token"], origin="https://evil.example")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "origin_not_allowed"


def test_other_merchants_origin_is_rejected(client, product, consent):
    assert query(client, product, consent["token"], origin="https://other.example").status_code == 403


def test_wrong_origin_cannot_get_a_consent_token(client, store):
    r = client.post("/v1/consents", json={"merchant": KEY, "origin": "https://evil.example"}, headers=user("alice"))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "origin_not_allowed"
    assert store.find("consents") == []                                # nothing was created


def test_origin_match_ignores_case_and_trailing_slash(client, product, consent):
    assert query(client, product, consent["token"], origin="https://SHOP.example/").status_code == 200


@pytest.mark.parametrize("key,code", [(None, "merchant_key_required"), ("pk_nope", "unknown_merchant")])
def test_missing_or_unknown_merchant_key_is_401(client, product, consent, key, code):
    headers = {"Origin": ORIGIN}
    if key:
        headers["X-Merchant-Key"] = key
    r = client.post("/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers=headers)
    assert r.status_code == 401 and r.json()["detail"]["code"] == code


def test_token_from_another_merchant_is_403(client, product, store, consent):
    other = client.post("/v1/consents", json={"merchant": "pk_test_other", "origin": "https://other.example"}, headers=user("alice")).json()
    r = query(client, product, other["token"])                          # other merchant's token, demo merchant's key
    assert r.status_code == 403 and r.json()["detail"]["code"] == "wrong_merchant"


# ----------------------------------------------------------------------------- consent lifecycle


def test_revoked_consent_returns_403(client, product, consent):
    assert query(client, product, consent["token"]).status_code == 200
    r = client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice"))
    assert r.status_code == 200 and r.json()["revoked_at"]
    after = query(client, product, consent["token"])
    assert after.status_code == 403 and after.json()["detail"]["code"] == "consent_revoked"


def test_revoke_is_owner_only_and_idempotent(client, consent):
    assert client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("bob")).status_code == 404
    first = client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice")).json()["revoked_at"]
    assert client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice")).json()["revoked_at"] == first


def test_after_revoking_a_new_consent_works_again(client, product, consent):
    client.post(f"/v1/consents/{consent['consent_id']}/revoke", headers=user("alice"))
    fresh = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("alice")).json()
    assert fresh["consent_id"] != consent["consent_id"]
    assert query(client, product, fresh["token"]).status_code == 200
    assert query(client, product, consent["token"]).status_code == 403   # the old token stays dead


def test_consent_list_shows_status_for_my_twin(client, consent):
    rows = client.get("/v1/consents", headers=user("alice")).json()
    assert rows[0]["merchant_name"] == "Demo Threads Co." and rows[0]["scope"] == "fit_verdict" and rows[0]["revoked_at"] is None
    assert client.get("/v1/consents", headers=user("bob")).json() == []


def test_consent_requires_sign_in(client):
    assert client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}).status_code == 401
    assert client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers={"Authorization": "Bearer garbage"}).status_code == 401


# ----------------------------------------------------------------------------- tokens


def test_token_is_30_days_and_carries_no_user_or_body_data(consent):
    claims = verify_consent_token(consent["token"])
    assert set(claims) == {"iss", "cid", "mid", "scope", "iat", "exp"}
    assert claims["exp"] - claims["iat"] == 30 * 86400


def test_expired_token_is_401(client, product):
    token, _ = issue_consent_token("c1", "m_demo", now=datetime.now(timezone.utc) - timedelta(days=31))
    r = query(client, product, token)
    assert r.status_code == 401 and r.json()["detail"]["code"] == "consent_expired"


def test_tampered_and_foreign_tokens_are_401(client, product, consent):
    forged = jwt.encode({"iss": "measured-fit", "cid": consent["consent_id"], "mid": "m_demo", "scope": "fit_verdict",
                         "iat": 1, "exp": 4102444800}, "wrong-secret-wrong-secret-wrong-secret", algorithm="HS256")
    assert query(client, product, forged).status_code == 401
    assert query(client, product, consent["token"][:-3] + "abc").status_code == 401
    none_alg = jwt.encode({"cid": "c", "mid": "m_demo"}, key=None, algorithm="none")
    assert query(client, product, none_alg).status_code == 401


def test_missing_jwt_secret_fails_closed(monkeypatch):
    monkeypatch.delenv("CONSENT_JWT_SECRET")
    monkeypatch.delenv("AUTH_MODE")
    with pytest.raises(RuntimeError):
        issue_consent_token("c", "m")


# ----------------------------------------------------------------------------- other failures, log, rate limit


def test_no_twin_gives_409(client, product, store):
    token = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("carol")).json()["token"]
    r = query(client, product, token)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_twin"


def test_chart_with_no_comparable_zones_is_422(client, product, consent):
    product["size_chart"]["sizes"] = {"M": {"neck": 40}}
    assert query(client, product, consent["token"]).status_code == 422


def test_query_log_has_no_body_data_and_no_user(client, product, consent, store):
    query(client, product, consent["token"])
    rows = store.find("fit_queries")
    assert len(rows) == 1
    assert set(rows[0]) == {"id", "merchant_id", "product_ref", "recommended_size", "confidence", "ts"}
    blob = str(rows)
    assert "alice" not in blob and "chest" not in blob and "value_cm" not in blob


def test_product_ref_from_merchant_is_logged_when_given(client, product, consent, store):
    product["ref"] = "SKU-KURTA-1"
    query(client, product, consent["token"])
    assert store.find("fit_queries")[0]["product_ref"] == "SKU-KURTA-1"


def test_rate_limit_is_per_merchant(client, product, consent, store):
    store.update("merchants", "m_demo", {"rate_limit_per_min": 3})
    codes = [query(client, product, consent["token"]).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    # a different merchant is unaffected
    other = client.post("/v1/consents", json={"merchant": "pk_test_other", "origin": "https://other.example"}, headers=user("alice")).json()
    assert query(client, product, other["token"], key="pk_test_other", origin="https://other.example").status_code == 200


# ----------------------------------------------------------------------------- CORS follows the registry


def test_cors_preflight_allows_registered_origin_only(client):
    ok = client.options("/v1/fit-query", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
    assert ok.status_code == 204 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "x-merchant-key" in ok.headers["access-control-allow-headers"]
    bad = client.options("/v1/fit-query", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert bad.status_code == 403 and "access-control-allow-origin" not in bad.headers


def test_cors_header_on_real_response_only_for_registered_origin(client, product, consent):
    good = query(client, product, consent["token"])
    assert good.headers.get("access-control-allow-origin") == ORIGIN
    evil = query(client, product, consent["token"], origin="https://evil.example")
    assert "access-control-allow-origin" not in evil.headers


def test_merchant_lookup_tells_the_popup_if_the_opener_is_registered(client):
    ok = client.get("/v1/merchants/lookup", params={"key": KEY, "origin": ORIGIN}).json()
    assert ok == {"name": "Demo Threads Co.", "origin_allowed": True}
    assert client.get("/v1/merchants/lookup", params={"key": KEY, "origin": "https://evil.example"}).json()["origin_allowed"] is False
    assert client.get("/v1/merchants/lookup", params={"key": "pk_nope", "origin": ORIGIN}).status_code == 401


def test_routes_also_exist_under_api_prefix_for_firebase_hosting(client, product, consent):
    r = client.post("/api/v1/fit-query", json={"product": product, "consent_token": consent["token"]}, headers={"X-Merchant-Key": KEY, "Origin": ORIGIN})
    assert r.status_code == 200
