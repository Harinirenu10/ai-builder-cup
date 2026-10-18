"""Tests for POST /v1/me/fit-from-url (apilib/fitquery.py::fit_from_url_for_me).

The customer-authenticated counterpart to the merchant-facing /v1/fit-query/from-url: no
merchant key or consent token, the authenticated caller (current_uid) IS the shopper, and their
own twin is always resolved server-side. Mirrors test_fit_query_from_url.py's approach of
mocking at the HTTP boundary (apilib.fitquery's own `requests` reference) -- the Reader itself is
not run here; see the audit report for that distinction.
"""

from __future__ import annotations

import json
from typing import Any

from tests.conftest import ROOT, user

KURTA = json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8"))
TWIN_ZONES = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]


class _FakeExceptions:
    class RequestException(Exception):
        pass

    class Timeout(RequestException):
        pass


class _FakeResponse:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class _FakeRequests:
    exceptions = _FakeExceptions

    def __init__(self, respond) -> None:
        self._respond = respond
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: Any = None, timeout: float | None = None) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self._respond(url, json, timeout)


def _patch_reader(monkeypatch, respond) -> _FakeRequests:
    fake = _FakeRequests(respond)
    monkeypatch.setattr("apilib.fitquery.requests", fake)
    return fake


_UNSET = object()


def _reader_ok_response(size_chart=_UNSET) -> dict:
    return {
        "status": "ok",
        "url": "https://shop.example/products/kurta",
        "product": {
            "id": "kurta_ref_123",
            "brand": "Demo Threads Co.",
            "title": "Classic Kurta",
            "category": KURTA["category"],
            "fit_type": KURTA["fit_type"],
            "fabric": [],
            "stretch_class": "none",
            "pre_shrunk": None,
            "size_chart": KURTA["size_chart"] if size_chart is _UNSET else size_chart,
        },
        "reviews": None,
        "flags": [],
        "cached": False,
    }


def _ask(client, url: str, uid: str):
    return client.post("/v1/me/fit-from-url", json={"product_url": url}, headers=user(uid))


# ----------------------------------------------------------------------------- authentication


def test_requires_authentication(client, monkeypatch):
    fake = _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    resp = client.post("/v1/me/fit-from-url", json={"product_url": "https://shop.example/p"})
    assert resp.status_code == 401
    assert fake.calls == []  # never reached the Reader: auth is checked first


# ----------------------------------------------------------------------------- happy path


def test_valid_product_url_returns_a_fit_recommendation_for_my_own_twin(client, monkeypatch):
    fake = _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    resp = _ask(client, "https://shop.example/products/kurta", "alice")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommended_size"] == "L"
    assert body["zone_verdicts"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["json"] == {"url": "https://shop.example/products/kurta"}
    assert fake.calls[0]["url"].endswith("/read")


def test_response_has_no_measurement_or_body_fields(client, monkeypatch):
    """Same privacy contract as /fit-query and /fit-query/from-url: verdict words only."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    body = _ask(client, "https://shop.example/products/kurta", "alice").json()
    blob = json.dumps(body).lower()
    for forbidden in ("value_cm", "measurement", "chest\":", "waist\":", "betas"):
        assert forbidden not in blob


# ----------------------------------------------------------------------------- Reader outcomes


def test_unsupported_page(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "unsupported", "reason": "not a garment"}))
    resp = _ask(client, "https://shop.example/products/toaster", "alice")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unsupported_category"


def test_blocked_page(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "blocked", "reason": "captcha"}))
    resp = _ask(client, "https://blocked.example/p", "alice")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "page_blocked"


def test_missing_size_chart(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response(size_chart=None)))
    resp = _ask(client, "https://shop.example/products/no-chart", "alice")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "no_size_chart"


def test_reader_unreachable(client, monkeypatch):
    def respond(u, j, t):
        raise _FakeExceptions.RequestException("connection refused")
    _patch_reader(monkeypatch, respond)
    resp = _ask(client, "https://unreachable.example/p", "alice")
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "reader_unavailable"


def test_reader_timeout(client, monkeypatch):
    def respond(u, j, t):
        raise _FakeExceptions.Timeout()
    _patch_reader(monkeypatch, respond)
    resp = _ask(client, "https://slow.example/p", "alice")
    assert resp.status_code == 504
    assert resp.json()["detail"]["code"] == "reader_timeout"


# ----------------------------------------------------------------------------- no-fit result


def test_no_size_fits_is_reported_clearly_not_as_an_error(client, monkeypatch):
    """A real chart where every size is wildly too small for alice's seeded twin: a normal 200
    response with recommended_size null, not an error -- the customer still gets zone verdicts
    explaining why."""
    tiny_chart = {
        "sizes": {"XS": {z: v * 0.3 for z, v in KURTA["size_chart"]["sizes"]["S"].items()}},
        "chart_type": KURTA["size_chart"]["chart_type"],
    }
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response(size_chart=tiny_chart)))
    resp = _ask(client, "https://shop.example/products/tiny", "alice")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommended_size"] is None
    # zone_verdicts() only reports the RECOMMENDED size's zones (existing, unmodified Fit Engine
    # behavior -- fitcore/fit/verdict.py::zone_verdicts), so an empty list here is the correct,
    # existing contract when nothing fits, not a bug in this endpoint.
    assert body["zone_verdicts"] == []
    assert isinstance(body["confidence"], float)


# ----------------------------------------------------------------------------- no twin yet


def test_no_twin_yet_is_409_not_a_crash(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    resp = _ask(client, "https://shop.example/products/kurta", "nobody_with_a_twin")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_twin"


# ----------------------------------------------------------------------------- cross-user isolation


def test_two_different_users_get_independent_results_from_their_own_twins(client, monkeypatch):
    """alice and bob are seeded (tests/conftest.py) with different chest measurements for the
    same product; each call must use ONLY the calling uid's own twin -- there is no field in the
    request body that could name a different user's twin, so this proves the server-side
    resolution is actually per-caller, not a shared/fixed twin."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    resp_alice = _ask(client, "https://shop.example/products/kurta", "alice")
    resp_bob = _ask(client, "https://shop.example/products/kurta", "bob")
    assert resp_alice.status_code == 200 and resp_bob.status_code == 200
    verdicts_alice = {zv["zone"]: zv["verdict"] for zv in resp_alice.json()["zone_verdicts"]}
    verdicts_bob = {zv["zone"]: zv["verdict"] for zv in resp_bob.json()["zone_verdicts"]}
    assert verdicts_alice.get("chest") != verdicts_bob.get("chest")


def test_rate_limited_per_uid(client, store, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    uid = "rate-limit-test-uid"
    store.set("twins", uid, {"measurements": {"zones": TWIN_ZONES}, "owner_id": uid})
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    first = _ask(client, "https://shop.example/products/kurta", uid)
    second = _ask(client, "https://shop.example/products/kurta", uid)
    assert first.status_code == 200, first.text
    assert second.status_code == 429


# ----------------------------------------------------------------------------- regression tests for Reader 502 & safe logging


def test_reader_internal_error_status_raises_502_with_safe_logging(client, caplog, monkeypatch):
    """When Reader reports status=error in JSON, API returns 502 reader_error and logs structured diagnostic."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "error", "reason": "extraction_failed: KeyError"}))
    with caplog.at_level("ERROR", logger="api.fitquery"):
        resp = _ask(client, "https://shop.example/products/error-item", "alice")
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "reader_error"
    assert "Reader returned status=error" in caplog.text
    assert "extraction_failed: KeyError" in caplog.text


def test_reader_http_502_server_error_surfaces_502(client, caplog, monkeypatch):
    """When Reader service responds with HTTP 502, API returns 502 reader_error (not 422)."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(502, {"detail": "Playwright browser timeout"}))
    with caplog.at_level("ERROR", logger="api.fitquery"):
        resp = _ask(client, "https://shop.example/products/broken", "alice")
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "reader_error"
    assert "Reader HTTP 502 server error" in caplog.text
    assert "Playwright browser timeout" in caplog.text


def test_reader_safe_logging_never_logs_tokens_or_query_params(client, caplog, monkeypatch):
    """Server-side diagnostic logs must strip query params and sensitive tokens from target URLs."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(502, {"detail": "upstream failure"}))
    sensitive_url = "https://user:password@shop.example/products/item?access_token=SUPER_SECRET_123&session=priv_456"
    with caplog.at_level("ERROR", logger="api.fitquery"):
        resp = _ask(client, sensitive_url, "alice")
    assert resp.status_code == 502
    assert "SUPER_SECRET_123" not in caplog.text
    assert "priv_456" not in caplog.text
    assert "password" not in caplog.text
    assert "shop.example/products/item" in caplog.text


def test_local_demo_store_product_fitting_success(client, monkeypatch):
    """Local demo-store URL (e.g. http://localhost:5173/store/shirt.html) returns successful fit."""
    demo_shirt_chart = {
        "chart_type": "garment",
        "sizes": {
            "S": {"chest": 108.0, "shoulder_width": 44.0, "sleeve_length": 60.0},
            "M": {"chest": 114.0, "shoulder_width": 46.0, "sleeve_length": 61.5},
            "L": {"chest": 120.0, "shoulder_width": 48.0, "sleeve_length": 63.0},
            "XL": {"chest": 126.0, "shoulder_width": 50.0, "sleeve_length": 64.5},
        },
    }
    shirt_resp = {
        "status": "ok",
        "url": "http://localhost:5173/store/shirt.html",
        "product": {
            "id": "demo_shirt_1",
            "brand": "Demo Threads Co.",
            "title": "Oxford Button-Down Shirt",
            "category": "shirt",
            "fit_type": "regular",
            "fabric": [],
            "stretch_class": "none",
            "pre_shrunk": None,
            "size_chart": demo_shirt_chart,
        },
        "reviews": None,
        "flags": [],
        "cached": False,
    }
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, shirt_resp))
    resp = _ask(client, "http://localhost:5173/store/shirt.html", "alice")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommended_size"] in ("S", "M", "L", "XL")
    assert isinstance(body["confidence"], float)
    assert len(body["zone_verdicts"]) > 0
