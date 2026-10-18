"""Integration tests for POST /v1/fit-query/from-url: the real contract between services/api
and the Reader service (services/reader), connecting a customer-submitted product URL to the
Fit Engine with no manually-copied seed data.

The Reader itself (Playwright + Gemini) is not run here -- these tests mock at the HTTP
boundary (apilib.fitquery's own `requests` reference), which is the actual, documented contract
between the two services (readerlib/models.py::ReadResponse: status ok/blocked/unsupported/error,
and Product/size_chart on success). This proves services/api calls the Reader correctly and
handles every one of its documented outcomes with a clear, safe error -- it does NOT claim live
page extraction was verified; see the audit report for that distinction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import KEY, ORIGIN, user

ROOT = Path(__file__).resolve().parents[3]
KURTA = json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8"))


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
    """Drop-in replacement for the `requests` module, as apilib.fitquery uses it (.post, .exceptions)."""
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


def _reader_ok_response() -> dict:
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
            "size_chart": KURTA["size_chart"],
        },
        "reviews": None,
        "flags": [],
        "cached": False,
    }


def _from_url(client, url: str, token: str, *, key=KEY, origin=ORIGIN):
    headers = {"X-Merchant-Key": key}
    if origin:
        headers["Origin"] = origin
    return client.post("/v1/fit-query/from-url", json={"product_url": url, "consent_token": token}, headers=headers)


# ----------------------------------------------------------------------------- happy path / the real contract


def test_from_url_calls_reader_with_the_submitted_url_and_returns_a_fit_result(client, consent, monkeypatch):
    fake = _patch_reader(monkeypatch, lambda url, json, timeout: _FakeResponse(200, _reader_ok_response()))

    resp = _from_url(client, "https://shop.example/products/kurta", consent["token"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommended_size"] == "L"
    assert body["zone_verdicts"]
    # the actual contract: the reader was asked to read exactly the submitted URL
    assert len(fake.calls) == 1
    assert fake.calls[0]["json"] == {"url": "https://shop.example/products/kurta"}
    assert fake.calls[0]["url"].endswith("/read")


def test_from_url_response_has_no_measurement_or_body_fields(client, consent, monkeypatch):
    """Same privacy contract as /fit-query: verdict words only."""
    _patch_reader(monkeypatch, lambda url, json, timeout: _FakeResponse(200, _reader_ok_response()))
    body = _from_url(client, "https://shop.example/products/kurta", consent["token"]).json()
    blob = json.dumps(body).lower()
    for forbidden in ("value_cm", "measurement", "chest\":", "waist\":"):
        assert forbidden not in blob


# ----------------------------------------------------------------------------- Reader outcomes -> clear, safe errors


def test_from_url_reader_blocked_page(client, consent, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "blocked", "reason": "captcha"}))
    resp = _from_url(client, "https://blocked.example/p", consent["token"])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "page_blocked"


def test_from_url_unsupported_category(client, consent, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "unsupported", "reason": "not a garment"}))
    resp = _from_url(client, "https://shop.example/products/toaster", consent["token"])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unsupported_category"


def test_from_url_reader_internal_error_status(client, consent, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, {"status": "error", "reason": "extraction_failed: ValueError"}))
    resp = _from_url(client, "https://shop.example/products/weird", consent["token"])
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "reader_error"


def test_from_url_reader_product_missing_size_chart(client, consent, monkeypatch):
    ok = _reader_ok_response()
    ok["product"]["size_chart"] = None
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, ok))
    resp = _from_url(client, "https://shop.example/products/no-chart", consent["token"])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "no_size_chart"


def test_from_url_reader_http_error_response_surfaces_its_detail(client, consent, monkeypatch):
    """The Reader's own /read raises HTTPException(502/422, detail=reason) for unsafe/bad URLs."""
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(422, {"detail": "Only http(s) URLs are supported."}))
    resp = _from_url(client, "ftp://shop.example/p", consent["token"])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "reader_error"
    assert "http(s)" in resp.json()["detail"]["message"]


def test_from_url_reader_timeout(client, consent, monkeypatch):
    def respond(u, j, t):
        raise _FakeExceptions.Timeout()
    _patch_reader(monkeypatch, respond)
    resp = _from_url(client, "https://slow.example/p", consent["token"])
    assert resp.status_code == 504
    assert resp.json()["detail"]["code"] == "reader_timeout"


def test_from_url_reader_unreachable(client, consent, monkeypatch):
    def respond(u, j, t):
        raise _FakeExceptions.RequestException("connection refused")
    _patch_reader(monkeypatch, respond)
    resp = _from_url(client, "https://unreachable.example/p", consent["token"])
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "reader_unavailable"


# ----------------------------------------------------------------------------- existing auth/ownership preserved


def test_from_url_rejects_unknown_merchant_before_calling_reader(client, consent, monkeypatch):
    fake = _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    resp = _from_url(client, "https://shop.example/p", consent["token"], key="pk_bad")
    assert resp.status_code == 401
    assert fake.calls == []  # never reached the Reader: auth is checked first, same as /fit-query


def test_from_url_rejects_consent_for_a_different_merchant(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    # a consent minted for "m_other" used against the demo merchant's key
    r = client.post(
        "/v1/consents",
        json={"merchant": "pk_test_other", "origin": "https://other.example"},
        headers=user("alice"),
    )
    assert r.status_code == 200, r.text
    resp = _from_url(client, "https://shop.example/p", r.json()["token"])
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "wrong_merchant"


def test_from_url_no_twin_still_reported_clearly(client, monkeypatch):
    _patch_reader(monkeypatch, lambda u, j, t: _FakeResponse(200, _reader_ok_response()))
    r = client.post("/v1/consents", json={"merchant": KEY, "origin": ORIGIN}, headers=user("nobody_with_a_twin"))
    token = r.json()["token"]
    resp = _from_url(client, "https://shop.example/p", token)
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "no_twin"
