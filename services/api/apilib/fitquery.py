"""Business-facing fit query and the consent flow behind it."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from apilib.auth import RateLimiter, current_uid, err, get_limiter, issue_consent_token, verify_consent_token, SCOPE
from apilib.models import (
    ConsentRequest, ConsentResponse, ConsentSummary, FitQueryFromUrlRequest, FitQueryProduct, FitQueryRequest,
    FitQueryResponse, MeFitFromUrlRequest, MerchantPublic, ZoneVerdict,
)
from apilib.stores import Store, get_store
from fitcore.fit import evaluate
from fitcore.fit.verdict import alteration_hint, guarantee_eligible, zone_verdicts
from fitcore.schemas import FitResult, Measurements, Product
from services.geometry.cloud_client import get_firestore_client

logger = logging.getLogger("api.fitquery")

router = APIRouter()
DEFAULT_RATE_PER_MIN = 60
# The Reader service's own hard timeout on one page read (readerlib/pipeline.py::HARD_LIMIT_S)
# is 25s; add margin for the network hop and the Reader's own startup/queueing time.
READER_READ_TIMEOUT_S = 35.0


def _safe_url_for_logging(url: str | None) -> str:
    """Safe URL representation for logging: scheme, host, and path only.

    Query parameters, fragments, userinfo, and raw payloads are stripped to prevent leaking
    tokens, session IDs, or personal data.
    """
    if not url:
        return "<empty-url>"
    try:
        raw = str(url).strip()
        parsed = urlparse(raw)
        if not parsed.scheme and not parsed.netloc:
            return parsed.path.split("?")[0]
        host = parsed.netloc.split("@")[-1]
        return f"{parsed.scheme}://{host}{parsed.path}"
    except Exception:
        return "<unparseable-url>"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def merchant_by_key(store: Store, key: str | None) -> dict[str, Any]:
    if not key:
        raise err(401, "merchant_key_required", "Send the merchant publishable key in X-Merchant-Key.")
    found = store.find("merchants", publishable_key=key)
    if not found:
        raise err(401, "unknown_merchant", "Unknown merchant key.")
    return found[0]


def check_origin(merchant: dict[str, Any], origin: str | None, *, required: bool) -> None:
    """Browsers always send Origin; it must be one the merchant registered. Server-to-server calls send none."""
    if origin is None:
        if required:
            raise err(403, "origin_required", "This request must come from a registered store origin.")
        return
    allowed = {norm_origin(o) for o in merchant.get("allowed_origins", [])}
    if norm_origin(origin) not in allowed:
        raise err(403, "origin_not_allowed", "This origin is not registered for the merchant.")


def enforce_rate_limit(m: dict[str, Any], limiter: RateLimiter) -> None:
    limit = int(m.get("rate_limit_per_min") or os.environ.get("RATE_LIMIT_PER_MIN", DEFAULT_RATE_PER_MIN))
    if limiter.check(f"merchant:{m['id']}", limit) is not None:
        raise err(429, "rate_limited", "Too many requests for this merchant. Try again shortly.")


def resolve_consent(store: Store, merchant: dict[str, Any], token: str | None) -> tuple[str | None, str]:
    """(uid, status) for an optional consent token. Status is ok, none, revoked or invalid; never raises."""
    if not token:
        return None, "none"
    try:
        claims = verify_consent_token(token)
    except HTTPException:
        return None, "invalid"
    if claims["mid"] != merchant["id"]:
        return None, "invalid"
    consent = store.get("consents", claims["cid"])
    if not consent or consent["merchant_id"] != merchant["id"] or consent["scope"] != SCOPE:
        return None, "invalid"
    if consent.get("revoked_at"):
        return None, "revoked"
    return consent["uid"], "ok"


def product_ref(p: FitQueryProduct) -> str:
    if p.ref:
        return p.ref
    canon = json.dumps({"c": p.category.value, "f": p.fit_type.value, "s": p.size_chart.model_dump(mode="json")}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _real_twin_for_uid(uid: str) -> dict[str, Any] | None:
    """The shopper's most recent real Body Twin from the Prompt 1 geometry pipeline.

    That pipeline (services/geometry/worker.py) stores twins in Firestore at
    twins/{twin_id} (a generated id, not the uid), via services/geometry/cloud_client.py
    -- a different store than the one this module otherwise uses (Store, twins/{uid}, a
    seeded stand-in for demos/tests). Looked up only as a fallback, after the Store, so
    every existing seeded-twin test and demo keeps working unchanged.
    """
    try:
        return get_firestore_client().get_latest_twin_for_user(uid)
    except Exception:
        return None


def resolve_twin_measurements_for_uid(store: Store, uid: str) -> Measurements | None:
    """The authenticated caller's own body twin, server-side -- never trusted from a request
    body. Store first (keeps every existing seeded-twin test/demo working unchanged), then the
    real Prompt 1 Firestore twin (see _real_twin_for_uid). None if uid has no twin anywhere.

    This is the one place "whose measurements are these" is decided; every endpoint that needs a
    body twin (here and services/api/main.py's /v1/fit, /v1/biz/verdict, /v1/tailor-card) calls
    this instead of accepting measurements directly in the request, which is what the security
    audit in eval/results.md found and fixed: a signed-in caller could otherwise submit anyone's
    twin data and get a result for it.
    """
    twin = store.get("twins", uid) or _real_twin_for_uid(uid)
    if not twin or not twin.get("measurements"):
        return None
    return Measurements.model_validate(twin["measurements"])


def evaluate_for_uid(store: Store, uid: str, p: FitQueryProduct, merchant_name: str = "merchant") -> FitResult:
    twin_measurements = resolve_twin_measurements_for_uid(store, uid)
    if twin_measurements is None:
        raise err(409, "no_twin", "The shopper has not created a body twin yet.")
    product = Product(id=product_ref(p), brand=merchant_name, title=p.ref or "product", category=p.category, fit_type=p.fit_type,
                      fabric=p.fabric, stretch_class=p.stretch_class, pre_shrunk=p.pre_shrunk, size_chart=p.size_chart)
    try:
        return evaluate(twin_measurements, product)
    except ValueError as exc:
        raise err(422, "unsupported_chart", str(exc)) from exc


# ----------------------------------------------------------------------------- merchant lookup and consent


@router.get("/merchants/lookup", response_model=MerchantPublic)
def lookup(key: str = Query(...), origin: str = Query(...), store: Store = Depends(get_store)) -> MerchantPublic:
    """Used by the consent popup to show who is asking, and to stop early if the opener is not registered."""
    m = merchant_by_key(store, key)
    allowed = {norm_origin(o) for o in m.get("allowed_origins", [])}
    return MerchantPublic(name=m["name"], origin_allowed=norm_origin(origin) in allowed)


@router.post("/consents", response_model=ConsentResponse)
def create_consent(body: ConsentRequest, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> ConsentResponse:
    """Called by the popup after the user signs in and approves. The token is only ever posted to a registered origin."""
    m = merchant_by_key(store, body.merchant)
    check_origin(m, body.origin, required=True)
    active = [c for c in store.find("consents", uid=uid, merchant_id=m["id"], scope=SCOPE) if not c.get("revoked_at")]
    if active:
        cid = active[0]["id"]
    else:
        cid = store.add("consents", {"uid": uid, "merchant_id": m["id"], "scope": SCOPE, "created_at": now_iso(), "revoked_at": None})
    token, exp = issue_consent_token(cid, m["id"])
    return ConsentResponse(consent_id=cid, token=token, expires_at=exp.isoformat(), merchant_name=m["name"], origin=norm_origin(body.origin))


@router.get("/consents", response_model=list[ConsentSummary])
def list_consents(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> list[ConsentSummary]:
    """For the customer app (My twin): which stores can ask for a fit verdict."""
    out = []
    for c in store.find("consents", uid=uid):
        m = store.get("merchants", c["merchant_id"]) or {"name": "Unknown store"}
        out.append(ConsentSummary(consent_id=c["id"], merchant_name=m["name"], scope=c["scope"], created_at=c["created_at"], revoked_at=c.get("revoked_at")))
    return sorted(out, key=lambda c: c.created_at, reverse=True)


@router.post("/consents/{consent_id}/revoke", response_model=ConsentSummary)
def revoke_consent(consent_id: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> ConsentSummary:
    c = store.get("consents", consent_id)
    if not c or c["uid"] != uid:
        raise err(404, "not_found", "No such consent.")  # same answer for other people's ids
    if not c.get("revoked_at"):
        store.update("consents", consent_id, {"revoked_at": now_iso()})
        c = store.get("consents", consent_id)
    m = store.get("merchants", c["merchant_id"]) or {"name": "Unknown store"}
    return ConsentSummary(consent_id=c["id"], merchant_name=m["name"], scope=c["scope"], created_at=c["created_at"], revoked_at=c["revoked_at"])


# ----------------------------------------------------------------------------- the query


def _consent_or_raise(store: Store, m: dict[str, Any], consent_token: str) -> dict[str, Any]:
    """The consent document for this token, or raises the same errors fit_query always has."""
    claims = verify_consent_token(consent_token)
    if claims["mid"] != m["id"]:
        raise err(403, "wrong_merchant", "This consent was given to a different merchant.")
    consent = store.get("consents", claims["cid"])
    if not consent or consent["merchant_id"] != m["id"] or consent["scope"] != SCOPE:
        raise err(403, "consent_not_found", "No matching consent.")
    if consent.get("revoked_at"):
        raise err(403, "consent_revoked", "The shopper has revoked this consent.")
    return consent


def _fit_query_response(store: Store, m: dict[str, Any], uid: str, product: FitQueryProduct, product_reference: str) -> FitQueryResponse:
    fit = evaluate_for_uid(store, uid, product, m["name"])
    resp = FitQueryResponse(
        recommended_size=fit.recommended_size,
        confidence=round(fit.confidence, 2),
        zone_verdicts=[ZoneVerdict(zone=z, verdict=v) for z, v in zone_verdicts(fit)],  # type: ignore[arg-type]
        alteration_hint=alteration_hint(fit),
        guarantee_eligible=guarantee_eligible(fit),
    )
    # The log holds no body data and no user id: only which merchant asked about which product and what we answered.
    store.add("fit_queries", {"merchant_id": m["id"], "product_ref": product_reference,
                              "recommended_size": resp.recommended_size, "confidence": resp.confidence, "ts": now_iso()})
    return resp


@router.post("/fit-query", response_model=FitQueryResponse)
def fit_query(
    body: FitQueryRequest,
    x_merchant_key: str | None = Header(default=None),
    origin: str | None = Header(default=None),
    store: Store = Depends(get_store),
    limiter: RateLimiter = Depends(get_limiter),
) -> FitQueryResponse:
    """Which size fits this shopper? Authenticated by the merchant key and the shopper's consent token.

    Returns verdict words only. In production, call this from the merchant's backend rather than the browser.
    """
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    consent = _consent_or_raise(store, m, body.consent_token)
    return _fit_query_response(store, m, consent["uid"], body.product, product_ref(body.product))


def _reader_base_url() -> str:
    return os.environ.get("READER_URL", "http://localhost:8002").rstrip("/")


def read_product_from_url(product_url: str, http_client: Any = None) -> FitQueryProduct:
    """Call the Reader service live and convert its extracted Product into a FitQueryProduct.

    Raises the same apilib.auth.err(...) HTTPException shape as the rest of this module, with a
    clear, safe (never a raw stack trace or internal detail) reason for every documented Reader
    outcome: unreachable/timeout, blocked, unsupported, extraction error, and no size chart found.
    Logs structured diagnostics on failures without leaking credentials, tokens, or raw image content.
    """
    requester = http_client or requests
    reader_base = _reader_base_url()
    safe_url = _safe_url_for_logging(product_url)
    endpoint = f"{reader_base}/read"

    try:
        resp = requester.post(endpoint, json={"url": product_url}, timeout=READER_READ_TIMEOUT_S)
    except requests.exceptions.Timeout as exc:
        logger.error(
            "Reader timeout after %.1fs calling %s for product_url=%s: %s",
            READER_READ_TIMEOUT_S, endpoint, safe_url, exc
        )
        raise err(504, "reader_timeout", "That product page took too long to read. Please try again.")
    except requests.exceptions.RequestException as exc:
        logger.error(
            "Reader unreachable at %s for product_url=%s: %s: %s",
            endpoint, safe_url, type(exc).__name__, exc
        )
        raise err(502, "reader_unavailable", "Could not reach the product reader right now. Please try again shortly.")

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
            reason = detail or "Could not read that product page."
        except Exception:
            detail = None
            reason = "Could not read that product page."

        if resp.status_code >= 500:
            logger.error(
                "Reader HTTP %d server error from %s for product_url=%s: detail=%s",
                resp.status_code, endpoint, safe_url, detail or "unknown"
            )
            raise err(502, "reader_error", str(reason))
        else:
            logger.warning(
                "Reader HTTP %d client error from %s for product_url=%s: detail=%s",
                resp.status_code, endpoint, safe_url, detail or "unknown"
            )
            raise err(422, "reader_error", str(reason))

    data = resp.json()
    status = data.get("status")
    if status == "blocked":
        logger.info("Reader page blocked for product_url=%s (reason=%s)", safe_url, data.get("reason"))
        raise err(422, "page_blocked", "This store's page could not be read right now.")
    if status == "unsupported":
        logger.info("Reader unsupported product category for product_url=%s (reason=%s)", safe_url, data.get("reason"))
        raise err(422, "unsupported_category", "This product isn't a supported garment category yet.")
    if status == "error":
        logger.error("Reader returned status=error for product_url=%s: reason=%s", safe_url, data.get("reason"))
        raise err(502, "reader_error", "Could not read that product page. Please try again.")

    product = data.get("product")
    if not product or not product.get("size_chart"):
        logger.warning("Product page had no size chart for product_url=%s", safe_url)
        raise err(422, "no_size_chart", "This product page has no size chart, so a fit recommendation isn't possible.")

    return FitQueryProduct(
        size_chart=product["size_chart"],
        category=product["category"],
        fit_type=product.get("fit_type", "regular"),
        fabric=product.get("fabric", []),
        stretch_class=product.get("stretch_class", "none"),
        pre_shrunk=product.get("pre_shrunk"),
        ref=product.get("id"),
    )


@router.post("/fit-query/from-url", response_model=FitQueryResponse)
def fit_query_from_url(
    body: FitQueryFromUrlRequest,
    x_merchant_key: str | None = Header(default=None),
    origin: str | None = Header(default=None),
    store: Store = Depends(get_store),
    limiter: RateLimiter = Depends(get_limiter),
    http_client: Any = None,
) -> FitQueryResponse:
    """Same as /fit-query, but for a product page the merchant has not already embedded a size
    chart for: the Reader service reads product_url live and its extracted size chart is used
    directly, with no manual copying of seed/sample data.
    """
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    consent = _consent_or_raise(store, m, body.consent_token)
    product = read_product_from_url(body.product_url, http_client=http_client)
    return _fit_query_response(store, m, consent["uid"], product, product_ref(product))


@router.post("/me/fit-from-url", response_model=FitQueryResponse)
def fit_from_url_for_me(
    body: MeFitFromUrlRequest,
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
    limiter: RateLimiter = Depends(get_limiter),
    http_client: Any = None,
) -> FitQueryResponse:
    """Customer-facing: "does this product fit ME?" from a shared/pasted product URL.

    No merchant key or consent token: the authenticated caller (current_uid) IS the shopper, so
    there is no third party being asked about someone else. The caller's own twin is always
    resolved server-side by evaluate_for_uid -- the request body has no field for a twin,
    measurements, or a user id, so a signed-in caller cannot submit (or ask about) anyone else's
    body. Reuses the exact same Reader-calling (read_product_from_url) and Fit Engine
    (evaluate_for_uid) logic as the merchant-facing /fit-query/from-url above; this endpoint adds
    only the auth model and skips the fit_queries merchant-analytics log (not applicable to a
    self-check). Rate-limited per uid, same mechanism /fit-query/from-url uses per merchant.
    """
    limit = int(os.environ.get("RATE_LIMIT_PER_MIN", DEFAULT_RATE_PER_MIN))
    if limiter.check(f"me:{uid}", limit) is not None:
        raise err(429, "rate_limited", "Too many requests. Try again shortly.")
    product = read_product_from_url(body.product_url, http_client=http_client)
    fit = evaluate_for_uid(store, uid, product, merchant_name="you")
    return FitQueryResponse(
        recommended_size=fit.recommended_size,
        confidence=round(fit.confidence, 2),
        zone_verdicts=[ZoneVerdict(zone=z, verdict=v) for z, v in zone_verdicts(fit)],  # type: ignore[arg-type]
        alteration_hint=alteration_hint(fit),
        guarantee_eligible=guarantee_eligible(fit),
    )
