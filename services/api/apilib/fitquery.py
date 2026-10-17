"""Business-facing fit query and the consent flow behind it."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from apilib.auth import RateLimiter, current_uid, err, get_limiter, issue_consent_token, verify_consent_token, SCOPE
from apilib.models import (
    ConsentRequest, ConsentResponse, ConsentSummary, FitQueryProduct, FitQueryRequest, FitQueryResponse,
    MerchantPublic, ZoneVerdict,
)
from apilib.stores import Store, get_store
from fitcore.fit import evaluate
from fitcore.fit.verdict import alteration_hint, guarantee_eligible, zone_verdicts
from fitcore.schemas import FitResult, Measurements, Product

router = APIRouter()
DEFAULT_RATE_PER_MIN = 60


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


def evaluate_for_uid(store: Store, uid: str, p: FitQueryProduct, merchant_name: str = "merchant") -> FitResult:
    twin = store.get("twins", uid)
    if not twin or not twin.get("measurements"):
        raise err(409, "no_twin", "The shopper has not created a body twin yet.")
    product = Product(id=product_ref(p), brand=merchant_name, title=p.ref or "product", category=p.category, fit_type=p.fit_type,
                      fabric=p.fabric, stretch_class=p.stretch_class, pre_shrunk=p.pre_shrunk, size_chart=p.size_chart)
    try:
        return evaluate(Measurements.model_validate(twin["measurements"]), product)
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

    claims = verify_consent_token(body.consent_token)
    if claims["mid"] != m["id"]:
        raise err(403, "wrong_merchant", "This consent was given to a different merchant.")
    consent = store.get("consents", claims["cid"])
    if not consent or consent["merchant_id"] != m["id"] or consent["scope"] != SCOPE:
        raise err(403, "consent_not_found", "No matching consent.")
    if consent.get("revoked_at"):
        raise err(403, "consent_revoked", "The shopper has revoked this consent.")

    fit = evaluate_for_uid(store, consent["uid"], body.product, m["name"])
    resp = FitQueryResponse(
        recommended_size=fit.recommended_size,
        confidence=round(fit.confidence, 2),
        zone_verdicts=[ZoneVerdict(zone=z, verdict=v) for z, v in zone_verdicts(fit)],  # type: ignore[arg-type]
        alteration_hint=alteration_hint(fit),
        guarantee_eligible=guarantee_eligible(fit),
    )
    # The log holds no body data and no user id: only which merchant asked about which product and what we answered.
    store.add("fit_queries", {"merchant_id": m["id"], "product_ref": product_ref(body.product),
                              "recommended_size": resp.recommended_size, "confidence": resp.confidence, "ts": now_iso()})
    return resp
