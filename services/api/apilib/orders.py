"""Mock-store checkout (orders with a fit guarantee) and nudge events."""

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from apilib.auth import RateLimiter, get_limiter
from apilib.fitquery import (
    check_origin, enforce_rate_limit, evaluate_for_uid, merchant_by_key, now_iso, resolve_consent,
)
from apilib.models import FitQueryProduct
from apilib.stores import Store, get_store
from fitcore.fit.config import get_config

router = APIRouter()


class OrderItemIn(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=160)
    size: str = Field(min_length=1, max_length=16)
    qty: int = Field(default=1, ge=1, le=10)
    product: FitQueryProduct


class OrderIn(BaseModel):
    items: list[OrderItemIn] = Field(min_length=1, max_length=20)
    consent_token: str | None = None


class OrderItemOut(BaseModel):
    sku: str
    size: str
    guarantee: bool
    recommended_size: str | None
    confidence: float | None
    note: str | None = None


class OrderOut(BaseModel):
    order_id: str
    consent: Literal["ok", "none", "revoked", "invalid"]
    items: list[OrderItemOut]


class EventIn(BaseModel):
    type: Literal["nudge_shown", "nudge_accepted", "nudge_dismissed"]
    product_ref: str = Field(min_length=1, max_length=64)


def guarantee_applies(purchased: str, recommended: str | None, confidence: float | None) -> bool:
    """The fit guarantee: the size bought is the recommended size, and confidence reaches the configured threshold."""
    return recommended is not None and confidence is not None and purchased == recommended and confidence >= get_config().guarantee.min_confidence


def price_items(store: Store, uid: str | None, items: list[OrderItemIn], merchant_name: str) -> list[dict[str, Any]]:
    """Per item: what the twin recommended at purchase and whether the guarantee applies. The client's claims are never used."""
    out = []
    for it in items:
        rec: str | None = None
        conf: float | None = None
        note: str | None = None
        if uid is None:
            note = "no_consent"
        else:
            try:
                fit = evaluate_for_uid(store, uid, it.product, merchant_name)
                rec, conf = fit.recommended_size, round(fit.confidence, 2)
            except HTTPException as exc:
                note = exc.detail["code"] if isinstance(exc.detail, dict) else "error"
        out.append({"sku": it.sku, "title": it.title, "qty": it.qty, "size": it.size, "recommended_size_at_purchase": rec,
                    "confidence_at_purchase": conf, "guarantee": guarantee_applies(it.size, rec, conf), "note": note})
    return out


def create_order(store: Store, merchant: dict[str, Any], uid: str | None, items: list[OrderItemIn], created_at: datetime | None = None,
                 order_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    priced = price_items(store, uid, items, merchant["name"])
    snapshots = {it.sku: {"title": it.title or it.sku, "category": it.product.category.value, "fit_type": it.product.fit_type.value,
                          "size_chart": it.product.size_chart.model_dump(mode="json")} for it in items}
    doc_items = [{k: v for k, v in p.items() if k != "note"} for p in priced]
    doc = {"merchant_id": merchant["id"], "uid": uid, "items": doc_items, "product_snapshots": snapshots,
           "created_at": (created_at or datetime.now(timezone.utc)).isoformat()}
    if order_id:
        store.set("orders", order_id, doc)
        return order_id, priced
    return store.add("orders", doc), priced


@router.post("/orders", response_model=OrderOut)
def post_order(body: OrderIn, x_merchant_key: str | None = Header(default=None), origin: str | None = Header(default=None),
               store: Store = Depends(get_store), limiter: RateLimiter = Depends(get_limiter)) -> OrderOut:
    """Place a (simulated) order. With a valid consent the server records the recommended size and confidence at purchase.

    Checkout never fails because of consent: without it the order is placed with no guarantee.
    """
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    uid, status = resolve_consent(store, m, body.consent_token)
    oid, priced = create_order(store, m, uid, body.items)
    return OrderOut(order_id=oid, consent=status,  # type: ignore[arg-type]
                    items=[OrderItemOut(sku=p["sku"], size=p["size"], guarantee=p["guarantee"], recommended_size=p["recommended_size_at_purchase"],
                                        confidence=p["confidence_at_purchase"], note=p["note"]) for p in priced])


@router.post("/store-events", status_code=204)
def post_event(body: EventIn, x_merchant_key: str | None = Header(default=None), origin: str | None = Header(default=None),
               store: Store = Depends(get_store), limiter: RateLimiter = Depends(get_limiter)) -> None:
    """Log a nudge event. No user id and no body data are recorded."""
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    store.add("events", {"merchant_id": m["id"], "type": body.type, "product_ref": body.product_ref, "ts": now_iso()})
