"""Returns: the shopper files one, the routing rules decide what they can, and the returns desk makes the rest of the decisions."""

from datetime import datetime, timezone
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, Header, Query
from pydantic import BaseModel, Field

from apilib.auth import RateLimiter, current_uid, err, get_limiter
from apilib.fitquery import check_origin, enforce_rate_limit, merchant_by_key, now_iso
from apilib.stores import Store, get_store
from fitcore.returns.routing import OrderItemFacts, Reason, Routing, Unboxing, route
from fitcore.returns.summary import case_facts, summarize
from fitcore.schemas import MeasurementZone, SizeChart

router = APIRouter()
OPEN = {"pending_review", "in_review"}


# ----------------------------------------------------------------------------- models


class ReturnIn(BaseModel):
    order_id: str = Field(min_length=8, max_length=64)
    item_index: int = Field(ge=0)
    reason: Reason
    note: str | None = Field(default=None, max_length=500)
    unboxing_measurement_id: str | None = Field(default=None, max_length=64)


class ReturnOut(BaseModel):
    return_id: str
    status: Literal["approved_exchange", "approved_refund", "pending_review"]
    exchange_size: str | None
    message: str


class DecisionIn(BaseModel):
    action: Literal["approve", "exchange", "review"]   # there is deliberately no "reject": a person approves, exchanges or escalates
    note: str | None = Field(default=None, max_length=500)


class Evidence(BaseModel):
    unboxing_measurement_id: str | None
    overlay_image_url: str | None
    source: str | None
    spec_cm: dict[str, float]
    measured_cm: dict[str, float]
    diff_cm: dict[str, float]
    label_check: dict[str, Any] | None


class ReturnCard(BaseModel):
    id: str
    created_at: str
    status: str
    reason: str
    note: str | None
    order_id: str
    sku: str
    product_title: str
    ordered_size: str
    guarantee: bool
    twin_verdict: dict[str, Any]       # size and confidence at purchase only
    routing: dict[str, Any]
    routing_reason: str
    decision_log: list[dict[str, Any]]
    evidence: Evidence
    summary: dict[str, Any] | None
    simulated: bool = False


class Kpis(BaseModel):
    orders: int
    guaranteed_orders: int
    guaranteed_share: float
    returns: int
    open_returns: int
    exchanges: int
    refunds: int
    nudges_shown: int
    nudges_accepted: int
    nudges_accept_rate: float
    seller_label_errors: int
    seller_label_error_skus: list[str]


# ----------------------------------------------------------------------------- core


def customer_message(status: str, route_name: str, exchange_size: str | None) -> str:
    """Plain words for the shopper. Nothing about suspicion; every outcome is an approval or a person looking at it."""
    if status == "approved_exchange":
        size = f" to size {exchange_size}" if exchange_size else ""
        return f"Your exchange{size} is approved. We will email you the next steps."
    if status == "approved_refund":
        return "Your return is approved and your refund is on its way."
    if route_name == "human_review":
        return "Thank you. A person at the store will look at your return and get back to you. Nothing has been declined."
    return "Thank you. We have your return and a person at the store will confirm the next step."


def create_return(store: Store, merchant: dict[str, Any], order: dict[str, Any], body: ReturnIn, now: datetime | None = None,
                  return_id: str | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if body.item_index >= len(order["items"]):
        raise err(422, "bad_item", "That order has no such item.")
    item = order["items"][body.item_index]
    if any(r["id"] != return_id and r["item"]["sku"] == item["sku"] and r["status"] in OPEN | {"approved_exchange", "approved_refund"}
           for r in store.find("returns", order_id=order["id"])):
        raise err(409, "already_filed", "A return for this item has already been filed.")
    unboxing = None
    if body.unboxing_measurement_id:
        rec = store.get("unboxing_measurements", body.unboxing_measurement_id)
        if not rec or rec["order_id"] != order["id"] or rec["sku"] != item["sku"]:
            raise err(422, "bad_unboxing", "That unboxing measurement does not belong to this order item.")
        unboxing = Unboxing(id=rec["id"], size_label=rec["size_label"], measured_cm={MeasurementZone(z): v for z, v in rec["measured_cm"].items()},
                            overlay_image_url=rec.get("overlay_image_url"), source=rec.get("source", "simulated"))
    chart = SizeChart.model_validate(order["product_snapshots"][item["sku"]]["size_chart"])
    routing: Routing = route(
        item=OrderItemFacts(sku=item["sku"], size=item["size"], guarantee=item["guarantee"], recommended_size_at_purchase=item.get("recommended_size_at_purchase"),
                            confidence_at_purchase=item.get("confidence_at_purchase")),
        reason=body.reason, chart=chart, unboxing=unboxing, policy=merchant.get("return_policy"), window_days=merchant.get("return_window_days"),
        ordered_at=datetime.fromisoformat(order["created_at"]), now=now)
    doc = {"merchant_id": merchant["id"], "order_id": order["id"], "item": item, "reason": body.reason, "note": body.note,
           "evidence": {"unboxing_measurement_id": body.unboxing_measurement_id},
           "routing": routing.model_dump(mode="json", exclude={"decision_log", "status"}), "status": routing.status,
           "decision_log": [e.model_dump() | {"ts": now.isoformat()} for e in routing.decision_log], "created_at": now.isoformat(), "summary": None}
    if return_id:
        store.set("returns", return_id, doc)
        return {**doc, "id": return_id}
    return {**doc, "id": store.add("returns", doc)}


def _spec_and_measured(store: Store, order: dict[str, Any], r: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, Any] | None]:
    item = r["item"]
    chart = SizeChart.model_validate(order["product_snapshots"][item["sku"]]["size_chart"])
    uid_ = r["evidence"].get("unboxing_measurement_id")
    rec = store.get("unboxing_measurements", uid_) if uid_ else None
    measured = {z: v for z, v in (rec or {}).get("measured_cm", {}).items()}
    spec_row = chart.sizes.get(item["size"], {}) if chart.chart_type == "garment" else {}
    spec = {z.value: v for z, v in spec_row.items() if z.value in measured} if measured else {z.value: v for z, v in spec_row.items()}
    return spec, measured, rec


def routing_reason(r: dict[str, Any]) -> str:
    log = r["decision_log"]
    key = next((e for e in log if e["rule"].startswith("rule_") or e["rule"] in ("standard_policy", "human_review")), None)
    return (key or log[-1])["text"] if log else ""


def card(store: Store, r: dict[str, Any]) -> ReturnCard:
    order = store.get("orders", r["order_id"]) or {"product_snapshots": {}, "items": []}
    snap = order["product_snapshots"].get(r["item"]["sku"], {})
    spec, measured, rec = _spec_and_measured(store, order, r) if snap else ({}, {}, None)
    item = r["item"]
    return ReturnCard(
        id=r["id"], created_at=r["created_at"], status=r["status"], reason=r["reason"], note=r.get("note"), order_id=r["order_id"], sku=item["sku"],
        product_title=snap.get("title", item["sku"]), ordered_size=item["size"], guarantee=item["guarantee"],
        twin_verdict={"recommended_size": item.get("recommended_size_at_purchase"), "confidence": item.get("confidence_at_purchase")},
        routing=r["routing"], routing_reason=routing_reason(r), decision_log=r["decision_log"],
        evidence=Evidence(unboxing_measurement_id=r["evidence"].get("unboxing_measurement_id"), overlay_image_url=(rec or {}).get("overlay_image_url"),
                          source=(rec or {}).get("source"), spec_cm=spec, measured_cm=measured,
                          diff_cm={z: round(measured[z] - spec[z], 1) for z in measured if z in spec}, label_check=r["routing"].get("label_check")),
        summary=r.get("summary"), simulated=bool((rec or {}).get("source") == "simulated"))


# ----------------------------------------------------------------------------- shopper


@router.post("/returns", response_model=ReturnOut)
def file_return(body: ReturnIn, x_merchant_key: str | None = Header(default=None), origin: str | None = Header(default=None),
                store: Store = Depends(get_store), limiter: RateLimiter = Depends(get_limiter)) -> ReturnOut:
    """The shopper files a return. The order id (an unguessable id from checkout) identifies the order; production stores should
    authenticate the shopper with their own session and call this from their backend."""
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    order = store.get("orders", body.order_id)
    if not order or order["merchant_id"] != m["id"]:
        raise err(404, "order_not_found", "We could not find that order.")
    r = create_return(store, m, order, body)
    return ReturnOut(return_id=r["id"], status=r["status"], exchange_size=r["routing"].get("exchange_size"),
                     message=customer_message(r["status"], r["routing"]["route"], r["routing"].get("exchange_size")))


# ----------------------------------------------------------------------------- desk


def desk_merchant(uid: str = Depends(current_uid), store: Store = Depends(get_store), x_merchant_id: str | None = Header(default=None)) -> dict[str, Any]:
    """The merchant whose staff list contains the signed-in user."""
    mine = [m for m in store.find("merchants") if uid in m.get("staff_uids", [])]
    if x_merchant_id:
        mine = [m for m in mine if m["id"] == x_merchant_id]
    if not mine:
        raise err(403, "not_staff", "This account is not on a merchant's returns desk.")
    return mine[0]


def _own_return(store: Store, m: dict[str, Any], return_id: str) -> dict[str, Any]:
    r = store.get("returns", return_id)
    if not r or r["merchant_id"] != m["id"]:
        raise err(404, "not_found", "No such return.")
    return r


@router.get("/biz/returns", response_model=list[ReturnCard])
def desk_list(status: str | None = Query(default=None), limit: int = Query(default=100, le=200),
              m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> list[ReturnCard]:
    rows = sorted(store.find("returns", merchant_id=m["id"]), key=lambda r: r["created_at"], reverse=True)
    if status:
        rows = [r for r in rows if r["status"] == status]
    return [card(store, r) for r in rows[:limit]]


@router.get("/biz/returns/{return_id}", response_model=ReturnCard)
def desk_detail(return_id: str, m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ReturnCard:
    return card(store, _own_return(store, m, return_id))


def get_summarizer() -> Callable[[dict[str, Any]], tuple[str, str]]:
    """Tests replace this; by default the flash model through fitcore.llm (with a template fallback)."""
    return summarize


@router.post("/biz/returns/{return_id}/summary", response_model=ReturnCard)
def desk_summary(return_id: str, refresh: bool = Query(default=False), m: dict = Depends(desk_merchant), store: Store = Depends(get_store),
                 summarizer: Callable = Depends(get_summarizer)) -> ReturnCard:
    """A neutral facts-and-numbers summary. Cached on the return so the model is not called twice for the same case."""
    r = _own_return(store, m, return_id)
    if r.get("summary") and not refresh:
        return card(store, r)
    c = card(store, r)
    days = (datetime.now(timezone.utc) - datetime.fromisoformat((store.get("orders", r["order_id"]) or {"created_at": now_iso()})["created_at"])).days
    facts = case_facts(product_title=c.product_title, ordered_size=c.ordered_size, reason=c.reason, guarantee=c.guarantee,
                       recommended_size=c.twin_verdict["recommended_size"], confidence=c.twin_verdict["confidence"], routing_route=r["routing"]["route"],
                       routing_status=r["status"], decision_log=[e["text"] for e in r["decision_log"]], spec_cm=c.evidence.spec_cm or None,
                       measured_cm=c.evidence.measured_cm or None, has_note=bool(r.get("note")), days_since_order=max(days, 0))
    text, source = summarizer(facts)
    store.update("returns", return_id, {"summary": {"text": text, "source": source, "created_at": now_iso()}})
    return card(store, store.get("returns", return_id))


_ACTIONS = {"approve": ("approved_refund", "approved the return"), "exchange": ("approved_exchange", "approved an exchange"), "review": ("in_review", "sent the case to review")}


@router.post("/biz/returns/{return_id}/decision", response_model=ReturnCard)
def desk_decision(return_id: str, body: DecisionIn, uid: str = Depends(current_uid), m: dict = Depends(desk_merchant),
                  store: Store = Depends(get_store)) -> ReturnCard:
    r = _own_return(store, m, return_id)
    status, verb = _ACTIONS[body.action]
    entry = {"rule": "human_decision", "text": f"A person on the desk {verb}." + (f" Note: {body.note}" if body.note else ""), "actor": uid, "action": body.action, "ts": now_iso()}
    store.update("returns", return_id, {"status": status, "decision_log": r["decision_log"] + [entry]})
    return card(store, store.get("returns", return_id))


@router.get("/biz/kpis", response_model=Kpis)
def desk_kpis(m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> Kpis:
    orders = store.find("orders", merchant_id=m["id"])
    guaranteed = [o for o in orders if any(i["guarantee"] for i in o["items"])]
    returns = store.find("returns", merchant_id=m["id"])
    events = store.find("events", merchant_id=m["id"])
    shown = sum(e["type"] == "nudge_shown" for e in events)
    accepted = sum(e["type"] == "nudge_accepted" for e in events)
    skus = sorted({r["item"]["sku"] for r in returns if r["routing"].get("seller_label_error")})
    return Kpis(orders=len(orders), guaranteed_orders=len(guaranteed), guaranteed_share=round(len(guaranteed) / len(orders), 3) if orders else 0.0,
                returns=len(returns), open_returns=sum(r["status"] in OPEN for r in returns),
                exchanges=sum(r["status"] == "approved_exchange" for r in returns), refunds=sum(r["status"] == "approved_refund" for r in returns),
                nudges_shown=shown, nudges_accepted=accepted, nudges_accept_rate=round(accepted / shown, 3) if shown else 0.0,
                seller_label_errors=sum(bool(r["routing"].get("seller_label_error")) for r in returns), seller_label_error_skus=skus)
