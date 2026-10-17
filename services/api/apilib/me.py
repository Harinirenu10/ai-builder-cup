"""The signed-in shopper's own screens: twin, product checks, closet, calendar, occasion test, orders and a gift preview.

Owner-only: every read is keyed by the caller's uid. These are the only endpoints that return the shopper's measurements.
Fit results and heatmaps were precomputed by seed/seed.py, so nothing here waits on the geometry service or a model.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from apilib.auth import current_uid, err
from apilib.models import GiftGiverView
from apilib.stores import Store, get_store
from fitcore.occasion import ClosetItem, OccasionResult, calendar_event, occasion_test
from fitcore.schemas import Measurements

router = APIRouter()


def _twin(store: Store, uid: str) -> dict[str, Any]:
    t = store.get("twins", uid)
    if not t or t.get("owner_id") != uid:
        raise err(404, "no_twin", "No twin yet for this account.")
    return t


@router.get("/me/home")
def home(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> dict[str, Any]:
    t = _twin(store, uid)
    return {"uid": uid, "simulated": bool(t.get("simulated")), "merchant_id": f"m_{uid}", "gift_id": f"{uid}-gift"}


@router.get("/me/twin")
def my_twin(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> dict[str, Any]:
    t = _twin(store, uid)
    return {"simulated": bool(t.get("simulated")), "note": t.get("note"), "body_model": t.get("body_model"), "height_cm": t.get("height_cm"),
            "zones": t["measurements"]["zones"], "pose_deltas": t.get("pose_deltas"), "glb_url": t.get("glb_url"),
            "privacy": "Only you can read this. Stores you connect get a size and a fit verdict, never these numbers."}


@router.get("/me/products")
def my_products(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> list[dict[str, Any]]:
    out = [{"ref": f["ref"], "slug": f["slug"], "title": f["title"], "price": f["price"], "category": f["category"],
            "recommended_size": f["recommended_size"], "confidence": f["confidence"], "image": f"/store/img/{f['slug']}-front.svg"}
           for f in store.find("fits", uid=uid)]
    return sorted(out, key=lambda p: p["title"])


def _zone_rows(fit: dict[str, Any]) -> list[dict[str, Any]]:
    size = next((s for s in fit["result"]["sizes"] if s["size"] == fit["shown_size"]), None)
    rows = []
    for z in (size or {"zones": {}})["zones"].values():
        rows.append({"zone": z["zone"], "hard": z["hard"], "verdict": z["verdict"], "verdict_after_wash": z["verdict_after_wash"],
                     "ease_now_cm": z["ease_now_cm"], "ease_after_wash_cm": z["ease_after_wash_cm"],
                     "pose_strain": [name for name, p in z.get("poses", {}).items() if p["strain"]]})
    return rows


@router.get("/me/fits/{ref}")
def my_fit(ref: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> dict[str, Any]:
    fit = store.get("fits", f"{uid}__{ref}")
    if not fit or fit.get("uid") != uid:
        raise err(404, "not_found", "No such product check.")
    r = fit["result"]
    return {"ref": ref, "title": fit["title"], "price": fit["price"], "category": fit["category"], "recommended_size": fit["recommended_size"],
            "shown_size": fit["shown_size"], "confidence": fit["confidence"], "zones": _zone_rows(fit),
            "sizes": [{"size": s["size"], "score": s["score"]} for s in r["sizes"]], "warnings": [w["message"] for w in r["warnings"]],
            "alterations": [a["text"] for a in r["alterations"]], "facts": r["explanation_facts"], "review_notes": fit["review_notes"],
            "heatmap_glb_url": fit["heatmap_glb_url"], "store_view": fit["store_view"], "simulated": True,
            "legend": {"ideal": "within the target range", "tight": "less ease than the cut needs", "loose": "more ease than the cut needs",
                       "short": "shorter than the target", "long": "longer than the target"}}


@router.get("/me/closet")
def my_closet(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> list[dict[str, Any]]:
    rows = sorted(store.find("wardrobe", uid=uid), key=lambda d: d["item_id"])
    return [{k: d[k] for k in ("item_id", "name", "slot", "style", "formality", "circles", "color", "size_label")} | {"measured": d.get("chart") is not None} for d in rows]


def _events(store: Store, uid: str) -> list:
    return sorted((calendar_event(d) for d in store.find("events", uid=uid)), key=lambda e: e.starts_at)


@router.get("/me/events")
def my_events(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> list[dict[str, Any]]:
    return [e.model_dump(mode="json") for e in _events(store, uid)]


class OccasionIn(BaseModel):
    event_id: str


@router.post("/me/occasion-test", response_model=OccasionResult)
def occasion(body: OccasionIn, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> OccasionResult:
    event = next((e for e in _events(store, uid) if e.id == body.event_id), None)
    if event is None:
        raise err(404, "not_found", "No such event.")
    t = _twin(store, uid)
    closet = [ClosetItem.model_validate(d) for d in store.find("wardrobe", uid=uid)]
    seated = ((t.get("pose_deltas") or {}).get("poses") or {}).get("seated", {}).get("delta_cm")
    return occasion_test(event, closet, Measurements.model_validate(t["measurements"]), seated)


@router.get("/me/orders")
def my_orders(uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> list[dict[str, Any]]:
    out = []
    for o in sorted(store.find("orders", uid=uid), key=lambda d: d["created_at"]):
        returns = store.find("returns", order_id=o["id"])
        unboxings = store.find("unboxing_measurements", order_id=o["id"])
        items = []
        for it in o["items"]:
            ret = next((r for r in returns if r["item"]["sku"] == it["sku"]), None)
            ub = next((u for u in unboxings if u["sku"] == it["sku"]), None)
            snap = o["product_snapshots"][it["sku"]]
            chart_row = snap["size_chart"]["sizes"].get(it["size"], {})
            items.append({"sku": it["sku"], "title": snap["title"], "size": it["size"], "guarantee": it["guarantee"],
                          "recommended_size_at_purchase": it.get("recommended_size_at_purchase"),
                          "unboxing": ub and {"measured_cm": ub["measured_cm"], "label_chart_cm": {z: chart_row.get(z) for z in ub["measured_cm"]},
                                              "overlay_image_url": ub.get("overlay_image_url"), "source": ub["source"]},
                          "return": ret and {"status": ret["status"], "route": ret["routing"]["route"], "reason": ret["reason"]}})
        out.append({"order_id": o["id"], "placed_at": o["created_at"], "items": items})
    return out


@router.get("/me/gifts/{gift_id}/giver-view", response_model=GiftGiverView)
def gift_giver_view(gift_id: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> GiftGiverView:
    """Exactly what the giver of a gift you approved can see. Shown to the recipient for transparency."""
    g = store.get("gifts", gift_id)
    if not g or g.get("recipient_uid") != uid:
        raise err(404, "not_found", "No such gift request.")
    exp = datetime.fromisoformat(g["expires_at"])
    status = "expired" if g["status"] == "pending" and exp < datetime.now(timezone.utc) else g["status"]
    return GiftGiverView(id=gift_id, status=status, recommended_size=g["recommended_size"], confidence=g["confidence"])  # type: ignore[arg-type]
