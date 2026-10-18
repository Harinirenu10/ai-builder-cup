"""Gift mode: a giver asks for a size, the recipient approves once, the giver sees only size and confidence."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from apilib.auth import current_uid, err
from apilib.fitquery import evaluate_for_uid, now_iso
from apilib.models import GiftCreate, GiftCreated, GiftGiverView, GiftRecipientView
from apilib.stores import Store, get_store

router = APIRouter()
GIFT_DAYS = 14


def _status(g: dict) -> str:
    if g["status"] == "pending" and datetime.fromisoformat(g["expires_at"]) < datetime.now(timezone.utc):
        return "expired"
    return g["status"]


def _load(store: Store, gift_id: str) -> dict:
    g = store.get("gifts", gift_id)
    if not g:
        raise err(404, "not_found", "No such gift request.")
    return g


@router.post("/gifts", response_model=GiftCreated)
def create_gift(body: GiftCreate, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> GiftCreated:
    exp = datetime.now(timezone.utc) + timedelta(days=GIFT_DAYS)
    gid = store.add("gifts", {"giver_uid": uid, "recipient_uid": None, "status": "pending", "title": body.title, "note": body.note,
                              "product": body.product.model_dump(mode="json"), "created_at": now_iso(), "expires_at": exp.isoformat(),
                              "recommended_size": None, "confidence": None})
    return GiftCreated(id=gid, recipient_path=f"/app/gift/{gid}", expires_at=exp.isoformat())


@router.get("/gifts/{gift_id}")
def get_gift(gift_id: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> GiftGiverView | GiftRecipientView:
    """The giver gets only the size and confidence. Anyone else with the link gets the approval view, until one person has used it."""
    g = _load(store, gift_id)
    status = _status(g)
    if g["giver_uid"] == uid:
        return GiftGiverView(id=gift_id, status=status, recommended_size=g["recommended_size"], confidence=g["confidence"])  # type: ignore[arg-type]
    if g["recipient_uid"] not in (None, uid):
        raise err(403, "gift_used", "This gift request has already been answered.")
    return GiftRecipientView(id=gift_id, status=status, title=g["title"], note=g["note"], category=g["product"]["category"],  # type: ignore[arg-type]
                             can_approve=status == "pending")


@router.post("/gifts/{gift_id}/approve", response_model=GiftRecipientView)
def approve_gift(gift_id: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> GiftRecipientView:
    """The recipient approves once. Only the size and confidence are kept for the giver; the answer is not recomputed later."""
    g = _load(store, gift_id)
    if g["giver_uid"] == uid:
        raise err(403, "own_gift", "You cannot approve your own request.")
    if _status(g) == "expired":
        raise err(410, "gift_expired", "This gift request has expired.")
    if g["status"] != "pending" or g["recipient_uid"] not in (None, uid):
        raise err(409, "gift_used", "This gift request has already been answered.")
    from apilib.models import FitQueryProduct

    fit = evaluate_for_uid(store, uid, FitQueryProduct.model_validate(g["product"]))
    store.update("gifts", gift_id, {"status": "approved", "recipient_uid": uid, "recommended_size": fit.recommended_size,
                                    "confidence": round(fit.confidence, 2), "answered_at": now_iso()})
    return GiftRecipientView(id=gift_id, status="approved", title=g["title"], note=g["note"], category=g["product"]["category"], can_approve=False)  # type: ignore[arg-type]


@router.post("/gifts/{gift_id}/decline", response_model=GiftRecipientView)
def decline_gift(gift_id: str, uid: str = Depends(current_uid), store: Store = Depends(get_store)) -> GiftRecipientView:
    g = _load(store, gift_id)
    if g["giver_uid"] == uid:
        raise err(403, "own_gift", "You cannot answer your own request.")
    if g["status"] != "pending" or g["recipient_uid"] not in (None, uid):
        raise err(409, "gift_used", "This gift request has already been answered.")
    store.update("gifts", gift_id, {"status": "declined", "recipient_uid": uid, "answered_at": now_iso()})
    return GiftRecipientView(id=gift_id, status="declined", title=g["title"], note=g["note"], category=g["product"]["category"], can_approve=False)  # type: ignore[arg-type]
