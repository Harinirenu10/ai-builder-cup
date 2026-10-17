"""Demo sandboxes: POST /demo/session clones the seeded persona into a fresh uid and returns a sign-in token.

One sandbox per judge. Every cloned document carries `expire_at` (a Firestore TTL policy on that field deletes it after
DEMO_TTL_HOURS, see infra/setup.sh) and the sandbox gets its own merchant, so a judge's returns decisions never touch another judge's.
The persona is written by seed/seed.py. Nothing here calls a model.
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from apilib.auth import RateLimiter, err, get_limiter
from apilib.stores import TTL_FIELD, Store, get_store

router = APIRouter()
PERSONA = "demo-persona"
SHARED_MERCHANT = "m_demo_threads"
PER_IP_PER_MIN = 6
GLOBAL_PER_MIN = 120


class DemoSession(BaseModel):
    uid: str
    custom_token: str
    auth_mode: str
    expires_at: str
    merchant_id: str
    gift_id: str
    simulated: bool = True
    note: str = "Everything in this sandbox is simulated demo data. It is deleted automatically."


def mint_token(uid: str) -> tuple[str, str]:
    """(token, auth_mode). Dev mode returns the dev bearer; otherwise a Firebase custom token for signInWithCustomToken."""
    if os.environ.get("AUTH_MODE") == "dev":
        return f"dev:{uid}", "dev"
    import firebase_admin
    from firebase_admin import auth

    if not firebase_admin._apps:
        firebase_admin.initialize_app()  # application default credentials; the service account must be able to sign tokens
    token = auth.create_custom_token(uid)
    return (token.decode() if isinstance(token, bytes) else token), "firebase"


def get_token_minter() -> Callable[[str], tuple[str, str]]:
    return mint_token


def clone_rows(store: Store, uid: str, expire_at: datetime, sid: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Every persona document, rewritten for `uid`. Ids and ownership are replaced with one blanket text substitution."""
    sandbox_merchant = f"m_{uid}"

    def rewrite(coll: str, doc_id: str, doc: dict[str, Any], new_id: str | None = None) -> tuple[str, str, dict[str, Any]]:
        text = json.dumps({k: v for k, v in doc.items() if k != "id"}).replace(PERSONA, uid).replace(SHARED_MERCHANT, sandbox_merchant)
        out = json.loads(text)
        out[TTL_FIELD] = expire_at.isoformat()
        out["demo_session"] = sid
        return coll, new_id or doc_id.replace(PERSONA, uid), out

    rows: list[tuple[str, str, dict[str, Any]]] = []
    twin = store.get("twins", PERSONA)
    if not twin:
        raise err(503, "demo_not_seeded", "The demo data has not been seeded. Run seed/seed.py.")
    rows.append(rewrite("twins", PERSONA, twin, uid))
    for coll in ("wardrobe", "events", "fits"):
        rows += [rewrite(coll, d["id"], d) for d in store.find(coll, uid=PERSONA)]
    orders = store.find("orders", uid=PERSONA)
    rows += [rewrite("orders", d["id"], d) for d in orders]
    for o in orders:
        rows += [rewrite("returns", d["id"], d) for d in store.find("returns", order_id=o["id"])]
        rows += [rewrite("unboxing_measurements", d["id"], d) for d in store.find("unboxing_measurements", order_id=o["id"])]
    for g in store.find("gifts", demo_template=True):
        c, i, d = rewrite("gifts", g["id"], g)
        d["demo_template"] = False
        rows.append((c, i, d))
    rows += [rewrite("charts", d["id"], d, f"{uid}-{d['id']}") for d in store.find("charts", merchant_id=SHARED_MERCHANT)]
    shared = store.get("merchants", SHARED_MERCHANT) or {}
    c, i, m = rewrite("merchants", SHARED_MERCHANT, shared, sandbox_merchant)
    m.pop("publishable_key", None)        # the store widget keeps using the shared merchant key; the sandbox merchant is for the desk only
    m.update({"allowed_origins": [], "staff_uids": [uid]})
    rows.append((c, i, m))
    return rows


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@router.post("/demo/session", response_model=DemoSession)
def demo_session(request: Request, store: Store = Depends(get_store), limiter: RateLimiter = Depends(get_limiter),
                 minter: Callable[[str], tuple[str, str]] = Depends(get_token_minter)) -> DemoSession:
    if limiter.check(f"demo-ip:{_client_ip(request)}", PER_IP_PER_MIN) is not None or limiter.check("demo-global", GLOBAL_PER_MIN) is not None:
        raise err(429, "rate_limited", "Too many demo sessions right now. Try again in a minute.")
    sid = uuid.uuid4().hex[:12]
    uid = f"demo-{sid}"
    expire = datetime.now(timezone.utc) + timedelta(hours=float(os.environ.get("DEMO_TTL_HOURS", "24")))
    store.set_many(clone_rows(store, uid, expire, sid))
    token, mode = minter(uid)
    return DemoSession(uid=uid, custom_token=token, auth_mode=mode, expires_at=expire.isoformat(), merchant_id=f"m_{uid}", gift_id=f"{uid}-gift")
