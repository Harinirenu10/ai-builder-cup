"""Seed the demo: shared store data and the template persona that POST /v1/demo/session clones for each judge.

    python seed/seed.py                                   # local API store (services/api/.data), GLBs into web/public/demo/glb
    python seed/seed.py --origin https://PROJECT.web.app --store-base https://PROJECT.web.app/store
    API_STORE=firestore GOOGLE_CLOUD_PROJECT=... python seed/seed.py --origin https://PROJECT.web.app ...

Everything is SIMULATED and labelled so: the persona's twin (a stand-in mannequin from samples/twin.json, not SMPL and not a
real person), the closet, the calendar, the orders, the returns and the unboxing measurements. Product records come from the
reader's recorded Gemini responses for the six Demo Threads Co. pages (no live model call), run through the same normaliser.
Fit results, heatmap GLBs and pose deltas are computed here once, so no demo flow waits on the geometry service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("packages/fitcore", "services/api", "web/store", "seed"):
    sys.path.insert(0, str(ROOT / sub))

import build_store  # noqa: E402
import glb  # noqa: E402
import persona  # noqa: E402
from apilib.fitquery import evaluate_for_uid  # noqa: E402
from apilib.models import FitQueryProduct  # noqa: E402
from apilib.orders import OrderItemIn, create_order  # noqa: E402
from apilib.returns import ReturnIn, create_return  # noqa: E402
from apilib.stores import Store, default_store  # noqa: E402
from fitcore.fit import evaluate  # noqa: E402
from fitcore.fit.reviews import apply as apply_reviews, from_review_signals  # noqa: E402
from fitcore.fit.verdict import business_verdict, zone_verdicts  # noqa: E402
from fitcore.read.models import PageExtraction  # noqa: E402
from fitcore.read.normalize import assemble  # noqa: E402
from fitcore.schemas import Measurements, PoseDelta, PoseDeltas  # noqa: E402

PERSONA = persona.PERSONA_UID
MERCHANT_ID = "m_demo_threads"
STAFF = "demo-staff"
DEFAULT_ORIGINS = ["http://localhost:8081", "http://127.0.0.1:8081", "http://localhost:5000", "http://localhost:5173"]
FIXTURES = ROOT / "services" / "reader" / "tests" / "fixtures" / "llm"
PRODUCTS = json.loads((ROOT / "web" / "store" / "products.json").read_text(encoding="utf-8"))
STORE_PRODUCTS = {p["slug"]: p for p in build_store.PRODUCTS}
GLB_URL = "/demo/glb"
SIM_NOTE = "SIMULATED demo data, not a real person"


def pose_deltas() -> PoseDeltas:
    """Precomputed pose deltas for the persona (from samples/poses_seated.json, plus a short stand-in for reaching forward)."""
    seated = json.loads((ROOT / "samples" / "poses_seated.json").read_text(encoding="utf-8"))["poses"]["seated"]
    return PoseDeltas(poses={"seated": PoseDelta(**seated), "arms_forward": PoseDelta(minutes=10, delta_cm={"upper_back": 2.0, "chest": 1.0})})  # type: ignore[arg-type]


def twin_doc() -> dict:
    return {"owner_id": PERSONA, "simulated": True, "note": SIM_NOTE, "body_model": "stand-in mannequin, not SMPL", "betas": None,
            "height_cm": persona.HEIGHT_CM, "gender": "neutral", "measurements": {"zones": persona.twin_zones()},
            "pose_deltas": pose_deltas().model_dump(mode="json"), "glb_url": f"{GLB_URL}/twin.glb", "created_at": datetime.now(timezone.utc).isoformat()}


def read_products(store_base: str) -> list[dict]:
    """Pre-read Product records via the reader's normaliser, from the recorded responses."""
    out = []
    for slug, p in STORE_PRODUCTS.items():
        ref = PRODUCTS[slug]["ref"]
        url = f"{store_base.rstrip('/')}/{slug}.html"
        ext = PageExtraction.model_validate_json((FIXTURES / f"reader_store_{slug}.json").read_text(encoding="utf-8"))
        outcome = assemble(ext, url=url, product_id=ref, fallback_brand="Demo Threads Co.", image_urls=[f"/store/img/{slug}-front.svg"])
        assert outcome.status == "ok" and outcome.product and outcome.product.size_chart, slug
        out.append({"slug": slug, "ref": ref, "url": url, "title": p["title"], "price": p["price"], "product": outcome.product, "reviews": outcome.reviews,
                    "flags": outcome.flags})
    return out


def seed_shared(store: Store, origins: list[str], items: list[dict], now: datetime) -> None:
    store.set("merchants", MERCHANT_ID, {"name": "Demo Threads Co.", "publishable_key": build_store.MERCHANT_KEY, "allowed_origins": origins,
                                         "staff_uids": [STAFF], "return_policy": "refund", "return_window_days": 30, "simulated": True})
    rows = []
    for it in items:
        read = {"status": "ok", "url": it["url"], "product": it["product"].model_dump(mode="json"), "reviews": it["reviews"].model_dump(mode="json") if it["reviews"] else None,
                "flags": it["flags"], "cached": False}
        rows.append(("products", it["ref"], {"ref": it["ref"], "slug": it["slug"], "title": it["title"], "price": it["price"], "page_url": it["url"],
                                            "image": f"/store/img/{it['slug']}-front.svg", "product": read["product"], "reviews": read["reviews"],
                                            "flags": it["flags"], "simulated": True}))
        # the same record in the reader's cache shape, so the chart studio finds the "old chart" and reviews for the URL
        rows.append(("products", hashlib.sha256(it["url"].encode()).hexdigest(), {"response": read, "url": it["url"], "fetched_at": now.isoformat()}))
        if it["product"].size_chart.chart_type == "garment":   # the studio publishes garment charts only
            sizes = {k: {z.value: v for z, v in row.items()} for k, row in it["product"].size_chart.sizes.items()}
            rows.append(("charts", f"demo-chart-{it['slug']}", {"merchant_id": MERCHANT_ID, "product_ref": it["ref"], "name": it["title"], "category": it["product"].category.value,
                                                                "fit_type": it["product"].fit_type.value, "chart_type": "garment", "sizes": sizes, "source": "edited",
                                                                "version": 1, "status": "published", "live": False, "created_by": STAFF, "created_at": now.isoformat(),
                                                                "grading_flags": [], "confidences": {}, "simulated": True}))
    store.set_many(rows)


def seed_persona(store: Store, items: list[dict], glb_dir: Path | None, now: datetime) -> dict:
    twin = twin_doc()
    zones = {k: v["value_cm"] for k, v in persona.twin_zones().items()}
    body = Measurements.model_validate(twin["measurements"])
    rows: list[tuple[str, str, dict]] = [("twins", PERSONA, twin)]
    for c in persona.closet():
        rows.append(("wardrobe", f"{PERSONA}-{c.id}", {**c.model_dump(mode="json"), "uid": PERSONA, "user_id": PERSONA, "item_id": c.id}))
    for e in persona.event_docs():
        rows.append(("events", f"{PERSONA}-{e['event_id']}", {**e, "uid": PERSONA, "kind": "calendar"}))

    if glb_dir:
        glb.make_twin_glb(zones, persona.HEIGHT_CM, None, glb_dir / "twin.glb")
    fits = []
    for it in items:
        signals = from_review_signals(it["reviews"]) if it["reviews"] else []
        adj, notes = apply_reviews(signals, body) if signals else (None, [])
        result = evaluate(body, it["product"], None, adj, pose_deltas())
        best = result.recommended_size or max(result.sizes, key=lambda s: s.score).size
        verdicts = dict(zone_verdicts(result)) if result.recommended_size else {z.value: v.verdict.value for z, v in result.size_fit(best).zones.items()}  # type: ignore[union-attr]
        verdicts = {z: getattr(v, "value", v) for z, v in verdicts.items()}
        if glb_dir:
            glb.make_twin_glb(zones, persona.HEIGHT_CM, verdicts, glb_dir / f"heat-{it['slug']}.glb")
        fits.append({"slug": it["slug"], "recommended": result.recommended_size, "confidence": round(result.confidence, 2)})
        rows.append(("fits", f"{PERSONA}__{it['ref']}", {
            "uid": PERSONA, "ref": it["ref"], "slug": it["slug"], "title": it["title"], "price": it["price"], "category": it["product"].category.value,
            "recommended_size": result.recommended_size, "shown_size": best, "confidence": round(result.confidence, 2), "verdicts": verdicts,
            "result": result.model_dump(mode="json"), "review_notes": notes, "store_view": business_verdict(result).model_dump(mode="json"),
            "heatmap_glb_url": f"{GLB_URL}/heat-{it['slug']}.glb", "simulated": True, "computed_at": now.isoformat()}))
    store.set_many(rows)
    return {"fits": fits}


def seed_orders(store: Store, now: datetime) -> list[dict]:
    """3 orders, 4 returns, one per routing branch, plus 1 incoming gift request. Same routing code the API uses."""
    merchant = store.get("merchants", MERCHANT_ID)
    ids = lambda n: f"{PERSONA}-order-{n}"  # noqa: E731

    def item(slug: str, offset: int = 0) -> tuple[OrderItemIn, str | None]:
        product = FitQueryProduct.model_validate(PRODUCTS[slug])
        rec = evaluate_for_uid(store, PERSONA, product, "Demo Threads Co.").recommended_size
        sizes = list(product.size_chart.sizes)
        chosen = sizes[min(max(sizes.index(rec) + offset, 0), len(sizes) - 1)] if rec else sizes[0]
        return OrderItemIn(sku=PRODUCTS[slug]["ref"], title=STORE_PRODUCTS[slug]["title"], size=chosen, product=product), rec

    # order number, slugs in the order (offset from the recommended size), days ago
    orders = [(1, [("shirt", 0), ("tshirt", 0)], 6), (2, [("jacket", 0)], 3), (3, [("trousers", 1)], 8)]
    # slug, order number, item index, reason, zones measured, real size offset from the label
    returns = [("shirt", 1, 0, "too_small", ("chest", "garment_length", "sleeve_length"), -1),
               ("tshirt", 1, 1, "too_small", ("chest", "garment_length"), 0),
               ("jacket", 2, 0, "wrong_size_received", ("chest", "sleeve_length", "garment_length"), 0),
               ("trousers", 3, 0, "too_big", ("waist", "hip", "inseam"), 0)]
    placed: dict[int, list[OrderItemIn]] = {}
    for n, entries, days in orders:
        placed[n] = [item(slug, off)[0] for slug, off in entries]
        create_order(store, merchant, PERSONA, placed[n], created_at=now - timedelta(days=days), order_id=ids(n))
    summary = []
    for slug, n, idx, reason, zones, real_off in returns:
        it = placed[n][idx]
        sizes = list(PRODUCTS[slug]["size_chart"]["sizes"])
        label_idx = sizes.index(it.size)
        if reason == "too_small" and label_idx == len(sizes) - 1:
            reason = "too_big"
        if label_idx + real_off < 0:
            real_off = 1
        real_size = sizes[min(label_idx + real_off, len(sizes) - 1)]
        row = PRODUCTS[slug]["size_chart"]["sizes"][real_size]
        measured = {z: round(row[z] + (0.3 if real_off == 0 else 0.0), 1) for z in zones if z in row}
        ub = f"{PERSONA}-unboxing-{slug}"
        created = now - timedelta(days=dict((a, c) for a, _, c in orders)[n])
        store.set("unboxing_measurements", ub, {"order_id": ids(n), "sku": it.sku, "size_label": it.size, "measured_cm": measured,
                                                "overlay_image_url": f"/store/img/unboxing/demo-unboxing-{slug}.svg", "source": "simulated", "created_at": created.isoformat()})
        r = create_return(store, merchant, store.get("orders", ids(n)),
                          ReturnIn(order_id=ids(n), item_index=idx, reason=reason, unboxing_measurement_id=ub, note="Simulated demo return."),
                          now=created + timedelta(days=2), return_id=f"{PERSONA}-return-{slug}")
        summary.append({"product": slug, "reason": reason, "route": r["routing"]["route"], "status": r["status"]})
    # one gift request waiting for the persona (the giver is a stand-in friend nobody can sign in as)
    gift_product = FitQueryProduct.model_validate(PRODUCTS["jacket"]).model_dump(mode="json")
    store.set("gifts", f"{PERSONA}-gift", {"giver_uid": f"{PERSONA}-friend", "recipient_uid": None, "status": "pending", "title": "Field Canvas Utility Jacket",
                                          "note": "Simulated request from a friend. Approve to share only a size and a confidence with them.",
                                          "product": gift_product, "created_at": now.isoformat(), "expires_at": (now + timedelta(days=3650)).isoformat(),
                                          "recommended_size": None, "confidence": None, "demo_template": True, "simulated": True})
    return summary


def seed_all(store: Store, origins: list[str], store_base: str, glb_dir: Path | None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    items = read_products(store_base)
    seed_shared(store, origins, items, now)
    out = seed_persona(store, items, glb_dir, now)
    out["returns"] = seed_orders(store, now)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", action="append", help="allowed store origin (the deployed web origin), repeatable")
    ap.add_argument("--store-base", default="http://localhost:8081", help="where the Demo Threads pages live; becomes each product's page URL")
    ap.add_argument("--glb-dir", default=str(ROOT / "web" / "public" / "demo" / "glb"))
    ap.add_argument("--no-glb", action="store_true")
    a = ap.parse_args()
    res = seed_all(default_store(), a.origin or DEFAULT_ORIGINS, a.store_base, None if a.no_glb else Path(a.glb_dir))
    for f in res["fits"]:
        print(f"fit    {f['slug']:<9} recommended {f['recommended']} confidence {f['confidence']}")
    for r in res["returns"]:
        print(f"return {r['product']:<9} {r['reason']:<20} -> {r['route']} / {r['status']}")
    print("\nSeeded the persona and shared demo data (all simulated).")
