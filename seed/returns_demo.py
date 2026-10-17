"""Seed the demo returns desk: four returns, one per routing branch, built with the same code the API uses.

    python seed/returns_demo.py                     # into the local API store (services/api/.data); also writes overlay SVGs
    python seed/returns_demo.py --origin http://localhost:8081

Everything is SIMULATED: the shopper, the twin (samples/twin.json), the unboxing measurements and the overlay images.
Branches: 1 label mismatch (shirt), 2 model miss (t-shirt), 3 wrong size received (jacket), 4 no guarantee, standard policy (trousers).
Prompt 12's demo reuses these. Staff sign-in for the desk: user id demo-staff (dev mode).
"""

import argparse
import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "web" / "store"))

import build_store  # noqa: E402
from apilib.fitquery import evaluate_for_uid  # noqa: E402
from apilib.models import FitQueryProduct  # noqa: E402
from apilib.orders import OrderItemIn, create_order  # noqa: E402
from apilib.returns import ReturnIn, create_return  # noqa: E402
from apilib.stores import Store, default_store  # noqa: E402

UID = "demo-shopper"
STAFF = "demo-staff"
MERCHANT_ID = "m_demo_threads"
DEFAULT_ORIGINS = ["http://localhost:8081", "http://127.0.0.1:8081", "http://localhost:5000"]
PRODUCTS = json.loads((ROOT / "web" / "store" / "products.json").read_text(encoding="utf-8"))
TITLES = {p["slug"]: p["title"] for p in build_store.PRODUCTS}


def overlay_svg(slug: str, title: str, size: str, measured: dict[str, float]) -> str:
    shape = build_store.SHAPES[slug]
    color = next(p["color"] for p in build_store.PRODUCTS if p["slug"] == slug)
    lines = "".join(f'<text x="20" y="{70 + 24 * i}" font-family="Arial" font-size="16" fill="#0f172a">{html.escape(z.replace("_", " "))}: {v:.1f} cm</text>'
                    for i, (z, v) in enumerate(measured.items()))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 700" width="600" height="700" role="img" aria-label="Unboxing measurement overlay, {html.escape(title)}">'
            f'<rect width="600" height="700" fill="#f1f5f9"/><path d="{shape}" fill="{color}" fill-opacity="0.55" stroke="#0f172a" stroke-width="3"/>'
            f'<text x="20" y="36" font-family="Arial" font-size="18" font-weight="bold" fill="#0f172a">{html.escape(title)}, labelled {html.escape(size)}</text>'
            f'{lines}<line x1="150" y1="320" x2="450" y2="320" stroke="#dc2626" stroke-width="3" stroke-dasharray="8 6"/>'
            f'<text x="300" y="660" font-family="Arial" font-size="18" text-anchor="middle" fill="#92400e">SIMULATED unboxing overlay</text></svg>')


def seed_returns(store: Store, origins: list[str], overlay_dir: Path | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    store.set("twins", UID, {"owner_id": UID, "simulated": True, "measurements": {"zones": twin["zones"]}})
    store.set("merchants", MERCHANT_ID, {"name": "Demo Threads Co.", "publishable_key": build_store.MERCHANT_KEY, "allowed_origins": origins,
                                         "staff_uids": [STAFF], "return_policy": "refund", "return_window_days": 30})
    merchant = store.get("merchants", MERCHANT_ID)

    def item(slug: str, offset: int = 0) -> tuple[OrderItemIn, str | None]:
        product = FitQueryProduct.model_validate(PRODUCTS[slug])
        rec = evaluate_for_uid(store, UID, product, "Demo Threads Co.").recommended_size
        sizes = list(product.size_chart.sizes)
        chosen = sizes[min(max(sizes.index(rec) + offset, 0), len(sizes) - 1)] if rec else sizes[0]
        return OrderItemIn(sku=PRODUCTS[slug]["ref"], title=TITLES[slug], size=chosen, product=product), rec

    def measure(slug: str, size: str, zones: tuple[str, ...], shift: float = 0.0) -> dict[str, float]:
        row = PRODUCTS[slug]["size_chart"]["sizes"][size]
        return {z: round(row[z] + shift, 1) for z in zones if z in row}

    # slug, order size offset from the recommended size, order id, return id, reason, zones measured,
    # which size the garment really measures like (offset from the label), days since the order
    cases = [
        ("shirt", 0, "demo-order-1-shirt", "demo-return-1-label", "too_small", ("chest", "garment_length", "sleeve_length"), -1, 4),
        ("tshirt", 0, "demo-order-2-tshirt", "demo-return-2-model-miss", "too_small", ("chest", "garment_length"), 0, 6),
        ("jacket", 0, "demo-order-3-jacket", "demo-return-3-wrong-size", "wrong_size_received", ("chest", "sleeve_length", "garment_length"), 0, 3),
        ("trousers", 1, "demo-order-4-trousers", "demo-return-4-policy", "too_big", ("waist", "hip", "inseam"), 0, 8),
    ]
    summary = []
    for slug, off, oid, rid, reason, zones, real_off, days in cases:
        it, rec = item(slug, offset=off)
        sizes = list(PRODUCTS[slug]["size_chart"]["sizes"])
        label_idx = sizes.index(it.size)
        if reason == "too_small" and label_idx == len(sizes) - 1:
            reason = "too_big"                         # nothing larger to exchange to: report the other direction
        if label_idx + real_off < 0:
            real_off = 1                               # nothing smaller to measure like: measure like the next size up instead
        created = now - timedelta(days=days)
        _, priced = create_order(store, merchant, UID, [it], created_at=created, order_id=oid)
        real_size = sizes[min(label_idx + real_off, len(sizes) - 1)]
        measured = measure(slug, real_size, zones, shift=0.3 if real_off == 0 else 0.0)   # a matching garment is never exact: +0.3 cm
        ub_id = f"demo-unboxing-{slug}"
        store.set("unboxing_measurements", ub_id, {"order_id": oid, "sku": it.sku, "size_label": it.size, "measured_cm": measured,
                                                   "overlay_image_url": f"/store/img/unboxing/{ub_id}.svg", "source": "simulated", "created_at": created.isoformat()})
        if overlay_dir:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            (overlay_dir / f"{ub_id}.svg").write_text(overlay_svg(slug, TITLES[slug], it.size, measured), encoding="utf-8")
        r = create_return(store, merchant, store.get("orders", oid),
                          ReturnIn(order_id=oid, item_index=0, reason=reason, unboxing_measurement_id=ub_id, note="Simulated demo return."),
                          now=created + timedelta(days=2), return_id=rid)
        summary.append({"return": rid, "product": slug, "ordered": it.size, "recommended": rec, "guarantee": priced[0]["guarantee"], "reason": reason,
                        "route": r["routing"]["route"], "status": r["status"]})

    # two orders without returns, so the guaranteed share is not trivial
    for n, (slug, off) in enumerate((("kurta", 0), ("blouse", 1)), start=5):
        it, _ = item(slug, offset=off)
        create_order(store, merchant, UID, [it], created_at=now - timedelta(days=n), order_id=f"demo-order-{n}-{slug}")
    # simulated nudge events for the KPI strip
    for i, t in enumerate(["nudge_shown"] * 5 + ["nudge_accepted"] * 3 + ["nudge_dismissed"]):
        store.set("events", f"demo-event-{i}", {"merchant_id": MERCHANT_ID, "type": t, "product_ref": "dtc-shirt", "ts": now.isoformat(), "source": "seed"})
    return {"merchant": MERCHANT_ID, "staff": STAFF, "returns": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", action="append", help="allowed store origin, repeatable")
    a = ap.parse_args()
    out = seed_returns(default_store(), a.origin or DEFAULT_ORIGINS, overlay_dir=ROOT / "web" / "store" / "img" / "unboxing")
    for r in out["returns"]:
        print(f"{r['return']:<28} {r['product']:<9} ordered {r['ordered']:<3} guarantee={str(r['guarantee']):<5} {r['reason']:<20} -> {r['route']} / {r['status']}")
    print(f"\nSeeded (simulated). Desk sign-in: user id {out['staff']} in dev mode.")
