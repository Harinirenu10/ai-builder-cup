"""Builds the static "Demo Threads Co." mock storefront (fictional brand, simulated products, invented reviews).

    python web/store/build_store.py                 # writes the pages next to this file (committed output)
    python web/store/build_store.py --out DIR --api http://127.0.0.1:8000 --app http://127.0.0.1:5173

One source of truth (PRODUCTS below) feeds the size chart on each page, the JSON handed to the fit widget and
truth.json, which the reader tests compare against.
"""

import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERCHANT_KEY = "pk_demo_threads"

# column: (label printed on the page, zone, {size: value or "lo-hi"})
PRODUCTS = [
    {
        "slug": "kurta", "title": "Jaipur Block-Print Rayon Kurta", "price": "$58.00", "color": "#c2410c",
        "category": "kurta", "fit_type": "regular", "stretch_class": "none", "pre_shrunk": False,
        "fabric": [("rayon", 100)], "fabric_text": "100% rayon (viscose)",
        "desc": "A straight-cut kurta with a hand block-printed motif and side slits.",
        "details": ["Straight cut, regular fit", "Not pre-shrunk: may shrink up to 3% on the first wash", "Side slits, mandarin collar"],
        "chart": {"title": "Size guide: body measurements", "note": "Find your size. The ranges show the body each size is made to fit, in inches.",
                  "type": "body", "unit": "in", "basis": "full", "evidence": "To fit"},
        "columns": [("To fit bust (in)", "chest", {"S": "34-35", "M": "36-37", "L": "38-39", "XL": "40-41", "XXL": "42-43"}),
                    ("To fit waist (in)", "waist", {"S": "28-29", "M": "30-31", "L": "32-33", "XL": "34-35", "XXL": "36-37"}),
                    ("To fit hip (in)", "hip", {"S": "36-37", "M": "38-39", "L": "40-41", "XL": "42-43", "XXL": "44-45"})],
        "reviews": [("Meera", 4, "Beautiful print. It runs small at the chest, so I ordered one size up."),
                    ("Anjali", 3, "Lovely fabric but too short for my long torso, it sits above my hip."),
                    ("Divya", 4, "It shrank after the first wash, about an inch in length. Wash cold."),
                    ("Kavya", 5, "True to size at the waist and the sleeves are a good length."),
                    ("Rhea", 4, "The colour is a little duller than in the photos."),
                    ("Sana", 3, "Runs small at the chest and shrank after the first wash. Size up.")],
        "review_counts": {"chest/runs_small": 2, "garment_length/short": 1, "shrinkage": 2, "color_mismatch": 1},
    },
    {
        "slug": "shirt", "title": "Oxford Button-Down Shirt", "price": "$64.00", "color": "#1d4ed8",
        "category": "shirt", "fit_type": "regular", "stretch_class": "none", "pre_shrunk": True,
        "fabric": [("cotton", 100)], "fabric_text": "100% cotton oxford",
        "desc": "A classic button-down in washed cotton oxford with a chest pocket.",
        "details": ["Regular fit", "Sanforized: pre-shrunk", "Button-down collar, chest pocket"],
        "chart": {"title": "Size guide: garment measurements", "note": "Measurements of the garment laid flat, full circumference, in cm.",
                  "type": "garment", "unit": "cm", "basis": "full", "evidence": "Garment"},
        "columns": [("Chest (cm)", "chest", {"S": 108, "M": 114, "L": 120, "XL": 126}),
                    ("Shoulder (cm)", "shoulder_width", {"S": 44, "M": 46, "L": 48, "XL": 50}),
                    ("Sleeve length (cm)", "sleeve_length", {"S": 60, "M": 61.5, "L": 63, "XL": 64.5}),
                    ("Length (cm)", "garment_length", {"S": 72, "M": 74, "L": 76, "XL": 78})],
        "reviews": [("Arjun", 4, "Good shirt but it runs small at the chest, I needed a size up."),
                    ("Mohan", 4, "Chest felt tight through the arms, sized up and it is fine."),
                    ("Leo", 5, "Pre-shrunk really means it: washed it three times and nothing changed."),
                    ("Priya", 4, "Sleeves are a bit long for me, I had them shortened."),
                    ("Ben", 3, "The fabric is a little see-through in the white.")],
        "review_counts": {"chest/runs_small_or_tight": 2, "sleeve_length/long": 1, "sheerness": 1, "shrinkage": 0},
    },
    {
        "slug": "trousers", "title": "Pleated Tapered Chinos", "price": "$79.00", "color": "#78716c",
        "category": "trousers", "fit_type": "regular", "stretch_class": "low", "pre_shrunk": None,
        "fabric": [("cotton", 98), ("elastane", 2)], "fabric_text": "98% cotton, 2% elastane",
        "desc": "Single-pleat chinos with a tapered leg and a little stretch.",
        "details": ["Regular fit through the thigh, tapered leg", "Low stretch from 2% elastane", "Side pockets, belt loops"],
        "chart": {"title": "Size guide: garment measurements", "note": "Measurements of the garment laid flat, full circumference, in cm.",
                  "type": "garment", "unit": "cm", "basis": "full", "evidence": "Garment"},
        "columns": [("Waist (cm)", "waist", {"30": 84, "32": 88, "34": 92, "36": 96}),
                    ("Hip (cm)", "hip", {"30": 104, "32": 108, "34": 112, "36": 116}),
                    ("Thigh (cm)", "thigh", {"30": 62, "32": 64, "34": 66, "36": 68}),
                    ("Inseam (cm)", "inseam", {"30": 80, "32": 80, "34": 80, "36": 80}),
                    ("Back rise (cm)", "back_rise", {"30": 27, "32": 27.5, "34": 28, "36": 28.5})],
        "reviews": [("Dev", 4, "Comfortable, but the waist runs small. The 32 felt tight so I took a 34."),
                    ("Rahul", 4, "Waist runs tight, size up if you are between sizes."),
                    ("Chris", 3, "The inseam is long for me at 5'6. Had to hem them."),
                    ("Nina", 5, "Love the taper. Fits like the chart says."),
                    ("Omar", 4, "Inseam is long, plan on hemming.")],
        "review_counts": {"waist/tight_or_runs_small": 2, "inseam/long": 2, "shrinkage": 0},
    },
    {
        "slug": "tshirt", "title": "Heavyweight Boxy Crewneck Tee", "price": "$48.00", "color": "#0f766e",
        "category": "tshirt", "fit_type": "relaxed", "stretch_class": "none", "pre_shrunk": True,
        "fabric": [("cotton", 100)], "fabric_text": "100% organic cotton, 240 GSM",
        "desc": "A boxy, dropped-shoulder crewneck tee in heavyweight organic cotton.",
        "details": ["Boxy relaxed fit", "Garment-dyed and enzyme washed, pre-shrunk", "No stretch"],
        "chart": {"title": "Size guide: garment measurements", "note": "Measurements of the garment laid flat, shown in cm (full chest circumference).",
                  "type": "garment", "unit": "cm", "basis": "full", "evidence": "Garment"},
        "columns": [("Chest (cm)", "chest", {"S": 112, "M": 118, "L": 124, "XL": 130}),
                    ("Length (cm)", "garment_length", {"S": 70, "M": 72, "L": 74, "XL": 76}),
                    ("Shoulder (cm)", "shoulder_width", {"S": 48, "M": 50, "L": 52, "XL": 54})],
        "reviews": [("Arjun", 5, "Great weight and drape. The chest runs roomy, which I like."),
                    ("Mia", 4, "Runs large in the chest. I sized down to S and it is perfect."),
                    ("Karthik", 3, "I'm 6'1 with a long torso and the tee is too short on me, it sits above the belt."),
                    ("Leah", 4, "Colour is a bit more grey than the photos show."),
                    ("Sam", 5, "Washed it twice, no shrinking at all.")],
        "review_counts": {"chest/runs_large": 2, "garment_length/short": 1, "color_mismatch": 1, "shrinkage": 0},
    },
    {
        "slug": "jacket", "title": "Field Canvas Utility Jacket", "price": "$95.00", "color": "#4d7c0f",
        "category": "jacket", "fit_type": "regular", "stretch_class": "none", "pre_shrunk": None,
        "fabric": [("cotton", 100)], "fabric_text": "100% cotton duck canvas",
        "desc": "An unlined utility jacket in rigid cotton duck canvas with four pockets.",
        "details": ["Regular fit, room to layer", "Unlined rigid weave", "Four patch pockets"],
        "chart": {"title": "Size guide: garment measurements", "note": "Garment laid flat. Chest is measured flat across (half the circumference). All values in cm.",
                  "type": "garment", "unit": "cm", "basis": "flat", "evidence": "Garment"},
        "columns": [("Chest (flat, across) (cm)", "chest", {"S": 55, "M": 58, "L": 61, "XL": 64}),
                    ("Sleeve length (cm)", "sleeve_length", {"S": 63, "M": 64, "L": 65, "XL": 66}),
                    ("Length (cm)", "garment_length", {"S": 70, "M": 72, "L": 74, "XL": 76}),
                    ("Across shoulder (cm)", "shoulder_width", {"S": 46, "M": 48, "L": 50, "XL": 52})],
        "reviews": [("Vik", 4, "Solid canvas. The sleeves are short on me, I have long arms."),
                    ("Anita", 4, "Sleeves a touch short, otherwise lovely."),
                    ("Greg", 3, "Chest is tight with a hoodie underneath. Size up if you layer."),
                    ("Sara", 4, "Colour is more olive than the khaki in the photos."),
                    ("Luis", 5, "Stiff at first, softens with wear. The chest is tight over a sweater, I ordered L.")],
        "review_counts": {"sleeve_length/short": 2, "chest/tight": 2, "color_mismatch": 1, "shrinkage": 0},
    },
    {
        "slug": "blouse", "title": "Silk-Blend Wrap Blouse", "price": "$72.00", "color": "#9d174d",
        "category": "blouse", "fit_type": "regular", "stretch_class": "none", "pre_shrunk": None,
        "fabric": [("viscose", 60), ("silk", 40)], "fabric_text": "60% viscose, 40% silk",
        "desc": "A fluid wrap blouse with three-quarter sleeves in a viscose and silk blend.",
        "details": ["Regular fit, wrap front", "Three-quarter sleeves", "Dry clean or gentle hand wash"],
        "chart": {"title": "Size guide: body measurements", "note": "Find your size. The values show the body each size is made to fit, in cm.",
                  "type": "body", "unit": "cm", "basis": "full", "evidence": "To fit"},
        "columns": [("To fit bust (cm)", "chest", {"XS": 82, "S": 86, "M": 90, "L": 94, "XL": 98}),
                    ("To fit waist (cm)", "waist", {"XS": 64, "S": 68, "M": 72, "L": 76, "XL": 80}),
                    ("To fit shoulder (cm)", "shoulder_width", {"XS": 36, "S": 37, "M": 38, "L": 39, "XL": 40})],
        "reviews": [("Isha", 4, "Lovely drape. It is tight across the bust in my usual size, so I sized up."),
                    ("Tara", 3, "The white is quite see-through, wear a camisole."),
                    ("Zoya", 4, "Runs small at the chest. The sleeves fall at the right place."),
                    ("Neha", 5, "Beautiful colour, exactly as pictured."),
                    ("Lily", 3, "The blouse is short for me at 5'10, the hem rides up.")],
        "review_counts": {"chest/tight_or_runs_small": 2, "sheerness": 1, "garment_length/short": 1},
    },
]

SHAPES = {  # simple garment silhouettes on a 600x700 canvas
    "kurta": "M230 110 L300 140 L370 110 L470 170 L440 260 L410 235 L420 600 L180 600 L190 235 L160 260 L130 170 Z",
    "shirt": "M220 110 L300 150 L380 110 L480 160 L450 300 L410 270 L415 560 L185 560 L190 270 L150 300 L120 160 Z",
    "trousers": "M190 100 L410 100 L440 600 L330 600 L300 300 L270 600 L160 600 Z",
    "tshirt": "M215 110 L300 145 L385 110 L500 190 L450 270 L410 240 L415 540 L185 540 L190 240 L150 270 L100 190 Z",
    "jacket": "M210 100 L300 135 L390 100 L500 170 L470 480 L420 470 L425 560 L175 560 L180 470 L130 480 L100 170 Z",
    "blouse": "M230 110 L300 190 L370 110 L470 170 L440 290 L405 270 L410 480 L190 480 L195 270 L160 290 L130 170 Z",
}


def svg(p: dict, view: str) -> str:
    neck = '<path d="M265 112 Q300 170 335 112" fill="none" stroke="#fff" stroke-width="5"/>' if view == "front" else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 700" width="600" height="700" role="img" aria-label="{html.escape(p["title"])} ({view})">'
            f'<rect width="600" height="700" fill="#f1f5f9"/><path d="{SHAPES[p["slug"]]}" fill="{p["color"]}" stroke="#0f172a" stroke-width="3"/>{neck}'
            f'<text x="300" y="660" font-family="Arial" font-size="20" text-anchor="middle" fill="#475569">Simulated image: {view}</text></svg>')


def mid(v) -> float:
    if isinstance(v, str) and "-" in v:
        lo, hi = v.split("-")
        return (float(lo) + float(hi)) / 2
    return float(v)


def fit_product(p: dict) -> dict:
    """What a merchant would hand to the widget: the chart in its own unit, category, fit, fabric."""
    sizes: dict[str, dict[str, float]] = {}
    for _, zone, vals in p["columns"]:
        for size, v in vals.items():
            sizes.setdefault(size, {})[zone] = mid(v) * (2 if p["chart"]["basis"] == "flat" and zone in ("chest", "waist", "hip", "thigh") else 1)
    out = {"size_chart": {"sizes": sizes, "chart_type": p["chart"]["type"], "size_system": "ALPHA" if not p["slug"] == "trousers" else "US_NUMERIC", "unit": p["chart"]["unit"]},
           "category": p["category"], "fit_type": p["fit_type"], "fabric": [{"fiber": f, "percent": pc} for f, pc in p["fabric"]],
           "stretch_class": p["stretch_class"], "ref": f"dtc-{p['slug']}"}
    if p["pre_shrunk"] is not None:
        out["pre_shrunk"] = p["pre_shrunk"]
    return out


def chart_table(p: dict) -> str:
    sizes = list(p["columns"][0][2])
    head = "<tr><th>Size</th>" + "".join(f"<th>{html.escape(c[0])}</th>" for c in p["columns"]) + "</tr>"
    rows = "".join("<tr><td>" + s + "</td>" + "".join(f"<td>{c[2][s]}</td>" for c in p["columns"]) + "</tr>" for s in sizes)
    return f"<table>{head}{rows}</table>"


def page(p: dict, api: str, app: str, widget_src: str) -> str:
    sizes = list(p["columns"][0][2])
    ld = {"@context": "https://schema.org", "@type": "Product", "name": p["title"], "brand": {"@type": "Brand", "name": "Demo Threads Co."},
          "image": [f"img/{p['slug']}-front.svg", f"img/{p['slug']}-back.svg"], "description": p["desc"], "material": p["fabric_text"],
          "offers": {"@type": "Offer", "priceCurrency": "USD", "price": p["price"].strip("$")},
          "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.2", "reviewCount": str(len(p["reviews"]))}}
    reviews = "".join(f'<div class="review"><b>{html.escape(a)}</b> <span aria-label="{s} stars">{"★" * s}{"☆" * (5 - s)}</span><p>{html.escape(t)}</p></div>' for a, s, t in p["reviews"])
    opts = "".join(f'<option value="{s}">{s}</option>' for s in sizes)
    widget_json = html.escape(json.dumps(fit_product(p), separators=(",", ":")), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(p['title'])} | Demo Threads Co.</title>
<meta property="og:image" content="img/{p['slug']}-front.svg">
<script type="application/ld+json">{json.dumps(ld)}</script>
<link rel="stylesheet" href="store.css"></head>
<body data-product="{p['slug']}" data-title="{html.escape(p['title'])}" data-price="{p['price'].strip('$')}">
<header class="top"><a class="brand" href="index.html">Demo Threads Co.</a><nav><a href="index.html">Shop</a><a href="cart.html">Bag (<span id="cart-count">0</span>)</a><a href="returns.html">Returns</a></nav></header>
<main>
<div class="sim">SIMULATED storefront for testing. Fictional brand, invented products and reviews.</div>
<div class="grid">
<div class="gallery"><img src="img/{p['slug']}-front.svg" alt="{html.escape(p['title'])}, front" width="600" height="700"><img src="img/{p['slug']}-back.svg" alt="{html.escape(p['title'])}, back" width="600" height="700"></div>
<div class="info">
<h1>{html.escape(p['title'])}</h1><p class="price">{p['price']}</p><p>{html.escape(p['desc'])}</p>
<h2>Details</h2><ul>{''.join(f'<li>{html.escape(d)}</li>' for d in p['details'])}</ul>
<p><b>Fabric:</b> {html.escape(p['fabric_text'])}</p>
<div class="row"><label for="size">Size</label><select id="size">{opts}</select>
<label for="qty">Qty</label><input id="qty" type="number" min="1" max="5" value="1">
<button type="button" id="size-guide-btn">Size guide</button></div>
<div class="widget-slot" data-fit-product='{widget_json}'></div>
<div class="row"><button type="button" id="add-to-bag" class="primary">Add to bag</button></div>
<p id="bag-note" role="status" aria-live="polite"></p>
</div></div>
<div id="chart-modal" class="modal" role="dialog" aria-modal="true" aria-labelledby="chart-title" hidden>
<button type="button" id="chart-close">Close</button><h2 id="chart-title">{html.escape(p['chart']['title'])}</h2><p>{html.escape(p['chart']['note'])}</p>{chart_table(p)}</div>
<section><h2>Customer reviews ({len(p['reviews'])})</h2>{reviews}</section>
</main>
<footer>Demo Threads Co. is a fictional brand used to demonstrate Measured Fit. Nothing here is for sale.</footer>
<script src="{html.escape(widget_src)}" data-merchant="{MERCHANT_KEY}" data-api="{html.escape(api)}" data-app="{html.escape(app)}" defer></script>
<script src="store.js" defer></script>
</body></html>
"""


def index_page(api: str, app: str) -> str:
    cards = "".join(f'<a class="card" href="{p["slug"]}.html"><img src="img/{p["slug"]}-front.svg" alt="{html.escape(p["title"])}" width="300" height="350"><h3>{html.escape(p["title"])}</h3><p>{p["price"]}</p></a>' for p in PRODUCTS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Demo Threads Co.</title><link rel="stylesheet" href="store.css"></head>
<body><header class="top"><a class="brand" href="index.html">Demo Threads Co.</a><nav><a href="index.html">Shop</a><a href="cart.html">Bag (<span id="cart-count">0</span>)</a></nav></header>
<main><div class="sim">SIMULATED storefront for testing. Fictional brand, invented products and reviews.</div><h1>New this season</h1><div class="cards">{cards}</div></main>
<footer>Demo Threads Co. is a fictional brand used to demonstrate Measured Fit.</footer><script src="store.js" defer></script></body></html>
"""


def _shell(title: str, body: str, api: str, app: str, widget_src: str, scripts: str = "") -> str:
    """Pages that need the widget's config and consent (cart, checkout, returns) but show no size finder."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} | Demo Threads Co.</title><link rel="stylesheet" href="store.css"></head>
<body><header class="top"><a class="brand" href="index.html">Demo Threads Co.</a><nav><a href="index.html">Shop</a><a href="cart.html">Bag (<span id="cart-count">0</span>)</a><a href="returns.html">Returns</a></nav></header>
<main><div class="sim">SIMULATED storefront for testing. Nothing here is for sale.</div>{body}</main>
<footer>Demo Threads Co. is a fictional brand used to demonstrate Measured Fit.</footer>
<script src="{html.escape(widget_src)}" data-merchant="{MERCHANT_KEY}" data-api="{html.escape(api)}" data-app="{html.escape(app)}" defer></script>
<script src="store.js" defer></script><script src="store-orders.js" defer></script>{scripts}</body></html>
"""


def cart_page(api: str, app: str, widget_src: str) -> str:
    return _shell("Your bag", """<h1>Your bag</h1><div id="nudge" aria-live="polite"></div><div id="cart-lines"></div><p id="cart-total"></p>
<p id="cart-two-sizes" role="status"></p><p><a id="to-checkout" class="button" href="checkout.html">Checkout</a></p>""", api, app, widget_src)


def checkout_page(api: str, app: str, widget_src: str) -> str:
    return _shell("Checkout", '<h1>Checkout</h1><div id="checkout-root"></div>', api, app, widget_src)


def returns_page(api: str, app: str, widget_src: str) -> str:
    return _shell("Returns", """<h1>Return or exchange</h1><p>Tell us what happened. A person at the store reviews anything the checks cannot settle, and nothing is declined automatically.</p>
<div id="returns-root"></div>""", api, app, widget_src)


def truth(p: dict) -> dict:
    return {"source": "generated from build_store.py (the values printed on the page)", "category": p["category"], "chart_type": p["chart"]["type"],
            "unit": p["chart"]["unit"], "circumference_basis": p["chart"]["basis"], "pre_shrunk": p["pre_shrunk"], "reviews_read": len(p["reviews"]),
            "sizes": {s: {zone: mid(vals[s]) for _, zone, vals in p["columns"]} for s in p["columns"][0][2]}, "review_counts": p["review_counts"]}


def build(out: Path, api: str = "/api", app: str = "", widget_src: str | None = None) -> None:
    """Writes the store. Built in place it uses ../widget/fit-widget.js (as hosted); built elsewhere it copies the widget alongside."""
    here = out.resolve() == HERE
    widget_src = widget_src or ("../widget/fit-widget.js" if here else "widget/fit-widget.js")
    (out / "img").mkdir(parents=True, exist_ok=True)
    if not here and widget_src == "widget/fit-widget.js":
        (out / "widget").mkdir(exist_ok=True)
        (out / "widget" / "fit-widget.js").write_bytes((HERE.parent / "widget" / "fit-widget.js").read_bytes())
    for p in PRODUCTS:
        (out / f"{p['slug']}.html").write_text(page(p, api, app, widget_src), encoding="utf-8")
        for view in ("front", "back"):
            (out / "img" / f"{p['slug']}-{view}.svg").write_text(svg(p, view), encoding="utf-8")
    if out.resolve() != HERE:  # building somewhere else (tests, a deploy folder): bring the static assets along
        for name in ("store.css", "store.js", "store-orders.js"):
            (out / name).write_bytes((HERE / name).read_bytes())
    (out / "index.html").write_text(index_page(api, app), encoding="utf-8")
    (out / "cart.html").write_text(cart_page(api, app, widget_src), encoding="utf-8")
    (out / "checkout.html").write_text(checkout_page(api, app, widget_src), encoding="utf-8")
    (out / "returns.html").write_text(returns_page(api, app, widget_src), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps({f"store_{p['slug']}": truth(p) for p in PRODUCTS}, indent=2), encoding="utf-8")
    (out / "products.json").write_text(json.dumps({p["slug"]: fit_product(p) for p in PRODUCTS}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--api", default="/api", help="API base the widget calls (Firebase Hosting rewrites /api/** to Cloud Run)")
    ap.add_argument("--widget", default=None, help="Path of fit-widget.js as seen from the store pages (default: ../widget/... in place, widget/... elsewhere)")
    ap.add_argument("--app", default="", help="Origin of the customer app that hosts /app/connect; empty means the store's own origin")
    a = ap.parse_args()
    build(Path(a.out), a.api, a.app, a.widget)
    print(f"Store written to {a.out}")
