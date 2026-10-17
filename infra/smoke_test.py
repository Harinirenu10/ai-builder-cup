"""Smoke test: runs every judge-path flow against a deployed (or local) URL and fails loudly.

    python infra/smoke_test.py https://PROJECT.web.app            # through Firebase Hosting (/api rewrite)
    python infra/smoke_test.py http://127.0.0.1:8000 --api-prefix ""   # straight at a local API

Needs only the standard library. Each step prints PASS or FAIL with the reason; any FAIL exits 1.
Signing in: with a Firebase custom token the script exchanges it for an ID token through the Identity Toolkit REST API, which needs the
web API key (FIREBASE_API_KEY or --firebase-api-key). A dev-mode API (AUTH_MODE=dev) returns a `dev:` token and needs no key.
It creates real sandboxes (expire after DEMO_TTL_HOURS) and never prints tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class Fail(Exception):
    pass


class Client:
    def __init__(self, base: str, prefix: str, firebase_key: str | None, assets_base: str | None = None):
        self.base, self.prefix, self.firebase_key = base.rstrip("/"), prefix.rstrip("/"), firebase_key
        self.assets_base = (assets_base or base).rstrip("/")

    def call(self, method: str, path: str, body: Any = None, token: str | None = None, headers: dict[str, str] | None = None, timeout: float = 30,
             raw: bool = False) -> Any:
        if path.startswith("/v1"):
            url = f"{self.base}{self.prefix}{path}"          # API call
        elif path.startswith("/api/") or path == "/healthz":
            url = f"{self.base}{path}"
        else:
            url = f"{self.assets_base}{path}"                 # a static file served by Firebase Hosting
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return payload if raw else (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as e:
            raise Fail(f"{method} {path} -> HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise Fail(f"{method} {path} -> {e}") from None

    def sign_in(self, session: dict) -> str:
        """Bearer token for the sandbox user."""
        if session["auth_mode"] == "dev":
            return session["custom_token"]
        if not self.firebase_key:
            raise Fail("the API returned a Firebase custom token; pass --firebase-api-key (or FIREBASE_API_KEY) so the test can exchange it")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={self.firebase_key}"
        req = urllib.request.Request(url, data=json.dumps({"token": session["custom_token"], "returnSecureToken": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["idToken"]
        except urllib.error.HTTPError as e:
            raise Fail(f"custom token exchange failed: HTTP {e.code} {e.read()[:200].decode(errors='replace')}") from None


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def run(base: str, prefix: str, firebase_key: str | None, merchant_key: str, origin: str | None, assets_base: str | None = None) -> int:
    c = Client(base, prefix, firebase_key, assets_base)
    ctx: dict[str, Any] = {}
    origin = origin or base.rstrip("/")

    def health() -> str:
        r = c.call("GET", f"{prefix}/healthz" if prefix else "/healthz")
        expect(r.get("status") == "ok", f"healthz said {r}")
        return "api answers"

    def demo_session() -> str:
        s = c.call("POST", "/v1/demo/session", timeout=60)
        ctx["session"], ctx["tok"] = s, c.sign_in(s)
        ctx["uid"] = s["uid"]
        expect(s["uid"].startswith("demo-") and s["simulated"] is True, f"unexpected session {list(s)}")
        return f"sandbox {s['uid']}"

    def twin_view() -> str:
        t = c.call("GET", "/v1/me/twin", token=ctx["tok"])
        expect(t["simulated"] and t["zones"].get("chest"), "twin has no chest or is not labelled simulated")
        glb = c.call("GET", t["glb_url"], raw=True)
        expect(glb[:4] == b"glTF", f"{t['glb_url']} is not a .glb")
        return f"{len(t['zones'])} zones, glb {len(glb) // 1024} KB"

    def product_check() -> str:
        prods = c.call("GET", "/v1/me/products", token=ctx["tok"])
        expect(len(prods) == 6, f"expected 6 products, got {len(prods)}")
        fit = c.call("GET", "/v1/me/fits/dtc-shirt", token=ctx["tok"])
        expect(fit["recommended_size"] and fit["zones"], "shirt has no recommendation")
        expect("value_cm" not in json.dumps(fit["store_view"]), "store view leaks measurements")
        heat = c.call("GET", fit["heatmap_glb_url"], raw=True)
        expect(heat[:4] == b"glTF", "heatmap is not a .glb")
        return f"shirt size {fit['recommended_size']}, heatmap {len(heat) // 1024} KB"

    def occasion_test() -> str:
        events = c.call("GET", "/v1/me/events", token=ctx["tok"])
        expect(len(events) == 5, f"expected 5 events, got {len(events)}")
        wedding = next(e for e in events if e["title"] == "Wedding reception")
        r = c.call("POST", "/v1/me/occasion-test", {"event_id": wedding["id"]}, token=ctx["tok"])
        slots = {p["slot"] for p in r["outfit"]}
        expect({"top", "bottom", "shoes"} <= slots, f"outfit is missing slots: {slots}")
        return f"wedding outfit: {', '.join(p['name'] for p in r['outfit'])}"

    def unboxing() -> str:
        orders = c.call("GET", "/v1/me/orders", token=ctx["tok"])
        items = [i for o in orders for i in o["items"]]
        expect(len(orders) == 3 and len(items) == 4, f"expected 3 orders and 4 items, got {len(orders)} and {len(items)}")
        expect(all(i["unboxing"] and i["return"] for i in items), "an item has no unboxing measurement or return")
        svg = c.call("GET", items[0]["unboxing"]["overlay_image_url"], raw=True)
        expect(b"<svg" in svg, "unboxing overlay is not an svg")
        return f"{len(items)} items, routes: {sorted({i['return']['route'] for i in items})}"

    def gift() -> str:
        gid = ctx["session"]["gift_id"]
        g = c.call("GET", f"/v1/gifts/{gid}", token=ctx["tok"])
        expect(g["status"] == "pending" and g["can_approve"], f"gift is not approvable: {g}")
        c.call("POST", f"/v1/gifts/{gid}/approve", token=ctx["tok"])
        v = c.call("GET", f"/v1/me/gifts/{gid}/giver-view", token=ctx["tok"])
        expect(v["status"] == "approved" and v["recommended_size"], "giver view has no size")
        expect(set(v) == {"id", "status", "recommended_size", "confidence"}, f"giver view exposes extra fields: {set(v)}")
        return f"giver sees only size {v['recommended_size']} and confidence {v['confidence']}"

    def widget_query() -> str:
        h = {"Origin": origin}
        consent = c.call("POST", "/v1/consents", {"merchant": merchant_key, "origin": origin}, token=ctx["tok"], headers=h)
        r = c.call("POST", "/v1/fit-query", {"product": _shirt_chart(), "consent_token": consent["token"]}, headers={"X-Merchant-Key": merchant_key, **h})
        expect(r["recommended_size"], "fit query returned no size")
        txt = json.dumps(r)
        expect("value_cm" not in txt and "ease" not in txt and "measure" not in txt, "business answer contains measurement fields")
        return f"store sees size {r['recommended_size']}, confidence {r['confidence']}, verdicts only"

    def returns_desk() -> str:
        h = {"X-Merchant-Id": ctx["session"]["merchant_id"]}
        rows = c.call("GET", "/v1/biz/returns", token=ctx["tok"], headers=h)
        expect(len(rows) == 4, f"expected 4 returns, got {len(rows)}")
        expect(len({r["routing"]["route"] for r in rows}) >= 3, "returns do not cover the routing branches")
        k = c.call("GET", "/v1/biz/kpis", token=ctx["tok"], headers=h)
        expect(isinstance(k, dict), "no KPIs")
        text = json.dumps(rows)
        expect("value_cm" not in text and "chest_cm" not in text, "desk exposes body measurements")
        return f"{len(rows)} returns on the desk"

    def isolation() -> str:
        other = c.call("POST", "/v1/demo/session", timeout=60)
        tok2 = c.sign_in(other)
        expect(other["uid"] != ctx["uid"], "two sessions got the same uid")
        mine = c.call("GET", "/v1/biz/returns", token=ctx["tok"], headers={"X-Merchant-Id": ctx["session"]["merchant_id"]})
        theirs = c.call("GET", "/v1/biz/returns", token=tok2, headers={"X-Merchant-Id": other["merchant_id"]})
        expect({r["id"] for r in mine}.isdisjoint({r["id"] for r in theirs}), "two sandboxes share returns")
        try:
            c.call("GET", "/v1/biz/returns", token=tok2, headers={"X-Merchant-Id": ctx["session"]["merchant_id"]})
        except Fail:
            return "a second judge cannot open the first judge's desk"
        raise Fail("a second sandbox could read the first sandbox's returns desk")

    steps: list[tuple[str, Callable[[], str]]] = [
        ("health", health), ("demo session", demo_session), ("twin view", twin_view), ("product check", product_check), ("occasion test", occasion_test),
        ("unboxing", unboxing), ("gift", gift), ("widget query", widget_query), ("returns desk", returns_desk), ("sandbox isolation", isolation)]
    failed = passed = 0
    started = time.time()
    for name, fn in steps:
        t0 = time.time()
        try:
            detail = fn()
            passed += 1
            print(f"PASS  {name:<18} {time.time() - t0:5.1f}s  {detail}")
        except Fail as e:
            failed += 1
            print(f"FAIL  {name:<18} {time.time() - t0:5.1f}s  {e}", file=sys.stderr)
            if name in ("health", "demo session"):
                print("      later steps need this one; stopping.", file=sys.stderr)
                break
        except Exception as e:  # a bug in the test or an unexpected payload must not look like a pass
            failed += 1
            print(f"FAIL  {name:<18} {time.time() - t0:5.1f}s  unexpected {type(e).__name__}: {e}", file=sys.stderr)
    print(f"\n{passed}/{len(steps)} steps passed in {time.time() - started:.1f}s")
    if failed:
        print("SMOKE TEST FAILED", file=sys.stderr)
    return 1 if failed else 0


def _shirt_chart() -> dict:
    """The Demo Threads shirt as the merchant would send it (same numbers as web/store/products.json)."""
    sizes = {"S": [108, 44, 60, 72], "M": [114, 46, 61.5, 74], "L": [120, 48, 63, 76], "XL": [126, 50, 64.5, 78]}
    zones = ("chest", "shoulder_width", "sleeve_length", "garment_length")
    return {"size_chart": {"sizes": {k: dict(zip(zones, v)) for k, v in sizes.items()}, "chart_type": "garment", "size_system": "ALPHA", "unit": "cm"},
            "category": "shirt", "fit_type": "regular", "fabric": [{"fiber": "cotton", "percent": 100}], "stretch_class": "none", "pre_shrunk": True, "ref": "dtc-shirt"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url", help="site origin, e.g. https://PROJECT.web.app")
    ap.add_argument("--api-prefix", default="/api", help="path prefix of the API on that origin (default /api; use '' for a bare API)")
    ap.add_argument("--firebase-api-key", default=os.environ.get("FIREBASE_API_KEY"))
    ap.add_argument("--assets-url", help="where static files (/demo/glb, /store) are served, if not the base URL (local runs)")
    ap.add_argument("--merchant-key", default="pk_demo_threads")
    ap.add_argument("--origin", help="Origin header for store calls (default: the base URL; it must be in the merchant's allowed origins)")
    a = ap.parse_args()
    if not a.base_url:
        sys.exit("usage: smoke_test.py BASE_URL")
    sys.exit(run(a.base_url, a.api_prefix, a.firebase_api_key, a.merchant_key, a.origin, a.assets_url))
