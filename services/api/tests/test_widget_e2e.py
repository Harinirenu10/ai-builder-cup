"""The widget on a store page, in a real browser, against the real API and the real consent popup.

Runs in Chromium and WebKit (Safari's engine; it is not Safari itself, so a manual check on a Mac or iPhone is still
worth doing). Needs `playwright install chromium webkit` and `npm install` in web/. Skips what is missing.
"""

import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.conftest import API_DIR, ROOT

WEB = ROOT / "web"
KEY = "pk_demo_threads"
RESULT = re.compile(r"^([A-Z0-9]+) fits you · (high|medium|low) confidence")

# A deterministic stand-in for the Reader service (real Playwright + a real LLM call would make
# this browser test slow and non-deterministic), in the same spirit as this file's other
# stand-ins (oracle_measurer, slow_memo, generate_standin): everything else in the request --
# auth, consent, the widget, the Fit Engine, privacy -- is real. See services/api/apilib/
# fitquery.py::read_product_from_url for the exact contract this fakes.
FAKE_READER_STATE: dict = {"status": 200, "body": {"status": "ok", "product": None}}


class FakeReaderHandler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # the request body (the URL) is not used by the fake
        payload = json.dumps(FAKE_READER_STATE["body"]).encode("utf-8")
        self.send_response(FAKE_READER_STATE["status"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_fake_reader(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), FakeReaderHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def browser_installed(name: str) -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            return Path(getattr(p, name).executable_path).exists()
    except Exception:
        return False


class SpaHandler(SimpleHTTPRequestHandler):
    """Static files, with unknown paths answered by index.html like Firebase Hosting's rewrite."""

    def log_message(self, *args):
        pass

    def send_head(self):
        path = self.translate_path(self.path.split("?")[0])
        if not os.path.exists(path) and "." not in Path(self.path.split("?")[0]).name:
            self.path = "/index.html"
        return super().send_head()


def serve(directory: Path, port: int, spa: bool = False) -> ThreadingHTTPServer:
    handler = SpaHandler if spa else SimpleHTTPRequestHandler
    if not spa:
        handler = type("Quiet", (SimpleHTTPRequestHandler,), {"log_message": lambda *a, **k: None})
    srv = ThreadingHTTPServer(("0.0.0.0", port), partial(handler, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="session")
def stack(tmp_path_factory):
    if not shutil.which("node") or not (WEB / "node_modules").is_dir():
        pytest.skip("Node and web/node_modules are needed to build the consent popup (run npm install in web/)")
    import uvicorn

    api_port, app_port, store_port = free_port(), free_port(), free_port()
    api_url, app_url, store_url = (f"http://127.0.0.1:{p}" for p in (api_port, app_port, store_port))
    tmp = tmp_path_factory.mktemp("e2e")

    # 1. The customer app, built with dev sign-in and this test's API address.
    dist = tmp / "app"
    env = {**os.environ, "VITE_AUTH_MODE": "dev", "VITE_API_URL": api_url}
    npx = shutil.which("npx") or "npx"
    build = subprocess.run([npx, "vite", "build", "--outDir", str(dist), "--emptyOutDir"], cwd=WEB, env=env, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout[-800:] + build.stderr[-800:]

    # 2. The storefront, generated for these ports, with the real widget file next to it.
    store_dir = tmp / "store"
    sys.path.insert(0, str(WEB / "store"))
    import build_store
    build_store.build(store_dir, api=api_url, app=app_url)       # copies the real widget next to the pages

    # A second page, not one of build_store.py's templates: the widget's data-fit-product holds
    # a bare product URL instead of embedded JSON, exercising POST /v1/fit-query/from-url.
    (store_dir / "from-url.html").write_text(
        '<!doctype html><html><body>'
        f'<div data-fit-product="http://example.invalid/kurta"></div>'
        f'<script src="widget/fit-widget.js" data-merchant="{KEY}" data-api="{api_url}" data-app="{app_url}" defer></script>'
        '</body></html>',
        encoding="utf-8",
    )

    # 2b. A deterministic stand-in for the Reader (see FakeReaderHandler above).
    reader_port = free_port()
    reader_url = f"http://127.0.0.1:{reader_port}"
    reader_srv = serve_fake_reader(reader_port)

    # 3. The API with a merchant registered for the store origin and a shopper who has a twin.
    os.environ.update({"AUTH_MODE": "dev", "CONSENT_JWT_SECRET": "e2e-secret-" + "y" * 24, "API_CORS_ORIGINS": app_url,
                       "READER_URL": reader_url})
    for v in ("GOOGLE_CLOUD_PROJECT", "LLM_BACKEND"):
        os.environ.pop(v, None)
    sys.path.insert(0, str(API_DIR))
    import apilib.stores as stores
    mem = stores.MemoryStore()
    mem.set("merchants", "m_demo", {"name": "Demo Threads Co.", "publishable_key": KEY, "allowed_origins": [store_url], "staff_uids": ["staff-1"], "return_policy": "review"})
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    mem.set("twins", "demo-shopper", {"owner_id": "demo-shopper", "measurements": {"zones": twin["zones"]}})
    stores._store = mem
    spec = importlib.util.spec_from_file_location("api_main_e2e", API_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The size chart studio measures with landmarks taken from the synthetic generator's truth (no model call).
    import hashlib
    from apilib.charts import get_measurer
    from fitcore.garment.landmarks import FileLandmarkFinder
    from fitcore.garment.measure import measure_garment
    photos: dict[str, object] = {}

    def oracle_measurer(image: bytes, category):
        photo = photos[hashlib.sha256(image).hexdigest()]
        return measure_garment(image, category, finder=FileLandmarkFinder({k: list(v) for k, v in photo.landmarks.items()}))

    mod.app.dependency_overrides[get_measurer] = lambda: oracle_measurer
    # The simulation page: the stand-in population (kept in memory) and a memo writer that takes a moment, like the real model.
    from apilib.simulate import get_memo_writer, get_population
    from fitcore.population import generate_standin
    from fitcore.sim_memo import Memo, MemoSection
    population = generate_standin()

    def slow_memo(result):
        time.sleep(2.0)
        return Memo(headline="Chart coverage memo", source="model", sections=[
            MemoSection(heading="Coverage", body=f"The chart covers {round(result.coverage * 100, 1)}% of the synthetic bodies."),
            MemoSection(heading="Caveat", body=result.banner)])

    mod.app.dependency_overrides[get_population] = lambda: population
    mod.app.dependency_overrides[get_memo_writer] = lambda: slow_memo
    server = uvicorn.Server(uvicorn.Config(mod.app, host="127.0.0.1", port=api_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(api_url + "/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("API did not start")

    app_srv, store_srv = serve(dist, app_port, spa=True), serve(store_dir, store_port)
    yield {"api": api_url, "app": app_url, "store": store_url, "store_port": store_port, "mem": mem, "photos": photos}
    reader_srv.shutdown()
    server.should_exit = True
    app_srv.shutdown()
    store_srv.shutdown()


@pytest.fixture(params=["chromium", "webkit"])
def page(request, stack, monkeypatch):
    from playwright.sync_api import sync_playwright

    monkeypatch.setenv("API_CORS_ORIGINS", stack["app"])        # the autouse test env points it elsewhere
    name = request.param
    if not browser_installed(name):
        pytest.skip(f"Playwright {name} is not installed")
    with sync_playwright() as p:
        browser = getattr(p, name).launch()
        ctx = browser.new_context()
        pg = ctx.new_page()
        pg.errors = []   # uncaught JS errors, and failed requests other than the favicon
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("response", lambda r: pg.errors.append(f"{r.status} {r.url}") if r.status >= 400 and "favicon" not in r.url and "googleapis.com" not in r.url else None)   # the Firebase SDK pings Google with the demo key
        pg.browser_name = name
        yield pg
        browser.close()


def approve_in_popup(page, trigger):
    with page.expect_popup() as pop:
        trigger()
    popup = pop.value
    popup.get_by_role("heading", name=re.compile("Share your size with Demo Threads Co")).wait_for(timeout=15000)
    popup.get_by_role("button", name="Sign in").click()
    popup.get_by_role("button", name="Approve").click()
    return popup


def test_widget_full_flow_then_remembers_consent_then_revocation(stack, page):
    page.goto(f"{stack['store']}/shirt.html")
    widget = page.locator(".mfw")
    widget.get_by_role("button", name="Find my size").wait_for()

    # First visit: consent popup, then the answer.
    approve_in_popup(page, lambda: widget.get_by_role("button", name="Find my size").click())
    result = widget.locator(".mfw-result")
    result.wait_for(timeout=20000)
    text = result.inner_text()
    m = RESULT.match(text)
    assert m, text
    assert m.group(1) in {"S", "M", "L", "XL"}
    assert page.locator("#size").input_value() == m.group(1)               # the store preselected the answer
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]
    shown = widget.inner_text()
    for z in twin.values():                                                # the widget never shows a body measurement
        assert not re.search(rf"(?<![\d.]){re.escape(str(z['value_cm']))}(?![\d.])", shown), z

    # Second visit: consent is remembered, so no popup.
    popups = []
    page.on("popup", lambda p: popups.append(p))
    page.reload()
    page.locator(".mfw").get_by_role("button", name="Find my size").click()
    page.locator(".mfw-result").wait_for(timeout=20000)
    assert popups == [] and RESULT.match(page.locator(".mfw-result").inner_text())

    # The shopper revokes in the customer app: the next query is refused and the widget asks to reconnect.
    consents = json.loads(urllib.request.urlopen(urllib.request.Request(stack["api"] + "/v1/consents", headers={"Authorization": "Bearer dev:demo-shopper"})).read())
    cid = consents[0]["consent_id"]
    urllib.request.urlopen(urllib.request.Request(f"{stack['api']}/v1/consents/{cid}/revoke", method="POST", headers={"Authorization": "Bearer dev:demo-shopper"}))
    page.reload()
    page.locator(".mfw").get_by_role("button", name="Find my size").click()
    page.get_by_text("Your connection to this store has ended").wait_for(timeout=15000)
    assert page.locator(".mfw-result").count() == 0
    assert [e for e in page.errors if not (e.startswith("403 ") and "/v1/fit-query" in e)] == [], page.errors   # only the refused query

    # Reconnecting works and creates a new consent.
    approve_in_popup(page, lambda: page.get_by_role("button", name="Connect again").click())
    page.locator(".mfw-result").wait_for(timeout=20000)
    assert RESULT.match(page.locator(".mfw-result").inner_text())


def test_unregistered_origin_gets_no_consent(stack, page):
    """The same store files served from an origin the merchant did not register (127.0.0.2 instead of 127.0.0.1)."""
    page.goto(f"http://127.0.0.2:{stack['store_port']}/shirt.html")
    with page.expect_popup() as pop:
        page.locator(".mfw").get_by_role("button", name="Find my size").click()
    popup = pop.value
    popup.get_by_test_id("connect-rejected").wait_for(timeout=15000)
    assert "not registered" in popup.get_by_test_id("connect-rejected").inner_text()
    assert popup.get_by_role("button", name="Approve").count() == 0            # nothing to approve
    popup.close()
    assert page.locator(".mfw-result").count() == 0
    assert page.evaluate("Object.keys(localStorage).filter(k => k.startsWith('mfw:consent'))") == []


def test_popup_message_goes_only_to_the_registered_origin(stack, page):
    """A page that is not the store cannot receive the token by listening for messages from the app popup."""
    page.goto(f"{stack['store']}/shirt.html")
    approve_in_popup(page, lambda: page.locator(".mfw").get_by_role("button", name="Find my size").click())
    page.locator(".mfw-result").wait_for(timeout=20000)
    # A message from the wrong origin or with a forged shape is ignored by the widget.
    page.evaluate("window.postMessage({type:'measured-fit:consent', token:'x'.repeat(40)}, '*')")
    page.wait_for_timeout(300)
    stored = json.loads(page.evaluate("localStorage.getItem('mfw:consent:pk_demo_threads')"))
    assert stored["token"] != "x" * 40


# ----------------------------------------------------------------------------- product-URL flow (POST /v1/fit-query/from-url)

SHIRT_GOOD_MATCH = {
    "status": "ok",
    "product": {
        "category": "shirt", "fit_type": "regular", "fabric": [], "stretch_class": "none", "pre_shrunk": None,
        "size_chart": {
            "chart_type": "garment", "unit": "cm",
            "sizes": {
                "S": {"chest": 108, "shoulder_width": 42, "sleeve_length": 57, "garment_length": 68},
                "M": {"chest": 112, "shoulder_width": 44, "sleeve_length": 59, "garment_length": 72},
                "L": {"chest": 118, "shoulder_width": 46, "sleeve_length": 61, "garment_length": 76},
            },
        },
    },
}

# Reproduces a real live-verification finding (see eval/results.md / the Reader audit): a BODY
# chart whose shoulder_width values put the shopper outside tshirt/regular's narrow shoulder-ease
# range (0.5-4cm) at every size -- a hard zone, so the Fit Engine correctly recommends no size.
TSHIRT_NO_GOOD_SIZE = {
    "status": "ok",
    "product": {
        "category": "tshirt", "fit_type": "regular", "fabric": [], "stretch_class": "none", "pre_shrunk": None,
        "size_chart": {
            "chart_type": "body", "unit": "cm",
            "sizes": {
                "S": {"chest": 98, "shoulder_width": 45, "garment_length": 70},
                "M": {"chest": 104, "shoulder_width": 47, "garment_length": 72},
                "L": {"chest": 110, "shoulder_width": 49, "garment_length": 74},
            },
        },
    },
}


def test_widget_from_url_reads_the_page_live_and_shows_a_recommendation(stack, page):
    """data-fit-product holds a bare URL (not JSON): the widget calls /v1/fit-query/from-url."""
    FAKE_READER_STATE["status"], FAKE_READER_STATE["body"] = 200, SHIRT_GOOD_MATCH
    page.goto(f"{stack['store']}/from-url.html")
    widget = page.locator(".mfw")
    widget.get_by_role("button", name="Find my size").wait_for()

    approve_in_popup(page, lambda: widget.get_by_role("button", name="Find my size").click())
    result = widget.locator(".mfw-result")
    result.wait_for(timeout=20000)
    text = result.inner_text()
    # Not RESULT.match(text): Chromium's innerText on this minimal (no stylesheet) test page
    # renders the widget's " · " separator with a non-breaking space, which looks identical to
    # a normal space when printed but does not match that literal character in the pattern.
    assert text.split()[0] == "M" and "fits you" in text and "confidence" in text, repr(text)

    # Same privacy rule as the embedded-chart flow: no body measurement ever appears.
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]
    shown = widget.inner_text()
    for z in twin.values():
        assert not re.search(rf"(?<![\d.]){re.escape(str(z['value_cm']))}(?![\d.])", shown), z
    assert page.errors == [], page.errors


def test_widget_from_url_never_forces_a_size_when_the_fit_engine_rejects_all_of_them(stack, page):
    FAKE_READER_STATE["status"], FAKE_READER_STATE["body"] = 200, TSHIRT_NO_GOOD_SIZE
    page.goto(f"{stack['store']}/from-url.html")
    widget = page.locator(".mfw")
    approve_in_popup(page, lambda: widget.get_by_role("button", name="Find my size").click())
    page.get_by_text("None of the sizes in this chart is a good match for this garment.").wait_for(timeout=20000)
    # the exact same no-forced-size UI the embedded-chart flow already uses; no size letter shown
    assert not RESULT.search(widget.inner_text())
    assert widget.get_by_role("button", name="Check again").count() == 1


def test_widget_from_url_unsupported_category_shows_a_clear_message_not_a_crash(stack, page):
    """The real Reader answers an unsupported garment with HTTP 200 and status:"unsupported" in
    the body (readerlib/pipeline.py), not an HTTP error -- the fake mirrors that exact shape."""
    FAKE_READER_STATE["status"] = 200
    FAKE_READER_STATE["body"] = {"status": "unsupported", "reason": "Not a supported garment category"}
    page.goto(f"{stack['store']}/from-url.html")
    widget = page.locator(".mfw")
    approve_in_popup(page, lambda: widget.get_by_role("button", name="Find my size").click())
    page.get_by_text("This product isn't a supported garment category yet.").wait_for(timeout=20000)
    assert widget.locator(".mfw-result").count() == 0


def test_widget_from_url_reader_http_error_shows_a_clear_message_not_the_raw_reason(stack, page):
    """The Reader's own /read can answer with an HTTP error (e.g. an unsafe URL, see
    readerlib/security.py); the widget must show its own clear text, not relay that detail
    verbatim. (A genuine client-side timeout is covered by the fast, deterministic unit test
    services/api/tests/test_fit_query_from_url.py::test_from_url_reader_timeout instead of here,
    since reproducing a real socket timeout would make this browser test slow and flaky.)"""
    FAKE_READER_STATE["status"] = 422
    FAKE_READER_STATE["body"] = {"detail": "Only http(s) URLs are allowed"}
    page.goto(f"{stack['store']}/from-url.html")
    widget = page.locator(".mfw")
    approve_in_popup(page, lambda: widget.get_by_role("button", name="Find my size").click())
    page.get_by_text("We could not read that product page. Please try again.").wait_for(timeout=20000)
    assert "Only http(s) URLs are allowed" not in widget.inner_text()
    assert widget.locator(".mfw-result").count() == 0


def test_size_guide_opens_and_bag_allows_two_sizes_of_one_item(stack, page):
    page.goto(f"{stack['store']}/tshirt.html")
    page.get_by_role("button", name="Size guide").click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for()
    assert "Chest" in dialog.inner_text() and dialog.locator("tr").count() == 5
    page.get_by_role("button", name="Close").click()
    for size in ("M", "L"):
        page.select_option("#size", size)
        page.get_by_role("button", name="Add to bag").click()
    assert "sizes M and L" in page.locator("#bag-note").inner_text()
    page.goto(f"{stack['store']}/cart.html")
    lines = page.locator(".line")
    assert lines.count() == 2
    assert "size M" in lines.nth(0).inner_text() and "size L" in lines.nth(1).inner_text()
    assert page.locator(".tag").count() == 2 and "more than one size" in page.locator("#cart-two-sizes").inner_text()


# ----------------------------------------------------------------------------- guarantee, nudge, returns desk

OVERLAY = "data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Crect width='60' height='60' fill='%23cbd5e1'/%3E%3C/svg%3E"
SHIRT = json.loads((ROOT / "web" / "store" / "products.json").read_text(encoding="utf-8"))["shirt"]


def connect_on_shirt(page, stack):
    page.goto(f"{stack['store']}/shirt.html")
    approve_in_popup(page, lambda: page.locator(".mfw").get_by_role("button", name="Find my size").click())
    page.locator(".mfw-result").wait_for(timeout=20000)
    return RESULT.match(page.locator(".mfw-result").inner_text()).group(1)


def add_size(page, size):
    page.select_option("#size", size)
    page.get_by_role("button", name="Add to bag").click()


def test_nudge_checkout_return_and_desk_end_to_end(stack, page):
    mem = stack["mem"]
    rec = connect_on_shirt(page, stack)
    other = next(s for s in SHIRT["size_chart"]["sizes"] if s != rec)

    # Two sizes of one item in the bag: the nudge names the recommended size and offers to remove the other.
    add_size(page, other)
    add_size(page, rec)
    before = len(mem.find("events", type="nudge_shown"))
    page.goto(f"{stack['store']}/cart.html")
    nudge = page.locator(".nudge")
    nudge.wait_for(timeout=20000)
    assert nudge.locator("span").inner_text() == f"Your twin says {rec} fits. Remove the {other}?"
    assert page.locator(".line").count() == 2
    nudge.get_by_role("button", name=f"Remove the {other}").click()
    page.wait_for_function("document.querySelectorAll('.line').length === 1")
    assert page.locator(".nudge").count() == 0 and f"size {rec}" in page.locator(".line").inner_text()
    page.wait_for_timeout(500)
    assert len(mem.find("events", type="nudge_shown")) == before + 1 and len(mem.find("events", type="nudge_accepted")) >= 1
    assert all(set(e) == {"id", "merchant_id", "type", "product_ref", "ts"} for e in mem.find("events"))

    # Checkout: the guarantee is decided on the server.
    page.get_by_role("link", name="Checkout").click()
    page.locator("#place-order").click()
    result = page.locator("#order-result")
    result.get_by_text("Order placed (simulated)").wait_for(timeout=20000)
    text = result.inner_text()
    assert "Fit guarantee: yes" in text
    order_id = re.search(r"Order ID: ([0-9a-f]{32})", text).group(1)
    doc = mem.find("orders", id=order_id)[0]
    assert doc["uid"] == "demo-shopper"                      # tied to the consenting shopper
    assert doc["items"][0]["guarantee"] is True and doc["items"][0]["recommended_size_at_purchase"] == rec

    # A simulated unboxing check: the garment labelled `rec` measures like the next size down, so the label is wrong.
    sizes = list(SHIRT["size_chart"]["sizes"])
    wrong = sizes[sizes.index(rec) - 1] if sizes.index(rec) else sizes[1]
    row = {z: v for z, v in SHIRT["size_chart"]["sizes"][wrong].items() if z in ("chest", "garment_length", "sleeve_length")}
    mem.set("unboxing_measurements", "ub-e2e", {"order_id": order_id, "sku": "dtc-shirt", "size_label": rec, "measured_cm": row,
                                                "overlay_image_url": OVERLAY, "source": "simulated"})
    page.goto(f"{stack['store']}/returns.html")
    page.select_option("#return-reason", "too_small")
    page.fill("#return-unboxing", "ub-e2e")
    page.get_by_role("button", name="Submit return").click()
    page.locator("#return-result").get_by_text("exchange").wait_for(timeout=20000)
    assert "approved" in page.locator("#return-result").inner_text()
    ret = mem.find("returns", order_id=order_id)[0]
    assert ret["routing"]["route"] == "auto_exchange" and ret["routing"]["seller_label_error"] is True

    # The desk: staff sign in, see the evidence, and act. A second return from the same item is refused.
    page.goto(f"{stack['app']}/biz/returns")
    page.fill("#desk-uid", "staff-1")
    page.get_by_role("button", name="Sign in").click()
    card = page.get_by_test_id("evidence-card")
    card.wait_for(timeout=20000)
    assert "Rule 1" in page.get_by_test_id("routing-reason").inner_text()
    assert f"size {rec}" in page.get_by_test_id("twin-verdict").inner_text()
    assert page.get_by_test_id("spec-table").locator("tbody tr").count() >= 2
    assert "SKU dtc-shirt is flagged" in card.inner_text()
    kpis = json.loads(urllib.request.urlopen(urllib.request.Request(stack["api"] + "/v1/biz/kpis", headers={"Authorization": "Bearer dev:staff-1"})).read())
    assert f"{round(kpis['guaranteed_share'] * 100)}%" in page.get_by_test_id("kpi-guaranteed").inner_text()      # the strip shows what the API computed
    labels = page.get_by_test_id("kpi-labels").inner_text()
    assert str(kpis["seller_label_errors"]) in labels and "dtc-shirt" in labels and kpis["seller_label_errors"] >= 1
    assert f"{kpis['nudges_accepted']} of {kpis['nudges_shown']}" in page.get_by_test_id("kpi-nudges").inner_text() and kpis["nudges_accepted"] >= 1
    page.get_by_test_id("summarize").click()
    page.get_by_test_id("summary-text").wait_for(timeout=20000)
    summary = page.get_by_test_id("summary-text").inner_text()
    assert not re.search(r"fraud|lying|liar|scam|cheat|suspicious", summary, re.I)
    page.get_by_role("button", name="Send to review").click()
    page.get_by_test_id("card-status").get_by_text("in review").wait_for(timeout=15000)
    assert mem.get("returns", ret["id"])["status"] == "in_review"
    assert page.get_by_role("button", name=re.compile("reject", re.I)).count() == 0
    assert page.get_by_alt_text("Unboxing measurement overlay").evaluate("img => img.complete && img.naturalWidth > 0")   # the overlay loaded
    assert page.errors == [], page.errors


def test_desk_refuses_a_user_who_is_not_staff(stack, page):
    page.goto(f"{stack['app']}/biz/returns")
    page.fill("#desk-uid", "demo-shopper")
    page.get_by_role("button", name="Sign in").click()
    page.get_by_test_id("desk-error").wait_for(timeout=15000)
    assert "not on a merchant" in page.get_by_test_id("desk-error").inner_text()
    assert page.get_by_test_id("evidence-card").count() == 0


# ----------------------------------------------------------------------------- size chart studio

ZONES5 = ("chest", "waist", "shoulder_width", "sleeve_length", "garment_length")
STUDIO_SIZES = {"S": (108, 104, 44, 60, 72), "M": (114, 110, 46, 61.5, 74), "L": (120, 116, 48, 63, 76), "XL": (118, 122, 50, 64.5, 78)}   # XL chest is 2 cm below L


def test_chart_studio_photos_to_published_chart_on_the_store(stack, page):
    import hashlib
    from fitcore.garment.synthetic import render_flatlay
    from fitcore.schemas import Category, MeasurementZone as Z

    mem = stack["mem"]
    photos = {}
    for i, (label, vals) in enumerate(STUDIO_SIZES.items()):
        photo = render_flatlay(Category.shirt, {Z(z): v for z, v in zip(ZONES5, vals)}, seed=40 + i, tilt=0.03)
        stack["photos"][hashlib.sha256(photo.jpeg).hexdigest()] = photo
        photos[label] = photo
    url = f"{stack['store']}/shirt.html"
    mem.set("products", hashlib.sha256(url.encode()).hexdigest(), {"url": url, "fetched_at": "2026-10-17T00:00:00+00:00", "response": {
        "status": "ok", "product": {"size_chart": {"sizes": {s: {"chest": 110 + 6 * i} for i, s in enumerate(STUDIO_SIZES)}, "chart_type": "garment", "unit": "cm"}},
        "reviews": {"reviews_read": 6, "zone_signals": [{"zone": "chest", "direction": "runs_small", "strength": 0.6, "mention_count": 3, "reviewer_context": []}]}}})

    page.goto(f"{stack['app']}/biz/charts")
    page.fill("#studio-uid", "staff-1")
    page.get_by_role("button", name="Sign in").click()
    page.get_by_test_id("category").wait_for(timeout=20000)
    page.get_by_test_id("category").select_option("shirt")
    page.get_by_test_id("product-name").fill("Oxford Button-Down Shirt")
    page.get_by_test_id("product-ref").fill("dtc-shirt")
    page.get_by_test_id("product-url").fill(url)
    for label, photo in photos.items():
        page.get_by_test_id(f"photo-{label}").set_input_files({"name": f"{label}.jpg", "mimeType": "image/jpeg", "buffer": photo.jpeg})
    page.get_by_test_id("measure-all").click()
    for label in STUDIO_SIZES:
        page.get_by_test_id(f"overlay-{label}").wait_for(timeout=40000)
        assert page.get_by_test_id(f"status-{label}").inner_text().startswith("Measured")

    # The table holds what was measured; the planted anomaly is flagged in words, with the reviews beside it.
    for label, vals in STUDIO_SIZES.items():
        for z, truth in zip(ZONES5, vals):
            assert abs(float(page.get_by_test_id(f"cell-{label}-{z}").input_value()) - truth) <= 2.5, (label, z)
    page.get_by_test_id("flag").first.wait_for(timeout=20000)
    flag_text = " ".join(page.get_by_test_id("flag").all_inner_texts())
    assert re.search(r"XL chest is [0-9.]+ cm smaller than L", flag_text) and "mislabeled sample" in flag_text      # the real size of the gap is a measurement
    note = page.get_by_test_id("review-note").first
    note.wait_for(timeout=20000)
    assert "3 reviewers say it runs small at the chest" in note.inner_text() and "your old chart" in note.inner_text()

    # cm / inch toggle changes only what is shown.
    cm_value = float(page.get_by_test_id("cell-M-chest").input_value())
    page.get_by_test_id("unit-in").click()
    assert abs(float(page.get_by_test_id("cell-M-chest").input_value()) - cm_value / 2.54) <= 0.06
    page.get_by_test_id("unit-cm").click()
    assert abs(float(page.get_by_test_id("cell-M-chest").input_value()) - cm_value) <= 0.06

    # Fix the mislabeled value by hand: the flag goes away.
    page.get_by_test_id("cell-XL-chest").fill("126")
    page.wait_for_function("!/XL chest is [0-9.]+ cm smaller than L/.test(document.body.innerText)", timeout=15000)

    page.get_by_test_id("publish").click()
    page.get_by_test_id("published").wait_for(timeout=20000)
    version = int(re.search(r"version (\d+)", page.get_by_test_id("published").inner_text()).group(1))      # 2 on the second browser: the store is shared
    exported = json.loads(page.get_by_test_id("export-json").input_value())
    assert exported["product_ref"] == "dtc-shirt" and exported["unit"] == "cm" and list(exported["sizes"]) == ["S", "M", "L", "XL"]
    assert exported["sizes"]["XL"]["chest"] == 126.0                                      # the edited value is what was published
    assert "<table" in page.get_by_test_id("export-html").input_value()
    elapsed = float(re.search(r"([0-9.]+) s", page.get_by_test_id("elapsed").inner_text()).group(1))
    assert elapsed < 60, elapsed

    page.get_by_test_id("use-on-store").click()
    page.get_by_test_id("live-note").wait_for(timeout=15000)

    # The store page now shows the measured chart and the widget uses it.
    page.goto(f"{stack['store']}/shirt.html")
    page.wait_for_function(f"document.body.getAttribute('data-chart-version') === '{version}'", timeout=20000)
    page.get_by_role("button", name="Size guide").click()
    dialog = page.get_by_role("dialog")
    assert f"Measured by the seller (version {version})" in dialog.inner_text() and "126" in dialog.inner_text()
    assert [o.inner_text() for o in page.locator("#size option").all()] == ["S", "M", "L", "XL"]
    assert json.loads(page.locator("[data-fit-product]").get_attribute("data-fit-product"))["size_chart"]["chart_type"] == "garment"
    page.get_by_role("button", name="Close").click()
    page.locator(".mfw").get_by_role("button", name="Find my size").wait_for()             # the widget was re-mounted with the new chart


# ----------------------------------------------------------------------------- return simulation

KURTA_BODY_IN = {"category": "kurta", "chart_type": "body", "unit": "in", "sizes": {
    "S": {"chest": 34.5, "waist": 28.5, "hip": 36.5}, "M": {"chest": 36.5, "waist": 30.5, "hip": 38.5}, "L": {"chest": 38.5, "waist": 32.5, "hip": 40.5}}}


def test_return_simulation_statistics_first_then_the_memo_and_the_banner_stays(stack, page):
    page.goto(f"{stack['app']}/biz/simulate")
    banner = page.get_by_test_id("sim-banner")
    banner.wait_for(timeout=20000)                                           # on screen even before signing in
    assert "Synthetic data." in banner.inner_text() and "Calibrate with consented, anonymized twins before relying on it." in banner.inner_text()
    page.fill("#sim-uid", "staff-1")
    page.get_by_role("button", name="Sign in").click()
    page.get_by_test_id("example-shirt").click()
    assert "example shirt" in page.get_by_test_id("chart-source").inner_text()
    started = time.monotonic()
    page.get_by_test_id("sim-run").click()
    page.get_by_test_id("kpi-covered").wait_for(timeout=20000)
    assert time.monotonic() - started < 20                                     # the statistics do not wait for the memo
    assert page.get_by_test_id("memo-writing").is_visible() and page.get_by_test_id("memo").count() == 0

    banner_text = banner.inner_text()
    assert "stand-in" in banner_text and "not the SMPL shape space" in banner_text         # never claims SMPL for the stand-in
    for size in ("S", "M", "L", "XL", "No size recommended"):
        assert page.get_by_test_id(f"mix-{size}").is_visible()
    assert page.get_by_test_id("zone-M-chest").is_visible() and "tight" in page.get_by_test_id("zone-M-chest").inner_text()
    covered = float(page.get_by_test_id("kpi-covered").inner_text().split("%")[0].split()[-1])
    assert 0 <= covered <= 100
    assert "now," in page.get_by_test_id("regrade-summary").inner_text()

    page.mouse.wheel(0, 4000)                                                  # scroll down: the banner stays in view
    page.wait_for_timeout(300)
    assert banner.bounding_box()["y"] <= 1 and banner.is_visible()

    page.get_by_test_id("memo").wait_for(timeout=30000)
    assert page.get_by_test_id("memo-headline").inner_text() == "Chart coverage memo"
    assert "Written by the pro model" in page.get_by_test_id("memo-source").inner_text()
    assert "Synthetic population from a statistical stand-in" in page.get_by_test_id("memo").inner_text()      # the caveat is in the memo
    page.get_by_test_id("show-facts").click()
    facts = json.loads(page.get_by_test_id("facts").inner_text())
    assert facts["population"]["bodies"] == 4000 and abs(facts["covered_percent"] - covered) < 0.11      # the facts are the numbers on the page


def test_return_simulation_with_an_uploaded_chart_in_inches_and_a_missing_chart(stack, page):
    page.goto(f"{stack['app']}/biz/simulate")
    page.fill("#sim-uid", "staff-1")
    page.get_by_role("button", name="Sign in").click()
    page.get_by_test_id("sim-run").click()
    assert "Pick a published chart" in page.get_by_test_id("sim-error").inner_text()           # nothing chosen yet
    page.get_by_test_id("chart-file").set_input_files({"name": "kurta.json", "mimeType": "application/json", "buffer": json.dumps(KURTA_BODY_IN).encode()})
    page.wait_for_function("document.querySelector('[data-testid=chart-source]').innerText.includes('kurta.json')")
    assert page.get_by_test_id("sim-category").input_value() == "kurta"
    page.get_by_test_id("sim-run").click()
    page.get_by_test_id("kpi-covered").wait_for(timeout=20000)
    assert page.get_by_test_id("mix-L").is_visible() and page.get_by_test_id("zone-L-hip").is_visible()
    assert page.get_by_test_id("sim-error").count() == 0
    assert page.errors == [], page.errors
