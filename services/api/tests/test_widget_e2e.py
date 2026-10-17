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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.conftest import API_DIR, ROOT

WEB = ROOT / "web"
KEY = "pk_demo_threads"
RESULT = re.compile(r"^([A-Z0-9]+) fits you · (high|medium|low) confidence")


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

    # 3. The API with a merchant registered for the store origin and a shopper who has a twin.
    os.environ.update({"AUTH_MODE": "dev", "CONSENT_JWT_SECRET": "e2e-secret-" + "y" * 24, "API_CORS_ORIGINS": app_url})
    for v in ("GOOGLE_CLOUD_PROJECT", "LLM_BACKEND"):
        os.environ.pop(v, None)
    sys.path.insert(0, str(API_DIR))
    import apilib.stores as stores
    mem = stores.MemoryStore()
    mem.set("merchants", "m_demo", {"name": "Demo Threads Co.", "publishable_key": KEY, "allowed_origins": [store_url]})
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    mem.set("twins", "demo-shopper", {"owner_id": "demo-shopper", "measurements": {"zones": twin["zones"]}})
    stores._store = mem
    spec = importlib.util.spec_from_file_location("api_main_e2e", API_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
    yield {"api": api_url, "app": app_url, "store": store_url, "store_port": store_port, "mem": mem}
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
        pg.on("response", lambda r: pg.errors.append(f"{r.status} {r.url}") if r.status >= 400 and "favicon" not in r.url else None)
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
