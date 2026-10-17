import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

READER_DIR = Path(__file__).resolve().parents[1]
SAMPLES = READER_DIR.parents[1] / "samples" / "products"
sys.path.insert(0, str(READER_DIR))

CAPTCHA_PAGE = b"""<html><head><title>Just a moment...</title></head><body>
<h1>Please verify you are human</h1><p>Complete the captcha below to continue.</p>
<div class="g-recaptcha" data-sitekey="fake" style="width:300px;height:80px;border:1px solid #000">captcha widget</div></body></html>"""
LOGIN_PAGE = b"""<html><body><h1>Sign in to continue</h1><form><input name=email><input type=password name=pw><button>Log in</button></form></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):
        routes = {"/_captcha": (200, CAPTCHA_PAGE), "/_login": (200, LOGIN_PAGE),
                  "/_forbidden": (403, b"<html><body>Forbidden</body></html>"),
                  "/_ratelimited": (429, b"<html><body>Too many requests</body></html>")}
        if self.path in routes:
            code, body = routes[self.path]
            self.send_response(code)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


@pytest.fixture(scope="session")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(SAMPLES)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in ("LLM_BACKEND", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_MAX_CALLS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "vertex")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "recorded-fixture")
    monkeypatch.setenv("READER_ALLOW_PRIVATE", "1")
    monkeypatch.setenv("READER_OFFLINE", "1")
    monkeypatch.setenv("READER_CACHE", "memory")


def chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


@pytest.fixture(scope="session")
def need_chromium():
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed (run: playwright install chromium)")
