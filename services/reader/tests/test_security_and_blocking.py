import time

import pytest

from readerlib.capture import capture_page, detect_block
from readerlib.security import UnsafeURL, validate_url


# ----------------------------------------------------------------------------- URL safety


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "http://localhost:8000/x",
                                 "http://127.0.0.1/x", "http://169.254.169.254/latest", "http://metadata.google.internal/", "http://10.0.0.5/", "http://[::1]/", "notaurl"])
def test_unsafe_urls_rejected(url):
    with pytest.raises(UnsafeURL):
        validate_url(url, allow_private=False)


def test_private_allowed_only_when_enabled():
    assert validate_url("http://127.0.0.1:9/x", allow_private=True) == "http://127.0.0.1:9/x"


# ----------------------------------------------------------------------------- block detection (pure)

SHOP_TEXT = "Menu Sign in Cart Heavyweight Tee " + "great fabric and fit " * 300


@pytest.mark.parametrize("status,text,captcha,password,path,expected", [
    (403, "", False, False, "/", "http_403"),
    (401, "", False, False, "/", "http_401"),
    (429, "", False, False, "/", "http_429"),
    (200, "Please verify you are human. Complete the captcha.", False, False, "/", "captcha"),
    (200, "Welcome", True, False, "/", "captcha"),
    (200, "Sign in to continue", False, True, "/login", "login_wall"),
    (200, SHOP_TEXT, False, False, "/p/1", None),                  # 'Sign in' in a normal shop header is fine
    (200, SHOP_TEXT + " captcha", True, False, "/p/1", None),      # a newsletter recaptcha on a long page is fine
    (200, SHOP_TEXT, False, True, "/p/1", None),                   # a login box on a long page is fine
    (200, "Access Denied", False, False, "/", "captcha"),
])
def test_detect_block(status, text, captcha, password, path, expected):
    assert detect_block(status, text, captcha, password, path) == expected


# ----------------------------------------------------------------------------- blocked pages (real Chromium, local server)


@pytest.mark.parametrize("route,reason", [("/_captcha", "captcha"), ("/_forbidden", "http_403"), ("/_login", "login_wall"), ("/_ratelimited", "http_429")])
def test_blocked_pages_return_blocked_quickly(need_chromium, server, route, reason):
    start = time.monotonic()
    cap = capture_page(server + route)
    elapsed = time.monotonic() - start
    assert cap.status == "blocked" and cap.reason == reason
    assert elapsed < 25, f"took {elapsed:.1f}s"
    assert cap.text == "" and cap.chart_png is None  # nothing is extracted from a barrier page


def test_missing_page_is_an_error_not_a_product(need_chromium, server):
    cap = capture_page(server + "/does-not-exist.html")
    assert cap.status == "error" and cap.reason == "http_404"


def test_unreachable_host_is_an_error_within_budget(need_chromium):
    start = time.monotonic()
    cap = capture_page("http://127.0.0.1:9/", budget_s=10)
    assert cap.status == "error" and time.monotonic() - start < 25


def test_ok_page_has_text_jsonld_images_and_chart(need_chromium, server):
    cap = capture_page(server + "/demo_threads_tee.html")
    assert cap.status == "ok" and cap.chart_opened and cap.chart_png and cap.full_page_jpeg
    assert "Size chart (opened):" in cap.text and "Customer reviews" in cap.text
    assert cap.json_ld and cap.json_ld[0]["name"] == "Heavyweight Boxy Crewneck Tee"
    assert any("tee-front" in u for u in cap.image_urls)
