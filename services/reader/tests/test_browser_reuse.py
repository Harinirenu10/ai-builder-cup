"""Regression tests for the shared-browser capture host (readerlib/capture.py::_BrowserHost).

Every /read used to start Playwright's driver and a fresh Chromium, then close both -- about
6.7s of the ~9s a cache-missing capture spent, measured against the demo store page whose own
page work is ~2.2s. On top of the extraction model call that pushed a first read past the
caller's 15s request timeout, so the browser gave up on a page the reader could render fine.

The browser process is now shared across reads and warmed at service startup, while each visit
still gets its own fresh context. These tests pin the three properties that matters for: the
browser is reused rather than relaunched, visits stay isolated from each other, and a dead
browser is replaced instead of failing every later read.
"""

from __future__ import annotations

import pytest

from readerlib import capture as capture_mod
from readerlib.capture import capture_page, close_browser, warm_browser

PAGE = "/demo_threads_tee.html"


@pytest.fixture(autouse=True)
def _fresh_host():
    """Each test starts and ends with no shared browser, so order cannot matter."""
    close_browser()
    yield
    close_browser()


def test_warm_browser_launches_before_any_capture(need_chromium):
    assert capture_mod._host._browser is None
    warm_browser()
    browser = capture_mod._host._browser
    assert browser is not None and browser.is_connected()


def test_the_same_browser_process_serves_consecutive_captures(need_chromium, server):
    first = capture_page(server + PAGE)
    browser_after_first = capture_mod._host._browser
    second = capture_page(server + PAGE)
    browser_after_second = capture_mod._host._browser

    assert first.status == "ok" and second.status == "ok"
    assert browser_after_first is not None
    # Reused, not relaunched: the same object, still connected.
    assert browser_after_second is browser_after_first
    assert browser_after_second.is_connected()


def test_each_capture_closes_its_own_context(need_chromium, server):
    """Isolation: a visit's context (cookies, storage, cache) must not outlive it or be shared."""
    capture_page(server + PAGE)
    capture_page(server + PAGE)
    browser = capture_mod._host._browser
    assert browser is not None and browser.is_connected()
    assert browser.contexts == [], "a capture left its browser context open"


def test_capture_still_returns_the_same_page_content_as_before(need_chromium, server):
    """Reuse must not change what a capture produces."""
    first = capture_page(server + PAGE)
    second = capture_page(server + PAGE)
    assert first.status == second.status == "ok"
    assert first.text and first.text == second.text
    assert first.final_url == second.final_url


def test_a_dead_browser_is_relaunched_for_the_next_capture(need_chromium, server):
    """A crashed or externally killed Chromium must not break every later read."""
    import asyncio

    capture_page(server + PAGE)
    dead = capture_mod._host._browser
    assert dead is not None
    asyncio.run_coroutine_threadsafe(dead.close(), capture_mod._host._loop).result(timeout=30)
    assert not dead.is_connected()

    recovered = capture_page(server + PAGE)
    assert recovered.status == "ok", f"capture failed after the browser died: {recovered.reason}"
    assert capture_mod._host._browser is not dead
    assert capture_mod._host._browser.is_connected()


def test_close_browser_tears_everything_down(need_chromium, server):
    capture_page(server + PAGE)
    assert capture_mod._host._browser is not None
    close_browser()
    assert capture_mod._host._browser is None
    assert capture_mod._host._loop is None
    # And the host comes back to life on demand.
    assert capture_page(server + PAGE).status == "ok"


def test_unreachable_host_still_reports_an_error_not_a_hang(need_chromium):
    cap = capture_page("http://127.0.0.1:9/", budget_s=10)
    assert cap.status == "error"
