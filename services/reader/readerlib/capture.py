"""Read a product page with headless Chromium (async Playwright in its own event loop, so a hung page cannot outlive the budget).

One visit per URL. If the site shows a captcha, a login wall or answers 401/403/429, we stop and report
"blocked". Nothing here tries to get around that: no stealth, no retries, no proxies.
"""

import asyncio
import atexit
import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

BUDGET_S = 20.0
VIEWPORT = {"width": 1440, "height": 900}
MAX_SHOT_HEIGHT = 8000
BLOCK_STATUSES = {401, 403, 429}
SIZE_CHART_RE = re.compile(r"size (chart|guide)|measurement", re.I)

_CAPTCHA_TEXT = re.compile(
    r"captcha|verify (that )?you('re| are) (a )?human|are you a (robot|human)|unusual traffic|"
    r"press (&|and) hold|checking your browser|access denied|attention required", re.I)
_LOGIN_TEXT = re.compile(r"sign[ -]?in|log[ -]?in|login", re.I)

_SIGNALS_JS = """() => {
  const vis = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e);
    return r.width > 40 && r.height > 20 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const q = sel => Array.from(document.querySelectorAll(sel)).some(vis);
  return {
    captcha_visible: q("iframe[src*='recaptcha'], iframe[src*='hcaptcha'], iframe[src*='challenges.cloudflare.com'], .g-recaptcha, .h-captcha, #cf-challenge-running, #challenge-form, [data-sitekey]"),
    password_visible: q("input[type=password]"),
    title: document.title || "",
    path: location.pathname,
    scroll_height: document.documentElement.scrollHeight,
  };
}"""

_LD_JS = "() => Array.from(document.querySelectorAll('script[type=\"application/ld+json\"]')).map(s => s.textContent)"
_IMG_JS = """() => {
  const out = [];
  const og = document.querySelector("meta[property='og:image']"); if (og && og.content) out.push(og.content);
  for (const i of document.images) { if (i.naturalWidth >= 300 && i.naturalHeight >= 300 && /^https?:/.test(i.src)) out.push(i.src); }
  return out;
}"""
_HIDDEN_PANELS_JS = """() => {
  const sel = "[role=dialog], [aria-modal], dialog, [class*='modal'], [class*='drawer'], [class*='popup'], [class*='size-guide'], [class*='size_guide'], [id*='size']";
  const re = /chest|bust|waist|hip|length|shoulder|sleeve|inseam/i;
  const out = [];
  for (const e of document.querySelectorAll(sel)) {
    const t = (e.textContent || "").replace(/\\s+/g, " ").trim();
    if (re.test(t) && t.length > 40 && t.length < 6000 && !out.includes(t)) out.push(t);
  }
  return out.sort((a, b) => a.length - b.length).slice(0, 3);
}"""
_MODAL_SEL = "[role=dialog], [aria-modal='true'], dialog[open], [class*='modal'], [class*='drawer'], [class*='popup']"


@dataclass
class Capture:
    status: Literal["ok", "blocked", "error"]
    reason: str | None = None
    final_url: str | None = None
    http_status: int | None = None
    text: str = ""
    json_ld: list[dict[str, Any]] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    full_page_jpeg: bytes | None = None
    chart_png: bytes | None = None
    chart_opened: bool = False
    html: str = ""


def detect_block(http_status: int | None, text: str, captcha_visible: bool = False,
                 password_visible: bool = False, path: str = "") -> str | None:
    """Reason the page is a barrier, or None. Pure so it can be unit tested.

    Captcha and login words alone do not count on a long page (a normal shop mentions both in its header).
    """
    if http_status in BLOCK_STATUSES:
        return f"http_{http_status}"
    short = len(text.strip()) < 3000
    if short and (captcha_visible or _CAPTCHA_TEXT.search(text)):
        return "captcha"
    if password_visible and len(text.strip()) < 2500 and (_LOGIN_TEXT.search(text) or re.search(r"log-?in|sign-?in", path, re.I)):
        return "login_wall"
    return None


def _flatten_ld(raw: list[str]) -> list[dict[str, Any]]:
    """Product objects from JSON-LD blocks (handles @graph and lists)."""
    out: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                visit(n)
        elif isinstance(node, dict):
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if any(isinstance(x, str) and x.lower() == "product" for x in types):
                out.append(node)
            if "@graph" in node:
                visit(node["@graph"])

    for block in raw:
        try:
            visit(json.loads(block))
        except (ValueError, TypeError):
            continue
    return out


async def _step(coro: Any, cap_ms: float) -> Any:
    """Run one optional step under its own time limit; None if it fails or times out."""
    try:
        return await asyncio.wait_for(coro, timeout=cap_ms / 1000.0)
    except Exception:
        return None


async def _capture(url: str, budget_s: float, browser: Any) -> Capture:
    from playwright.async_api import Error as PWError
    from playwright.async_api import TimeoutError as PWTimeout

    deadline = time.monotonic() + budget_s

    def left_ms(cap: float | None = None) -> float:
        ms = max(300.0, (deadline - time.monotonic()) * 1000)
        return min(ms, cap) if cap else ms

    # A fresh context per capture keeps every visit isolated (its own cookies, storage and cache),
    # exactly as a fresh browser did -- only the expensive browser process itself is now shared.
    async with await browser.new_context(viewport=VIEWPORT) as context:
        try:
            page = await context.new_page()
            if url.startswith("file:") or os.environ.get("READER_OFFLINE") == "1":
                # Saved sample pages (and tests) must render the same offline: refuse requests to other hosts.
                origin = urlparse(url).netloc
                await page.route(re.compile(r"^https?://"),
                                 lambda route: route.continue_() if urlparse(route.request.url).netloc == origin else route.abort())
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=left_ms())
            except (PWTimeout, PWError) as exc:
                return Capture(status="error", reason=f"navigation_failed: {str(exc).splitlines()[0][:160]}")
            status = resp.status if resp else None
            cap = Capture(status="ok", final_url=page.url, http_status=status)
            if status in BLOCK_STATUSES:
                cap.status, cap.reason = "blocked", f"http_{status}"
                return cap
            if status is not None and status >= 400:
                return Capture(status="error", reason=f"http_{status}", final_url=page.url, http_status=status)
            if not url.startswith("file:"):
                await _step(page.wait_for_load_state("networkidle"), left_ms(8000))  # many shops never go idle

            # Scroll so lazy-loaded reviews appear, within the budget.
            last_height = 0
            for _ in range(8):
                if time.monotonic() > deadline - 6:
                    break
                got = await _step(page.evaluate(
                    "() => { window.scrollBy(0, 1200); return [document.documentElement.scrollHeight,"
                    " window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2]; }"), 2000)
                await asyncio.sleep(0.25)
                if got and got[1] and got[0] == last_height:
                    break  # bottom reached and nothing new loaded
                last_height = got[0] if got else last_height
            await _step(page.evaluate("window.scrollTo(0, 0)"), 1000)

            sig = await _step(page.evaluate(_SIGNALS_JS), 3000) or {
                "captcha_visible": False, "password_visible": False, "path": "", "scroll_height": 2000}
            text = await _step(page.inner_text("body"), left_ms(5000))
            if text is None:
                return Capture(status="error", reason="page_unresponsive", final_url=page.url, http_status=status)
            reason = detect_block(status, text, sig["captcha_visible"], sig["password_visible"], sig["path"])
            if reason:
                cap.status, cap.reason = "blocked", reason
                return cap

            cap.text = text
            cap.image_urls = list(dict.fromkeys(await _step(page.evaluate(_IMG_JS), 3000) or []))[:10]
            cap.json_ld = _flatten_ld(await _step(page.evaluate(_LD_JS), 3000) or [])
            height = min(int(sig["scroll_height"]), MAX_SHOT_HEIGHT)
            cap.full_page_jpeg = await _step(page.screenshot(
                full_page=True, type="jpeg", quality=60,
                clip={"x": 0, "y": 0, "width": VIEWPORT["width"], "height": max(height, 100)}, timeout=left_ms(5000)), left_ms(6000))
            await _open_size_chart(page, cap, left_ms)
            cap.html = await _step(page.content(), left_ms(4000)) or ""
            return cap
        except (PWTimeout, PWError) as exc:
            return Capture(status="error", reason=f"capture_failed: {str(exc).splitlines()[0][:160]}")
        finally:
            # Only this visit's context is torn down; the shared browser stays up for the next one.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(context.close(), timeout=3)


class _BrowserHost:
    """Owns one headless Chromium on one long-lived event loop, in its own daemon thread.

    Launching Playwright's driver and a Chromium process per request cost about 6.7s of the
    roughly 9s each cache-missing /read call spent in capture -- measured, and far more than the
    page work itself (~2.2s). That overhead, on top of the extraction model call, pushed a first
    read past the browser's own request timeout even though the page was trivial to render.

    The browser process is shared; each capture still gets its own fresh context, so visits stay
    isolated from one another. A dedicated thread with a persistent loop is what makes reuse
    possible at all: Playwright objects belong to the loop that created them, and capture_page is
    a synchronous call (pipeline.py runs it via asyncio.to_thread), so there is no caller loop to
    attach to. Captures are submitted to that loop instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._playwright: Any = None
        self._browser: Any = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="reader-browser", daemon=True).start()
            self._loop = loop
        return self._loop

    async def _ensure_browser(self) -> Any:
        from playwright.async_api import async_playwright

        if self._browser is not None and self._browser.is_connected():
            return self._browser
        # First use, or the browser died (crash, external kill): start a fresh one.
        self._browser = None
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        return self._browser

    def capture(self, url: str, budget_s: float) -> Capture:
        loop = self._ensure_loop()

        async def run() -> Capture:
            browser = await self._ensure_browser()
            return await asyncio.wait_for(_capture(url, budget_s, browser), timeout=budget_s + 2)

        with self._lock:  # serialize launches, and keep one page at a time as before
            future = asyncio.run_coroutine_threadsafe(run(), loop)
            try:
                return future.result(timeout=budget_s + 8)
            except (asyncio.TimeoutError, TimeoutError):
                future.cancel()
                return Capture(status="error", reason="timeout")

    def warm(self, timeout_s: float = 60.0) -> None:
        loop = self._ensure_loop()
        with self._lock:
            asyncio.run_coroutine_threadsafe(self._ensure_browser(), loop).result(timeout=timeout_s)

    def close(self) -> None:
        loop, self._loop = self._loop, None
        browser, self._browser = self._browser, None
        playwright, self._playwright = self._playwright, None
        if loop is None:
            return

        async def shutdown() -> None:
            with contextlib.suppress(Exception):
                if browser is not None:
                    await browser.close()
            with contextlib.suppress(Exception):
                if playwright is not None:
                    await playwright.stop()

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)


_host = _BrowserHost()
atexit.register(_host.close)


def capture_page(url: str, budget_s: float = BUDGET_S) -> Capture:
    """Visit the URL once. Never runs longer than budget_s plus a short cleanup, even if the page hangs."""
    return _host.capture(url, budget_s)


def warm_browser(timeout_s: float = 60.0) -> None:
    """Launch the shared browser now, so the first read does not pay for it."""
    _host.warm(timeout_s)


def close_browser() -> None:
    """Shut the shared browser and its loop down (service shutdown, and tests)."""
    _host.close()


async def _open_size_chart(page: Any, cap: Capture, left_ms: Any) -> None:
    """Click the first visible element whose text looks like a size chart link, then screenshot the modal."""
    try:
        candidates = (await _step(page.get_by_text(SIZE_CHART_RE).all(), 3000)) or []
        target = None
        for c in candidates[:15]:
            if await _step(c.is_visible(), 1000):
                target = c
                break
        if target is not None:
            await target.click(timeout=left_ms(2500))
            cap.chart_opened = True
            await asyncio.sleep(min(1.2, left_ms() / 1000))
            modal = None
            for m in (await _step(page.locator(_MODAL_SEL).all(), 2000)) or []:
                if await _step(m.is_visible(), 1000):
                    modal = m
                    break
            if modal is not None:
                cap.chart_png = await _step(modal.screenshot(type="png", timeout=left_ms(3000)), left_ms(4000))
                inner = await _step(modal.inner_text(timeout=left_ms(2000)), left_ms(3000))
                if inner:
                    cap.text += "\n\nSize chart (opened):\n" + inner
            else:
                cap.chart_png = await _step(page.screenshot(type="png", timeout=left_ms(3000)), left_ms(4000))
    except Exception:  # failing to open the chart is not fatal; the page may still hold it
        pass
    if "Size chart (opened):" not in cap.text:
        # The chart panel can be in the delivered page but hidden (nothing to click, or a script we did not run).
        panels = await _step(page.evaluate(_HIDDEN_PANELS_JS), 3000) or []
        if panels:
            cap.text += "\n\nSize chart (hidden panel in page):\n" + "\n---\n".join(panels)
