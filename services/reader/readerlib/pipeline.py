"""URL or image in, Product and ReviewSignals out. Cache first; one visit per URL."""

import asyncio
import base64
import hashlib
import io
import os
from typing import Any, Callable
from urllib.parse import urlparse

from fitcore.read.extract import extract
from fitcore.read.normalize import assemble
from fitcore.read.tables import parse_size_chart_html
from fitcore.schemas import Product, ReviewSignals
from readerlib.cache import ReadCache, url_key
from readerlib.capture import Capture, capture_page
from readerlib.models import ReadResponse
from readerlib.security import UnsafeURL, validate_url

HARD_LIMIT_S = 25.0
SPARSE_TEXT_CHARS = 200
# The extraction model call was the one unbounded step in a read: capture is held to
# HARD_LIMIT_S, but a slow or degraded model could run on past the caller's own budget (the API
# gives the reader READER_READ_TIMEOUT_S = 35s), turning a read that would have succeeded into a
# 504 at an arbitrary point. Bounding it keeps a read's worst case predictable and under that.
# 25s leaves room for a typical ~4s capture inside the API's 35s budget while still letting a
# slow-but-successful model call finish: a tighter bound turned those into hard failures.
EXTRACT_TIMEOUT_S = float(os.environ.get("READER_EXTRACT_TIMEOUT_S", "25.0"))


def _json_ld_hints(cap: Capture) -> list[dict]:
    return cap.json_ld[:3]


def _fill_chart_from_html(ext, cap: Capture) -> bool:
    """Take the size chart from the page's own HTML table when the model did not return one.

    The chart is the one part of an extraction that is often already machine readable, and the
    one a small local model drops most -- measured on the demo store's shirt, whose chart text is
    captured verbatim yet came back in roughly one attempt in three. Reading the table directly
    makes a structured chart deterministic; the model is still what reads category, fabric and
    reviews. Only filled when the model returned nothing, so a chart it did transcribe is never
    second-guessed, and only as a whole (sizes with their own unit and chart-type evidence) so
    numbers and units can never come from two different tables.
    """
    if ext.product.sizes or not cap.html:
        return False
    parsed = parse_size_chart_html(cap.html, ext.product.category)
    if parsed is None:
        return False
    ext.product.sizes = parsed.sizes
    if parsed.unit != "unknown":
        ext.product.unit = parsed.unit
    if parsed.chart_type != "unknown":
        ext.product.chart_type = parsed.chart_type
        ext.product.chart_type_evidence = parsed.chart_type_evidence
    return True


def _extract_from_capture(cap: Capture, url: str, client: Any, ollama_transport: Any) -> tuple[Any, bool]:
    images = [b for b in (cap.chart_png,) if b]
    if len(cap.text.strip()) < SPARSE_TEXT_CHARS and cap.full_page_jpeg:
        images.append(cap.full_page_jpeg)  # text-light page: let the model read the screenshot
    ext = extract(cap.text, _json_ld_hints(cap), images, url, client=client,
                  ollama_transport=ollama_transport, timeout=EXTRACT_TIMEOUT_S)
    return ext, _fill_chart_from_html(ext, cap)


def _response_from_outcome(outcome, url: str | None) -> ReadResponse:
    if outcome.status == "unsupported":
        return ReadResponse(status="unsupported", reason="Not a supported garment category", url=url, flags=outcome.flags)
    return ReadResponse(status="ok", url=url, product=outcome.product, reviews=outcome.reviews, flags=outcome.flags)


async def read_url(url: str, cache: ReadCache, *, client: Any = None, ollama_transport: Any = None,
                   capture_fn: Callable[[str], Capture] = capture_page) -> ReadResponse:
    try:
        url = validate_url(url)
    except UnsafeURL as exc:
        return ReadResponse(status="error", reason=str(exc), url=url)

    hit = cache.get(url)
    if hit:
        return ReadResponse.model_validate({**hit["response"], "cached": True})

    try:
        cap = await asyncio.wait_for(asyncio.to_thread(capture_fn, url), timeout=HARD_LIMIT_S)
    except asyncio.TimeoutError:
        return ReadResponse(status="error", reason="timeout", url=url)  # not cached: may be transient

    if cap.status == "blocked":
        resp = ReadResponse(status="blocked", reason=cap.reason, url=url)
        cache.put(url, {"response": resp.model_dump(mode="json")})  # do not visit again for 7 days
        return resp
    if cap.status == "error":
        return ReadResponse(status="error", reason=cap.reason, url=url)

    try:
        ext, chart_from_html = await asyncio.to_thread(_extract_from_capture, cap, url, client, ollama_transport)
    except Exception as exc:
        return ReadResponse(status="error", reason=f"extraction_failed: {type(exc).__name__}", url=url)
    host = urlparse(url).hostname or "unknown"
    outcome = assemble(ext, url=url, product_id=url_key(url)[:16], fallback_brand=host, image_urls=cap.image_urls)
    resp = _response_from_outcome(outcome, url)
    if chart_from_html:
        # Provenance: this chart was read from the page's HTML table, not transcribed by the model.
        resp.flags = [*(resp.flags or []), "size_chart_from_html"]

    if outcome.status == "unsupported" and (chart_from_html or not ext.product.title.strip()):
        # "unsupported" is meant to record that a page is genuinely not a garment, and it is
        # cached for 7 days like any other result. An extraction that came back with no category
        # at all is a different thing -- the model failed, the page did not change -- and a page
        # whose own garment size chart we just parsed is certainly a garment. Caching that would
        # pin a transient model failure to the URL for a week, which is exactly how the demo
        # store's shirt could end up permanently unreadable. Report it as the retryable error it
        # is instead, and leave the cache alone.
        return ReadResponse(status="error", reason="extraction_incomplete: no category", url=url, flags=resp.flags)
    cache.put(url, {"response": resp.model_dump(mode="json")})
    return resp


def _checked_image(b64: str) -> bytes:
    from PIL import Image

    data = base64.b64decode(b64)
    Image.open(io.BytesIO(data)).verify()  # raises if it is not an image
    return data


async def read_image(image_b64: str, page_url: str | None = None, *, client: Any = None,
                     ollama_transport: Any = None) -> ReadResponse:
    """Product from a size-chart screenshot: the fallback when a site blocks the reader."""
    try:
        data = _checked_image(image_b64)
    except Exception:
        return ReadResponse(status="error", reason="not a valid image", url=page_url)
    try:
        ext = await asyncio.to_thread(
            extract, "Size chart screenshot supplied by the shopper. Read the chart from the image.",
            None, [data], page_url, client, ollama_transport)
    except Exception as exc:
        return ReadResponse(status="error", reason=f"extraction_failed: {type(exc).__name__}", url=page_url)
    pid = hashlib.sha256(data).hexdigest()[:16]
    outcome = assemble(ext, url=page_url, product_id=pid, fallback_brand=(urlparse(page_url).hostname if page_url else None) or "unknown",
                       with_reviews=False)
    resp = _response_from_outcome(outcome, page_url)
    resp.reviews = None
    return resp
