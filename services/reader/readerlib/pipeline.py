"""URL or image in, Product and ReviewSignals out. Cache first; one visit per URL."""

import asyncio
import base64
import hashlib
import io
from typing import Any, Callable
from urllib.parse import urlparse

from fitcore.read.extract import extract
from fitcore.read.normalize import assemble
from fitcore.schemas import Product, ReviewSignals
from readerlib.cache import ReadCache, url_key
from readerlib.capture import Capture, capture_page
from readerlib.models import ReadResponse
from readerlib.security import UnsafeURL, validate_url

HARD_LIMIT_S = 25.0
SPARSE_TEXT_CHARS = 200


def _json_ld_hints(cap: Capture) -> list[dict]:
    return cap.json_ld[:3]


def _extract_from_capture(cap: Capture, url: str, client: Any, ollama_transport: Any):
    images = [b for b in (cap.chart_png,) if b]
    if len(cap.text.strip()) < SPARSE_TEXT_CHARS and cap.full_page_jpeg:
        images.append(cap.full_page_jpeg)  # text-light page: let the model read the screenshot
    return extract(cap.text, _json_ld_hints(cap), images, url, client=client, ollama_transport=ollama_transport)


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
        ext = await asyncio.to_thread(_extract_from_capture, cap, url, client, ollama_transport)
    except Exception as exc:
        return ReadResponse(status="error", reason=f"extraction_failed: {type(exc).__name__}", url=url)
    host = urlparse(url).hostname or "unknown"
    outcome = assemble(ext, url=url, product_id=url_key(url)[:16], fallback_brand=host, image_urls=cap.image_urls)
    resp = _response_from_outcome(outcome, url)
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
