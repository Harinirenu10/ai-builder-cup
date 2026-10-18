"""Reader Service: Product page fetching and size chart reading via Playwright."""

import sys
from pathlib import Path

# Same two-launch-method problem as services/api/main.py: `uvicorn main:app`
# from this directory (the Dockerfile) vs. `uvicorn services.reader.main:app`
# from the repo root (infra/Makefile's `dev` target). Only the first puts
# this directory on sys.path, which `readerlib.*` needs.
_READER_DIR = Path(__file__).resolve().parent
if str(_READER_DIR) not in sys.path:
    sys.path.insert(0, str(_READER_DIR))

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import HTTPException

from readerlib.cache import ReadCache
from readerlib.capture import close_browser, warm_browser
from readerlib.models import ReadImageRequest, ReadRequest, ReadResponse
from readerlib.pipeline import read_image, read_url

logger = logging.getLogger("reader.main")

# uvicorn's own logging config never touches the root logger, so without this every
# logger.info in this service (and in readerlib) is silently dropped -- the same gap that
# hid the geometry worker's progress logs (see services/api/main.py).
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start Chromium while the service boots, not inside the first shopper's request.

    readerlib.capture reuses one browser process across reads, but launching it still costs
    several seconds, and that cost would otherwise land entirely on whoever asks first -- enough
    on its own to push a first read past the caller's timeout. Warming here moves it into
    startup. Failures are deliberately non-fatal: the browser is launched lazily on first use
    anyway, so a warm-up problem must not stop the service from starting.
    """
    try:
        await asyncio.to_thread(warm_browser)
        logger.info("Reader browser warmed and ready.")
    except Exception as exc:
        logger.warning("Reader browser warm-up failed (will launch on first read): %s", exc)
    try:
        yield
    finally:
        await asyncio.to_thread(close_browser)


app = FastAPI(
    title="Measured Fit Reader Service",
    description="Playwright-powered ecommerce product page and size chart extractor",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "reader"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit Reader service online"}


_cache: ReadCache | None = None


def get_cache() -> ReadCache:
    global _cache
    if _cache is None:
        _cache = ReadCache()
    return _cache


@app.post("/read", response_model=ReadResponse)
async def read(req: ReadRequest) -> ReadResponse:
    """Read a product page. status is ok, blocked (captcha, login wall, 401/403/429), unsupported or error.

    Results are cached by sha256(url) in products/{hash} for 7 days, blocked results included.
    """
    resp = await read_url(req.url, get_cache())
    if resp.status == "error":
        raise HTTPException(status_code=502 if not (resp.reason or "").startswith(("Only http", "Internal", "Cannot resolve", "The host")) else 422,
                            detail=resp.reason)
    return resp


@app.post("/read-image", response_model=ReadResponse)
async def read_image_endpoint(req: ReadImageRequest) -> ReadResponse:
    """Product from a size-chart screenshot, for sites that block the reader."""
    resp = await read_image(req.image, req.page_url)
    if resp.status == "error":
        raise HTTPException(status_code=422 if resp.reason == "not a valid image" else 502, detail=resp.reason)
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
