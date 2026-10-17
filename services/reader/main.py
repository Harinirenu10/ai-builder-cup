"""Reader Service: Product page fetching and size chart reading via Playwright."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import HTTPException

from readerlib.cache import ReadCache
from readerlib.models import ReadImageRequest, ReadRequest, ReadResponse
from readerlib.pipeline import read_image, read_url

app = FastAPI(
    title="Measured Fit Reader Service",
    description="Playwright-powered ecommerce product page and size chart extractor",
    version="0.1.0",
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
