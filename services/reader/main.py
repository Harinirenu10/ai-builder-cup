"""Reader Service: Product page fetching and size chart reading via Playwright."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fitcore.schemas import Product, SizeChart

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
