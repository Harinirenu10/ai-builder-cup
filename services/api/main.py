"""API Service: Customer and Business endpoints, ADK planner orchestration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fitcore.schemas import FitResult, Product, MeasurementZone

app = FastAPI(
    title="Measured Fit API Service",
    description="Customer & Business endpoints, ADK Planner Agent, and Cloud Tasks orchestration",
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
    return {"status": "ok", "service": "api"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit API service online"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
