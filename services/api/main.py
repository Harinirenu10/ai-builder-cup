"""API Service: Customer and Business endpoints, ADK planner orchestration."""

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field

from apilib.cors import DynamicCORS
from apilib.fitquery import router as fit_query_router
from apilib.gifts import router as gifts_router
from fitcore.fit import apply_reviews, evaluate, explain, tailor_card
from fitcore.fit.verdict import business_verdict
from fitcore.schemas import (
    BizVerdict,
    FitPreferences,
    FitResult,
    Measurements,
    PoseDeltas,
    Product,
    ReviewSignal,
    TailorCard,
)

app = FastAPI(
    title="Measured Fit API Service",
    description="Customer & Business endpoints, ADK Planner Agent, and Cloud Tasks orchestration",
    version="0.1.0",
)

app.add_middleware(DynamicCORS)

# New business-facing routes live under /v1 and, for the Firebase Hosting rewrite (/api/**), under /api/v1.
for _prefix in ("/v1", "/api/v1"):
    app.include_router(fit_query_router, prefix=_prefix)
    app.include_router(gifts_router, prefix=_prefix)
app.include_router(gifts_router)  # the gift paths also work without a prefix: POST /gifts, GET /gifts/{id}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "api"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit API service online"}


class FitRequest(BaseModel):
    """Inputs for one fit evaluation. The twin is sent by its owner; Firestore lookup by twin_id comes later."""
    product: Product
    twin: Measurements
    review_signals: list[ReviewSignal] = Field(default_factory=list)
    preferences: FitPreferences | None = None
    pose_deltas: PoseDeltas | None = None


class FitResponse(BaseModel):
    result: FitResult
    review_notes: list[str] = Field(default_factory=list)
    explanation: str | None = None


def _evaluate(req: FitRequest) -> tuple[FitResult, list[str]]:
    adjustments, notes = (None, [])
    if req.review_signals:
        adjustments, notes = apply_reviews(req.review_signals, req.twin)
    try:
        return evaluate(req.twin, req.product, req.preferences, adjustments, req.pose_deltas), notes
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/v1/fit", response_model=FitResponse)
def fit(req: FitRequest, explain_text: bool = False, language: str = "en") -> FitResponse:
    """Customer endpoint: full fit result for the owner. Set explain_text=true for a Gemini/Gemma explanation."""
    result, notes = _evaluate(req)
    text = explain(result, language) if explain_text else None
    return FitResponse(result=result, review_notes=notes, explanation=text)


@app.post("/v1/biz/verdict", response_model=BizVerdict)
def biz_verdict(req: FitRequest) -> BizVerdict:
    """Business endpoint: verdicts only. Never returns measurements, ease values or cm figures."""
    result, _ = _evaluate(req)
    return business_verdict(result)


@app.post("/v1/tailor-card", response_model=TailorCard)
def tailor(req: FitRequest, languages: str = "en") -> TailorCard:
    """Customer endpoint: tailor card JSON. languages is comma separated, e.g. en,ta."""
    result, _ = _evaluate(req)
    if result.recommended_size is None:
        raise HTTPException(status_code=409, detail="No size fits, so there is no tailor card.")
    return tailor_card(result, req.twin, [l.strip() for l in languages.split(",") if l.strip()])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
