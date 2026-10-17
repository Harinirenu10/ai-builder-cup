"""API Service: Customer and Business endpoints, ADK planner orchestration."""

import datetime
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from apilib.charts import router as charts_router
from apilib.cors import DynamicCORS
from apilib.demo import router as demo_router
from apilib.fitquery import router as fit_query_router
from apilib.gifts import router as gifts_router
from apilib.me import router as me_router
from apilib.orders import router as orders_router
from apilib.returns import router as returns_router
from apilib.simulate import router as simulate_router
from fitcore.fit import apply_reviews, evaluate, explain, tailor_card
from fitcore.fit.verdict import business_verdict
from fitcore.schemas import (
    BizVerdict,
    BodyTwinJobRequest,
    FitPreferences,
    FitResult,
    JobStatus,
    Measurements,
    MeasurementZone,
    PoseDeltas,
    Product,
    ReviewSignal,
    TailorCard,
)
try:
    from services.geometry.cloud_client import get_firestore_client, get_storage_client
except ImportError as _cc_exc:  # the API image ships services/api only; the twin-job routes then answer 503
    _cloud_client_error = _cc_exc

    def get_firestore_client() -> Any:  # type: ignore[misc]
        raise HTTPException(status_code=503, detail=f"Twin jobs unavailable: {_cloud_client_error}")

    get_storage_client = get_firestore_client  # type: ignore[assignment]

try:
    from services.geometry.worker import JobNotFoundError, process_twin_job
except ImportError as _exc:  # geometry deps (smplx, mediapipe) are not installed in the lightweight API image/tests
    _geometry_import_error = _exc

    class JobNotFoundError(Exception):  # type: ignore[no-redef]
        pass

    def process_twin_job(*args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
        raise HTTPException(status_code=503, detail=f"Geometry pipeline unavailable: {_geometry_import_error}")


logger = logging.getLogger("api.main")

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
    app.include_router(orders_router, prefix=_prefix)
    app.include_router(returns_router, prefix=_prefix)
    app.include_router(charts_router, prefix=_prefix)
    app.include_router(simulate_router, prefix=_prefix)
    app.include_router(demo_router, prefix=_prefix)
    app.include_router(me_router, prefix=_prefix)
app.include_router(gifts_router)  # the gift paths also work without a prefix: POST /gifts, GET /gifts/{id}


@app.get("/api/healthz")  # reachable through the Firebase Hosting /api/** rewrite
@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "api"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit API service online"}


@app.post("/api/twin/jobs")
@app.post("/twin/jobs")
def create_twin_job_endpoint(request: BodyTwinJobRequest) -> dict[str, Any]:
    """Create a 3D body twin reconstruction job and trigger processing pipeline."""
    if not request.age_18_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Age confirmation (18+) is required prior to body photo capture and processing.",
        )

    # Validate that all 4 photos are present and non-empty
    for view, photo_val in (
        ("front", request.photo_front),
        ("back", request.photo_back),
        ("left", request.photo_left),
        ("right", request.photo_right),
    ):
        if not photo_val or not str(photo_val).strip():
            raise HTTPException(
                status_code=422,
                detail=f"Missing or empty photo for required view: '{view}'.",
            )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fs = get_firestore_client()

    job_data = {
        "id": job_id,
        "user_id": request.user_id,
        "status": JobStatus.checking.value,
        "height_cm": request.height_cm,
        "gender": request.gender,
        "age_18_confirmed": request.age_18_confirmed,
        "opt_in_keep_front_photo": request.opt_in_keep_front_photo,
        "photos": {
            "front": request.photo_front,
            "back": request.photo_back,
            "left": request.photo_left,
            "right": request.photo_right,
        },
        "created_at": now,
        "updated_at": now,
    }
    fs.update_job(job_id, job_data)

    try:
        result = process_twin_job(job_id=job_id, firestore_client=fs)
        twin_id = result.get("twin_id")
        twin_data = fs.get_twin(twin_id) if twin_id else None
        return {
            "job_id": job_id,
            "status": result.get("status", JobStatus.done.value),
            "twin_id": twin_id,
            "measurements": twin_data.get("measurements") if twin_data else None,
            "glb_url": f"/api/twin/{twin_id}/glb" if twin_data and twin_data.get("glb_storage_url") else None,
            "final_iou": (twin_data.get("final_mean_iou") if twin_data else None) or result.get("final_mean_iou"),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        }
    except Exception as exc:
        logger.exception("Error processing twin job %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to process twin job: {exc}")


@app.get("/api/twin/jobs/{job_id}")
@app.get("/twin/jobs/{job_id}")
def get_twin_job_endpoint(job_id: str) -> dict[str, Any]:
    """Retrieve job document and processing status."""
    fs = get_firestore_client()
    job = fs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.get("/api/twin/{twin_id}")
@app.get("/twin/{twin_id}")
def get_twin_endpoint(twin_id: str) -> dict[str, Any]:
    """Retrieve fitted 3D twin document by ID."""
    fs = get_firestore_client()
    twin = fs.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    return twin


@app.get("/api/twin/{twin_id}/glb")
@app.get("/twin/{twin_id}/glb")
def get_twin_glb_endpoint(twin_id: str) -> Response:
    """Stream the fitted 3D twin's GLB mesh bytes for browser rendering (e.g. <model-viewer>).

    The twin document stores a gs:// Cloud Storage URI, which a browser cannot fetch
    directly, so this endpoint downloads the bytes server-side and streams them back.
    """
    fs = get_firestore_client()
    twin = fs.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")

    storage_url = twin.get("glb_storage_url")
    if not storage_url:
        raise HTTPException(status_code=404, detail=f"No GLB asset available for twin '{twin_id}'.")

    st = get_storage_client()
    try:
        glb_bytes = st.download_bytes(storage_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load GLB asset: {exc}")

    return Response(content=glb_bytes, media_type="model/gltf-binary")


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
