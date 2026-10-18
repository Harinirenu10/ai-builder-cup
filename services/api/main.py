"""API Service: Customer and Business endpoints, ADK planner orchestration."""

import datetime
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# This module is imported two different ways depending on how the service is
# launched: `uvicorn main:app` with this directory as the working directory
# (the Dockerfile), and `uvicorn services.api.main:app` from the repo root
# (infra/Makefile's `dev` target). The first makes `apilib.*` resolve but not
# `services.*` (needed by this file and by apilib/fit_mesh.py and
# garment_routes.py, which import services.geometry.cloud_client directly);
# the second is the other way around. Putting this file's own directory AND
# the repo root on sys.path makes both `apilib.*` and `services.*` resolve
# under either launch method, without needing a PYTHONPATH that isn't set by
# either documented launch command.
_API_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent.parent
for _p in (_API_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from apilib.auth import current_uid, err, validate_security_config
from apilib.calendar_auth import router as calendar_router
from apilib.charts import router as charts_router
from apilib.cors import DynamicCORS
from apilib.fit_mesh import _get_owned_twin, router as fit_mesh_router
from apilib.fitquery import router as fit_query_router
from apilib.gifts import router as gifts_router
from apilib.orders import router as orders_router
from apilib.planner import router as planner_router
from apilib.returns import router as returns_router
from apilib.simulate import router as simulate_router
from apilib.wardrobe import router as wardrobe_router
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
from garment_routes import router as garment_router
from services.geometry.cloud_client import get_firestore_client, get_storage_client

try:
    from services.geometry.worker import JobNotFoundError, process_twin_job
except ImportError as _exc:  # geometry deps (smplx, mediapipe) are not installed in the lightweight API image/tests
    _geometry_import_error = _exc

    class JobNotFoundError(Exception):  # type: ignore[no-redef]
        pass

    def process_twin_job(*args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
        raise HTTPException(status_code=503, detail=f"Geometry pipeline unavailable: {_geometry_import_error}")


logger = logging.getLogger("api.main")

validate_security_config()  # fail startup, not the first request, on an unsafe auth config (see apilib/auth.py)

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
    app.include_router(wardrobe_router, prefix=_prefix)
    app.include_router(planner_router, prefix=_prefix)
    app.include_router(calendar_router, prefix=_prefix)
app.include_router(gifts_router)  # the gift paths also work without a prefix: POST /gifts, GET /gifts/{id}
app.include_router(wardrobe_router)
app.include_router(planner_router)
app.include_router(calendar_router)

app.include_router(garment_router)
app.include_router(fit_mesh_router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "api"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit API service online"}


def _get_owned_job(job_id: str, uid: str, fs: Any) -> dict[str, Any]:
    """Mirrors apilib/fit_mesh.py::_get_owned_twin for job documents (jobs/{job_id}): 404 if the
    job does not exist, 403 if it exists but belongs to a different user."""
    job = fs.get_job(job_id)
    if not job:
        raise err(404, "job_not_found", f"Job '{job_id}' not found.")
    if job.get("user_id") != uid:
        raise err(403, "forbidden", "This job belongs to another user.")
    return job


def _run_twin_job_in_background(job_id: str, fs: Any) -> None:
    """Runs the actual pipeline off the request/response cycle (FastAPI's built-in
    BackgroundTasks -- no Cloud Tasks or other new infrastructure). The full pipeline
    (photo validation, MediaPipe, SMPL fitting, critic refinement) can run well past any
    reasonable HTTP client timeout; create_twin_job_endpoint now returns as soon as the job
    document exists, and the caller polls GET /api/twin/jobs/{job_id} for progress.

    process_twin_job (services/geometry/worker.py) already self-reports every outcome --
    done/failed/needs_retake/rejected -- onto the job document via fs.update_job, including on
    exception (its own top-level try/except), so the polling endpoint sees the result without
    this function needing to return anything. The try/except here is a defensive backstop only,
    for a failure before process_twin_job's own handler can run (e.g. an import-time error) --
    without it, such a failure would otherwise leave the job stuck at "checking" forever, since
    there is no HTTP caller left to report an error to once this runs in the background.
    """
    existing = fs.get_job(job_id)
    if existing and existing.get("status") != JobStatus.checking.value:
        # Defensive guard: a job past its initial "checking" status has either already finished
        # or is already being processed. Running process_twin_job a second time for the same
        # job_id would redundantly re-run the whole pipeline and could overwrite a result that's
        # already there, so skip rather than process it twice.
        logger.warning(
            "Skipping duplicate processing for job %s: status is already %s",
            job_id,
            existing.get("status"),
        )
        return

    try:
        process_twin_job(job_id=job_id, firestore_client=fs)
    except Exception as exc:
        logger.exception("Unhandled error in background twin job %s: %s", job_id, exc)
        try:
            fs.update_job(
                job_id,
                {
                    "status": JobStatus.failed.value,
                    "error_code": "PROCESSING_ERROR",
                    "error_message": str(exc),
                    "current_step": "failed",
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
            )
        except Exception:
            logger.exception("Could not record failure for job %s after an unhandled error", job_id)


@app.post("/api/twin/jobs")
@app.post("/twin/jobs")
def create_twin_job_endpoint(
    request: BodyTwinJobRequest,
    background_tasks: BackgroundTasks,
    uid: str = Depends(current_uid),
) -> dict[str, Any]:
    """Create a 3D body twin reconstruction job and schedule processing in the background.

    The job is always created for the authenticated caller. `request.user_id` is accepted for
    schema compatibility but never used to decide ownership -- a signed-in caller cannot create
    (or attribute) a job under a different user_id by putting one in the request body.

    This returns immediately once the job document exists (status="checking"), rather than
    waiting for the full pipeline to finish -- the pipeline (photo validation through SMPL
    fitting and critic refinement) can take well over a minute, longer than a typical HTTP
    client timeout. Poll GET /api/twin/jobs/{job_id} for live status and the final result.
    """
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
        "user_id": uid,  # the authenticated caller, never request.user_id (see docstring above)
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

    # job_id is freshly generated above, so there is nothing already running for it to collide
    # with here; _run_twin_job_in_background is still scheduled exactly once per request.
    background_tasks.add_task(_run_twin_job_in_background, job_id, fs)

    return {
        "job_id": job_id,
        "status": JobStatus.checking.value,
        "twin_id": None,
        "measurements": None,
        "glb_url": None,
        "final_iou": None,
        "error_code": None,
        "error_message": None,
    }


@app.get("/api/twin/jobs/{job_id}")
@app.get("/twin/jobs/{job_id}")
def get_twin_job_endpoint(job_id: str, uid: str = Depends(current_uid)) -> dict[str, Any]:
    """Retrieve job document and processing status. The caller must be signed in and must be
    the job's own user_id (see _get_owned_job) -- job documents hold the same private
    measurements/status as the twin they produce, so this needs the same ownership check."""
    fs = get_firestore_client()
    return _get_owned_job(job_id, uid, fs)


@app.get("/api/twin/{twin_id}")
@app.get("/twin/{twin_id}")
def get_twin_endpoint(twin_id: str, uid: str = Depends(current_uid)) -> dict[str, Any]:
    """Retrieve fitted 3D twin document by ID. The caller must be signed in and must own this
    twin (apilib/fit_mesh.py::_get_owned_twin -- the same ownership check already used by the
    fit-mesh/pose-deltas endpoints, reused here rather than re-implemented)."""
    fs = get_firestore_client()
    return _get_owned_twin(twin_id, uid, fs)


@app.get("/api/twin/{twin_id}/glb")
@app.get("/twin/{twin_id}/glb")
def get_twin_glb_endpoint(twin_id: str, uid: str = Depends(current_uid)) -> Response:
    """Stream the fitted 3D twin's GLB mesh bytes for browser rendering (e.g. <model-viewer>).

    The twin document stores a gs:// Cloud Storage URI, which a browser cannot fetch
    directly, so this endpoint downloads the bytes server-side and streams them back.
    Requires the caller to own this twin, same as get_twin_endpoint above.
    """
    fs = get_firestore_client()
    twin = _get_owned_twin(twin_id, uid, fs)

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
    """Inputs for one fit evaluation, for the signed-in caller's OWN body twin.

    There is deliberately no `twin` field here. It used to take measurements directly in the
    body ("sent by its owner" was the stated intent, never enforced); a signed-in caller could
    submit anyone's measurements and get a result for them. The twin is now always looked up
    server-side by the authenticated uid (see _require_own_twin), reusing the same Store-then-
    Firestore lookup apilib/fitquery.py::evaluate_for_uid already uses for /v1/fit-query.
    """
    product: Product
    review_signals: list[ReviewSignal] = Field(default_factory=list)
    preferences: FitPreferences | None = None
    pose_deltas: PoseDeltas | None = None


class FitResponse(BaseModel):
    result: FitResult
    review_notes: list[str] = Field(default_factory=list)
    explanation: str | None = None


def _require_own_twin(uid: str) -> Measurements:
    """The authenticated caller's own body twin, or a safe 409 -- never the caller's own say-so."""
    from apilib.fitquery import resolve_twin_measurements_for_uid
    from apilib.stores import get_store

    twin = resolve_twin_measurements_for_uid(get_store(), uid)
    if twin is None:
        raise HTTPException(status_code=409, detail={"code": "no_twin", "message": "You have not created a body twin yet."})
    return twin


@app.get("/v1/me/twin")
@app.get("/api/v1/me/twin")
@app.get("/api/me/twin")
def get_my_twin(uid: str = Depends(current_uid)) -> dict[str, Any]:
    """Retrieve the signed-in user's latest body twin, or 404 if none exists."""
    fs = get_firestore_client()
    latest = fs.get_latest_twin_for_user(uid)
    if latest:
        t_id = latest.get("id") or latest.get("twin_id") or "twin_latest"
        return {
            "twin_id": t_id,
            "user_id": uid,
            "measurements": latest.get("measurements"),
            "glb_url": f"/api/twin/{t_id}/glb" if latest.get("glb_storage_url") else None,
            "final_iou": latest.get("final_mean_iou"),
        }
    from apilib.stores import get_store
    store_twin = get_store().get("twins", uid)
    if store_twin and store_twin.get("measurements"):
        return {
            "twin_id": f"twin_{uid}",
            "user_id": uid,
            "measurements": store_twin.get("measurements"),
            "glb_url": None,
            "final_iou": 0.95,
        }
    raise HTTPException(status_code=404, detail={"code": "no_twin", "message": "You have not created a body twin yet."})
 
 
class CustomerProgress(BaseModel):
    """Customer portal progress and draft state. Never contains private measurements or raw photos."""
    active_tab: str = "twin"
    height_cm: str = "175.0"
    gender: str = "neutral"
    age_18_confirmed: bool = False
    opt_in_keep_front_photo: bool = False
    captured_views: list[str] = Field(default_factory=list)
    last_job_id: str | None = None
    updated_at: str | None = None


@app.get("/v1/me/progress")
@app.get("/api/v1/me/progress")
@app.get("/api/me/progress")
def get_my_progress(uid: str = Depends(current_uid)) -> dict[str, Any]:
    """Retrieve the authenticated caller's own customer progress state."""
    from apilib.stores import get_store
    store = get_store()
    prog = store.get("progress", uid)
    if prog:
        return prog
    fs = get_firestore_client()
    if hasattr(fs, "get_user_progress"):
        fs_prog = fs.get_user_progress(uid)
        if fs_prog:
            return fs_prog
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "user_id": uid,
        "active_tab": "twin",
        "height_cm": "175.0",
        "gender": "neutral",
        "age_18_confirmed": False,
        "opt_in_keep_front_photo": False,
        "captured_views": [],
        "last_job_id": None,
        "updated_at": now,
    }


@app.put("/v1/me/progress")
@app.put("/api/v1/me/progress")
@app.put("/api/me/progress")
def save_my_progress(req: CustomerProgress, uid: str = Depends(current_uid)) -> dict[str, Any]:
    """Save customer progress state. Strictly enforces user ownership and never exposes measurements."""
    from apilib.stores import get_store
    store = get_store()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc = {
        "user_id": uid,
        "active_tab": req.active_tab,
        "height_cm": req.height_cm,
        "gender": req.gender,
        "age_18_confirmed": req.age_18_confirmed,
        "opt_in_keep_front_photo": req.opt_in_keep_front_photo,
        "captured_views": req.captured_views,
        "last_job_id": req.last_job_id,
        "updated_at": now,
    }
    store.set("progress", uid, doc)
    fs = get_firestore_client()
    if hasattr(fs, "set_user_progress"):
        fs.set_user_progress(uid, doc)
    return doc




def _evaluate(req: FitRequest, twin: Measurements) -> tuple[FitResult, list[str]]:
    adjustments, notes = (None, [])
    if req.review_signals:
        adjustments, notes = apply_reviews(req.review_signals, twin)
    try:
        return evaluate(twin, req.product, req.preferences, adjustments, req.pose_deltas), notes
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/v1/fit", response_model=FitResponse)
def fit(req: FitRequest, explain_text: bool = False, language: str = "en", uid: str = Depends(current_uid)) -> FitResponse:
    """Customer endpoint: full fit result for the signed-in caller's own twin.

    Set explain_text=true for a Gemini/Gemma explanation. Requires a signed-in caller
    (current_uid: Firebase ID token, or `Bearer dev:<uid>` when AUTH_MODE=dev); the twin is always
    looked up server-side for that uid, never accepted from the request body.
    """
    twin = _require_own_twin(uid)
    result, notes = _evaluate(req, twin)
    text = explain(result, language) if explain_text else None
    return FitResponse(result=result, review_notes=notes, explanation=text)


@app.post("/v1/biz/verdict", response_model=BizVerdict)
def biz_verdict(req: FitRequest, uid: str = Depends(current_uid)) -> BizVerdict:
    """Business endpoint: verdicts only. Never returns measurements, ease values or cm figures.

    Intended authentication model (documented here, not changed this task -- see HANDOFF.md's
    "Two business-verdict endpoints" section for the full writeup): this endpoint authenticates
    the caller as a *shopper* (`current_uid`, a Firebase ID token or AUTH_MODE=dev), and answers
    "what verdict would MY OWN twin get against this product" -- i.e. it is a customer-facing
    fit-check that happens to return the business-safe (no-measurements) shape, not a genuine
    business/merchant integration. `apilib/fitquery.py`'s merchant-key + consent-token flow
    (`/v1/fit-query`) is the actual intended model for a real external business caller asking
    about *a specific shopper who consented*, and is the one that should be used for any real
    merchant integration. The two are not reconciled; which one a given caller should use depends
    on whether the caller IS the shopper (`/v1/biz/verdict`) or is a merchant asking about a
    consenting shopper (`/v1/fit-query`).
    """
    twin = _require_own_twin(uid)
    result, _ = _evaluate(req, twin)
    return business_verdict(result)


@app.post("/v1/tailor-card", response_model=TailorCard)
def tailor(req: FitRequest, languages: str = "en", uid: str = Depends(current_uid)) -> TailorCard:
    """Customer endpoint: tailor card JSON for the signed-in caller's own twin. languages is comma separated, e.g. en,ta."""
    twin = _require_own_twin(uid)
    result, _ = _evaluate(req, twin)
    if result.recommended_size is None:
        raise HTTPException(status_code=409, detail="No size fits, so there is no tailor card.")
    return tailor_card(result, twin, [l.strip() for l in languages.split(",") if l.strip()])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
