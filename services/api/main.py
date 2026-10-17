"""API Service: Customer and Business endpoints, ADK planner orchestration."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fitcore.schemas import BodyTwinJobRequest, FitResult, JobStatus, MeasurementZone, Product
from services.geometry.cloud_client import get_firestore_client, get_storage_client
from services.geometry.worker import JobNotFoundError, process_twin_job

logger = logging.getLogger("api.main")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
