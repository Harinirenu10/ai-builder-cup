import logging
import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fitcore.schemas import MeasurementZone, Measurement, Measurements

logger = logging.getLogger("geometry.main")

app = FastAPI(
    title="Measured Fit Geometry Service",
    description="MediaPipe Pose Landmarker, SMPL 3D human body mesh fitting (CPU-only)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SMPL_DIR = Path(__file__).parent / "models" / "smpl"
TASK_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"


@app.get("/healthz")
def healthz() -> dict[str, str | bool]:
    """Health check endpoint checking service status and asset availability."""
    smpl_available = any(SMPL_DIR.glob("*.pkl"))
    task_available = TASK_PATH.exists()
    return {
        "status": "ok",
        "service": "geometry",
        "smpl_models_present": smpl_available,
        "mediapipe_task_present": task_available,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Measured Fit Geometry service online"}


@app.post("/twin/process/{job_id}")
def process_twin_job_endpoint(
    job_id: str,
    request: Request,
) -> dict[str, Any]:
    """Cloud Tasks worker endpoint for processing 3D body twin generation jobs."""
    from services.geometry.worker import JobNotFoundError, process_twin_job

    queue_name = request.headers.get("X-CloudTasks-QueueName")
    task_name = request.headers.get("X-CloudTasks-TaskName")
    logger.info("Received twin processing request for job %s (task=%s, queue=%s)", job_id, task_name, queue_name)

    try:
        result = process_twin_job(job_id=job_id)
        return result
    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Internal error processing twin job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
