"""Geometry Service: MediaPipe pose extraction, SMPL body fitting, critic loop, and mesh generation."""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fitcore.schemas import MeasurementZone, Measurement, Measurements

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
