"""Prompt 5, Phase 2: Body-Mesh Fit Heatmap and Pose-Deltas API.

POST /fit/mesh            -- compute a posed, zone-colored body-mesh overlay
GET  /fit/mesh/{mesh_id}/glb  -- stream the GLB bytes for the above
GET  /twin/{twin_id}/pose-deltas -- recomputed length-zone measurements in a
                                     named pose, vs standing

This is a body-mesh overlay visualization, not a simulated or draped
garment, and does not claim realistic cloth physics. All verdict/margin/
strain values come from the existing, already-tested Fit Engine
(fitcore.fit.engine.evaluate) and the existing MeshMeasurementExtractor --
nothing here invents a new measurement or strain number.

Geometry dependencies (torch/smplx) are not installed in the lightweight
API image/tests, so the heavy imports are guarded the same way
services/api/main.py already guards process_twin_job: on ImportError, every
endpoint in this router returns 503 instead of failing to import.
"""

from __future__ import annotations

import datetime
import secrets
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from apilib.auth import current_uid, err
from fitcore.fit import evaluate
from fitcore.schemas import FitPreferences, Measurements, MeasurementZone, PoseDeltas, Product, Verdict
from services.geometry.cloud_client import get_firestore_client, get_storage_client

try:
    from services.geometry.fitting import MeshMeasurementExtractor
    from services.geometry.garment_heatmap import build_zone_heatmap
    from services.geometry.exporter import export_zone_heatmap_glb
    from services.geometry.poses import generate_posed_mesh, generate_posed_output_with_joints
    _GEOMETRY_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # geometry deps (torch, smplx) are not installed in the lightweight API image/tests
    _GEOMETRY_IMPORT_ERROR = _exc


router = APIRouter()

_MESH_TTL_SECONDS = 900  # 15 minutes; these are temporary preview assets, not a persisted twin GLB
_MESH_STORAGE_PREFIX = "fit_meshes"

_LENGTH_ZONES: tuple[MeasurementZone, ...] = (
    MeasurementZone.torso_length,
    MeasurementZone.back_rise,
    MeasurementZone.inseam,
    MeasurementZone.garment_length,
)
# Circumference zones a rigid SMPL pose change (joint rotation only, no
# soft-tissue/skin simulation) cannot be expected to move -- reported as
# unmodeled rather than silently omitted or given a fabricated delta.
_UNMODELED_POSE_ZONES: tuple[MeasurementZone, ...] = (
    MeasurementZone.chest,
    MeasurementZone.waist,
    MeasurementZone.hip,
    MeasurementZone.neck,
    MeasurementZone.thigh,
)


def _require_geometry() -> None:
    if _GEOMETRY_IMPORT_ERROR is not None:
        raise err(503, "geometry_unavailable", f"Geometry pipeline unavailable: {_GEOMETRY_IMPORT_ERROR}")


def _get_owned_twin(twin_id: str, uid: str, fs: Any) -> dict[str, Any]:
    twin = fs.get_twin(twin_id)
    if not twin:
        raise err(404, "twin_not_found", f"Twin '{twin_id}' not found.")
    if twin.get("user_id") != uid:
        raise err(403, "forbidden", "This twin belongs to another user.")
    return twin


def _make_mesh_id(twin_id: str) -> tuple[str, float]:
    expires_at = time.time() + _MESH_TTL_SECONDS
    token = secrets.token_urlsafe(12)
    # "." is used as the separator rather than "_": twin_id is always
    # "twin_<hex>" (worker.py) and secrets.token_urlsafe's alphabet is
    # A-Za-z0-9-_, so a token that starts or ends with "_" (about 1 in 260
    # calls, confirmed by direct sampling) would make an "__"-based split
    # ambiguous and intermittently misparse a perfectly valid mesh id as
    # not found. "." appears in neither twin_id nor token_urlsafe's output.
    return f"{twin_id}.{token}.{int(expires_at)}", expires_at


def _parse_mesh_id(mesh_id: str) -> tuple[str, float]:
    parts = mesh_id.split(".")
    if len(parts) != 3:
        raise err(404, "mesh_not_found", "Mesh id is not valid.")
    twin_id, _token, expires_str = parts
    try:
        expires_at = float(expires_str)
    except ValueError:
        raise err(404, "mesh_not_found", "Mesh id is not valid.")
    return twin_id, expires_at


class FitMeshRequest(BaseModel):
    twin_id: str
    product: Product
    size: str
    pose: Literal["standing", "seated"] = "standing"
    pose_deltas: PoseDeltas | None = None
    preferences: FitPreferences | None = None


class ZoneHeatmapReport(BaseModel):
    status: str
    verdict: Verdict | None = None
    margin_cm: float | None = None
    strain: bool | None = Field(default=None, description="None = not computed for this pose, never fabricated")
    vertex_count: int = 0


class FitMeshResponse(BaseModel):
    twin_id: str
    size: str
    pose: str
    mesh_id: str
    glb_url: str
    expires_at: str
    zone_report: dict[MeasurementZone, ZoneHeatmapReport]


@router.post("/fit/mesh", response_model=FitMeshResponse)
def create_fit_mesh(req: FitMeshRequest, uid: str = Depends(current_uid)) -> FitMeshResponse:
    """Body-mesh fit-visualization overlay: not a simulated or draped garment."""
    _require_geometry()
    fs = get_firestore_client()
    st = get_storage_client()
    twin = _get_owned_twin(req.twin_id, uid, fs)

    betas = twin.get("betas") or []
    gender = twin.get("gender", "neutral")
    if not betas:
        raise err(422, "twin_incomplete", f"Twin '{req.twin_id}' has no fitted shape parameters yet.")

    raw_measurements = twin.get("measurements")
    if not raw_measurements:
        raise err(422, "twin_incomplete", f"Twin '{req.twin_id}' has no measurements yet.")
    twin_measurements = Measurements.model_validate(raw_measurements)

    try:
        fit_result = evaluate(twin_measurements, req.product, req.preferences, None, req.pose_deltas)
    except ValueError as exc:
        raise err(422, "evaluate_failed", str(exc))

    vertices, faces = generate_posed_mesh(betas, gender=gender, pose=req.pose)
    extractor = MeshMeasurementExtractor(gender=gender)
    try:
        heatmap = build_zone_heatmap(fit_result, req.size, req.pose, extractor, gender=gender)
    except ValueError as exc:
        raise err(404, "size_not_found", str(exc))

    glb_bytes = export_zone_heatmap_glb(vertices, faces, heatmap.vertex_colors)

    mesh_id, expires_at = _make_mesh_id(req.twin_id)
    storage_path = f"{_MESH_STORAGE_PREFIX}/{mesh_id}.glb"
    st.upload_bytes(storage_path, glb_bytes, content_type="model/gltf-binary")

    zone_report = {
        zone: ZoneHeatmapReport(
            status=status.status,
            verdict=status.verdict,
            margin_cm=status.margin_cm,
            strain=status.strain,
            vertex_count=status.vertex_count,
        )
        for zone, status in heatmap.zones.items()
    }

    return FitMeshResponse(
        twin_id=req.twin_id,
        size=req.size,
        pose=req.pose,
        mesh_id=mesh_id,
        glb_url=f"/fit/mesh/{mesh_id}/glb",
        expires_at=datetime.datetime.fromtimestamp(expires_at, tz=datetime.timezone.utc).isoformat(),
        zone_report=zone_report,
    )


@router.get("/fit/mesh/{mesh_id}/glb")
def get_fit_mesh_glb(mesh_id: str, uid: str = Depends(current_uid)) -> Response:
    fs = get_firestore_client()
    st = get_storage_client()
    twin_id, expires_at = _parse_mesh_id(mesh_id)
    storage_path = f"{_MESH_STORAGE_PREFIX}/{mesh_id}.glb"

    if time.time() > expires_at:
        st.delete_file(storage_path)
        raise err(410, "mesh_expired", "This fit-mesh preview has expired. Request a new one from POST /fit/mesh.")

    _get_owned_twin(twin_id, uid, fs)

    try:
        glb_bytes = st.download_bytes(storage_path)
    except Exception:
        raise err(404, "mesh_not_found", f"Fit mesh '{mesh_id}' not found.")

    return Response(content=glb_bytes, media_type="model/gltf-binary")


class LengthZoneDelta(BaseModel):
    standing_cm: float
    posed_cm: float
    delta_cm: float


class PoseDeltasResponse(BaseModel):
    twin_id: str
    pose: str
    measurements: dict[MeasurementZone, LengthZoneDelta]
    unmodeled_zones: list[MeasurementZone]


@router.get("/twin/{twin_id}/pose-deltas", response_model=PoseDeltasResponse)
def get_twin_pose_deltas(twin_id: str, pose: str = "seated", uid: str = Depends(current_uid)) -> PoseDeltasResponse:
    """Length-zone-only measurement deltas under a named pose.

    Circumference zones (chest/waist/hip/neck/thigh) are reported in
    unmodeled_zones, not given a computed delta: a rigid SMPL pose change
    (joint rotation only) does not simulate the soft-tissue/skin bulging a
    real body shows when seated, so this endpoint would be fabricating a
    number if it tried to report one for those zones.
    """
    _require_geometry()
    if pose not in ("standing", "seated"):
        raise err(422, "invalid_pose", f"Unknown pose '{pose}'. Expected 'standing' or 'seated'.")

    fs = get_firestore_client()
    twin = _get_owned_twin(twin_id, uid, fs)
    betas = twin.get("betas") or []
    gender = twin.get("gender", "neutral")
    if not betas:
        raise err(422, "twin_incomplete", f"Twin '{twin_id}' has no fitted shape parameters yet.")

    extractor = MeshMeasurementExtractor(gender=gender)
    v_standing, j_standing, _ = generate_posed_output_with_joints(betas, gender=gender, pose="standing")
    v_posed, j_posed, _ = generate_posed_output_with_joints(betas, gender=gender, pose=pose)

    standing_meas = extractor.extract_measurements(v_standing, j_standing)
    posed_meas = extractor.extract_measurements(v_posed, j_posed)

    measurements: dict[MeasurementZone, LengthZoneDelta] = {}
    for zone in _LENGTH_ZONES:
        standing_cm = float(standing_meas[zone].item())
        posed_cm = float(posed_meas[zone].item())
        measurements[zone] = LengthZoneDelta(
            standing_cm=round(standing_cm, 2),
            posed_cm=round(posed_cm, 2),
            delta_cm=round(posed_cm - standing_cm, 2),
        )

    return PoseDeltasResponse(
        twin_id=twin_id,
        pose=pose,
        measurements=measurements,
        unmodeled_zones=list(_UNMODELED_POSE_ZONES),
    )
