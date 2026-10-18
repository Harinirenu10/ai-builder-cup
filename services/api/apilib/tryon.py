"""Demo virtual try-on API: one predefined garment on the caller's own body twin.

GET /tryon/garments                          -- the predefined try-on catalog
GET /twin/{twin_id}/tryon/{garment_id}/glb   -- body + garment as one GLB scene

Scope: the predefined garments in services/geometry/tryon.py::GARMENTS -- a
crew-neck tee and fitted full-length trousers. There is no arbitrary retailer
URL ingestion and no generative garment modelling in this endpoint.

Each garment is real 3D geometry (a closed offset shell derived from that
twin's own SMPL surface), not an image composited over a render, and the body
mesh is returned unchanged alongside it. A bottom is built as a waistband plus
two separate legs with a real inseam, never as a single skirt-like tube. It is a visualization only: it produces no fit
verdict and no measurement. Fit answers stay with the Fit Engine
(fitcore.fit.engine) and the body-mesh fit heatmap (POST /fit/mesh).

Geometry dependencies (torch/smplx/trimesh) are not installed in the lightweight
API image/tests, so the heavy imports are guarded exactly as apilib/fit_mesh.py
guards them: on ImportError every endpoint here returns 503 instead of the
module failing to import.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel

from apilib.auth import current_uid, err
from services.geometry.cloud_client import get_firestore_client

try:
    from services.geometry.exporter import export_tryon_glb
    from services.geometry.tryon import GARMENTS, BottomGarmentSpec, GarmentShellSpec, build_demo_tryon
    _GEOMETRY_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # torch/smplx/trimesh absent in the lightweight API image/tests
    _GEOMETRY_IMPORT_ERROR = _exc
    GARMENTS = {}  # type: ignore[assignment]
    BottomGarmentSpec = ()  # type: ignore[misc, assignment]
    GarmentShellSpec = Any  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)

router = APIRouter()

# Shopper-facing copy per garment. The garments themselves come from the
# geometry module's GARMENTS registry, so adding jeans or shorts later is a
# spec there plus a blurb here -- not a new code path.
_GARMENT_BLURB: dict[str, str] = {
    "demo-threads-classic-crew-tee": (
        "A classic short-sleeve crew-neck tee in a matte cotton-jersey knit, "
        "shown with a close, even ease over the shoulders, chest and sleeves."
    ),
    "demo-threads-fitted-trousers": (
        "Fitted full-length trousers in a matte charcoal twill, shown with a "
        "waistband at the hip, a real inseam between two separate legs, and "
        "hems breaking just above the ankle."
    ),
}

# Language rule (CLAUDE.md): this copy describes the garment, never the body.

# Building the shell runs an SMPL forward pass plus mesh work (~2-3 s on CPU).
# The result is fully determined by (twin, garment, pose), so a small in-process
# cache keeps re-opening the viewer instant during a demo. It is per-instance and
# in-memory only: a cold Cloud Run instance simply rebuilds.
_CACHE_MAX_ENTRIES = 8
_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = threading.Lock()


def _require_geometry() -> None:
    if _GEOMETRY_IMPORT_ERROR is not None:
        raise err(503, "geometry_unavailable", f"Geometry pipeline unavailable: {_GEOMETRY_IMPORT_ERROR}")


def _spec_for(garment_id: str) -> GarmentShellSpec:
    spec = GARMENTS.get(garment_id)
    if spec is None:
        raise err(
            404,
            "garment_not_found",
            f"Unknown try-on garment '{garment_id}'. This demo endpoint serves only "
            f"{', '.join(sorted(GARMENTS))}.",
        )
    return spec


def _get_owned_twin(twin_id: str, uid: str) -> dict[str, Any]:
    fs = get_firestore_client()
    twin = fs.get_twin(twin_id)
    if not twin:
        raise err(404, "twin_not_found", f"Twin '{twin_id}' not found.")
    if twin.get("user_id") != uid:
        raise err(403, "forbidden", "This twin belongs to another user.")
    return twin


def _cache_key(twin_id: str, garment_id: str, pose: str, betas: list[float], gender: str) -> str:
    digest = hashlib.sha256(
        f"{garment_id}|{pose}|{gender}|{[round(float(b), 6) for b in betas]}".encode()
    ).hexdigest()[:16]
    return f"{twin_id}:{digest}"


class TryOnGarment(BaseModel):
    garment_id: str
    name: str
    # "top" (torso: body panel plus sleeves) or "bottom" (waistband plus two legs).
    kind: str
    description: str
    # Stated on every response so no caller can mistake this for a fit verdict.
    note: str = (
        "Visualization only: real 3D garment geometry fitted to your twin, with no cloth "
        "simulation and no fit verdict. Ask the Fit Engine for sizing answers."
    )


class TryOnCatalog(BaseModel):
    garments: list[TryOnGarment]


@router.get("/tryon/garments", response_model=TryOnCatalog)
def list_tryon_garments() -> TryOnCatalog:
    """The predefined try-on catalog. No auth needed: it is static demo content."""
    _require_geometry()
    return TryOnCatalog(
        garments=[
            TryOnGarment(
                garment_id=spec.garment_id,
                name=spec.name,
                kind="bottom" if isinstance(spec, BottomGarmentSpec) else "top",
                description=_GARMENT_BLURB.get(spec.garment_id, spec.name),
            )
            for spec in GARMENTS.values()
        ]
    )


@router.get("/twin/{twin_id}/tryon/{garment_id}/glb")
def get_tryon_glb(
    twin_id: str,
    garment_id: str,
    pose: Literal["standing", "seated"] = "standing",
    uid: str = Depends(current_uid),
) -> Response:
    """Stream one GLB holding the caller's own body twin plus the garment shell.

    Both meshes share a single grounding transform, so <model-viewer> can orbit
    the combined model with the garment staying on the body.
    """
    _require_geometry()
    spec = _spec_for(garment_id)
    twin = _get_owned_twin(twin_id, uid)

    betas = twin.get("betas") or []
    if not betas:
        raise err(422, "twin_incomplete", f"Twin '{twin_id}' has no fitted shape parameters yet.")
    gender = twin.get("gender", "neutral")

    key = _cache_key(twin_id, garment_id, pose, betas, gender)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
    if cached is not None:
        return Response(content=cached, media_type="model/gltf-binary")

    try:
        body_vertices, body_faces, shell = build_demo_tryon(
            [float(b) for b in betas], gender=gender, pose=pose, spec=spec
        )
        glb_bytes = export_tryon_glb(
            body_vertices,
            body_faces,
            shell.vertices,
            shell.faces,
            garment_base_color=spec.base_color,
            garment_roughness=spec.roughness,
            garment_metallic=spec.metallic,
            garment_name=spec.node_name,
        )
    except ValueError as exc:
        raise err(422, "tryon_failed", f"Could not build the try-on mesh for this twin: {exc}")

    logger.info(
        "try-on built twin=%s garment=%s pose=%s faces=%s clearance=%.2f-%.2fcm folded=%.2f%%",
        twin_id,
        garment_id,
        pose,
        shell.stats.get("shell_face_count"),
        shell.stats.get("min_clearance_cm", 0.0),
        shell.stats.get("max_clearance_cm", 0.0),
        100.0 * shell.stats.get("folded_face_ratio", 0.0),
    )

    with _cache_lock:
        _cache[key] = glb_bytes
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)

    return Response(content=glb_bytes, media_type="model/gltf-binary")
