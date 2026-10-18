"""Body-mesh fit-visualization overlay (Prompt 5, Phase 2).

This is explicitly NOT a simulated or draped garment, and makes no claim of
realistic cloth physics or fabric deformation. It colors the EXISTING SMPL
body mesh's own zone regions (already computed by fitting.py's
MeshMeasurementExtractor) using the EXISTING, already-tested Fit Engine
verdicts (fitcore.fit.engine.evaluate -> ZoneFit.verdict / margin_cm /
poses[pose].strain). No new measurement or strain value is computed here --
this module only maps already-computed per-zone results onto mesh vertices
and picks a display color.

Supported zones (a defensible, geometry-grounded vertex mapping exists):
- 5 circumference zones (chest, waist, hip, neck, thigh): every mesh vertex
  that is an endpoint of one of MeshMeasurementExtractor's own loop-crossing
  edges at that zone's fixed height -- the same vertices the differentiable
  circumference calculation already uses, read from the extractor's public
  loop attributes with zero changes to fitting.py.
- 2 paired-landmark zones (shoulder_width, upper_back): both already have
  TWO real mesh vertices (left/right acromion, left/right armscye). The
  region is every template-mesh vertex within a small distance of the
  straight segment between them -- a real multi-vertex band, not a single
  point standing in for a whole zone.

Unsupported zones (reported explicitly, never silently dropped or faked):
sleeve_length, inseam, torso_length, back_rise, garment_length. Each of
these is anchored by at least one SMPL JOINT or a loop's mean-plane height
rather than a second mesh vertex, so there is no defensible vertex-to-vertex
region to color without inventing a boundary the existing topology analysis
doesn't actually define.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from fitcore.schemas import FitResult, MeasurementZone, Verdict, ZoneFit
from services.geometry.fitting import MeshMeasurementExtractor
from services.geometry.loader import load_smpl
from services.geometry.poses import get_pose_tensor

# Display-only color buckets (RGBA uint8). A direct, documented lookup from
# the Fit Engine's own Verdict enum -- not a new strain computation.
COLOR_TIGHT = (214, 48, 49, 255)      # red
COLOR_LOOSE_OR_LENGTH = (235, 155, 40, 255)  # orange (loose, short, long)
COLOR_IDEAL = (64, 176, 98, 255)      # green
COLOR_UNSUPPORTED = (195, 195, 200, 255)  # same as exporter.DEFAULT_TWIN_COLOR

_VERDICT_COLOR = {
    Verdict.tight: COLOR_TIGHT,
    Verdict.loose: COLOR_LOOSE_OR_LENGTH,
    Verdict.short: COLOR_LOOSE_OR_LENGTH,
    Verdict.long: COLOR_LOOSE_OR_LENGTH,
    Verdict.ideal: COLOR_IDEAL,
}

_LOOP_ZONES: dict[MeasurementZone, str] = {
    MeasurementZone.chest: "chest_loop",
    MeasurementZone.waist: "waist_loop",
    MeasurementZone.hip: "hip_loop",
    MeasurementZone.neck: "neck_loop",
    MeasurementZone.thigh: "thigh_loop",
}

_PAIRED_LANDMARK_ZONES: dict[MeasurementZone, tuple[str, str]] = {
    MeasurementZone.shoulder_width: ("idx_l_acromion", "idx_r_acromion"),
    MeasurementZone.upper_back: ("idx_l_armscye", "idx_r_armscye"),
}

# Visualization choice, not a biomechanical claim: how wide a band around a
# paired-landmark segment counts as "that zone" for display purposes.
_PAIRED_LANDMARK_BAND_CM = 3.0

UNSUPPORTED_ZONES: tuple[MeasurementZone, ...] = (
    MeasurementZone.sleeve_length,
    MeasurementZone.inseam,
    MeasurementZone.torso_length,
    MeasurementZone.back_rise,
    MeasurementZone.garment_length,
)


def _template_vertices(gender: str, pose: str = "standing") -> np.ndarray:
    """Reference-pose (zero betas) SMPL vertices, independent of any fitted twin --
    used only to measure which vertices lie near a paired-landmark segment.
    Uses get_pose_tensor(pose) (canonical standing A-pose by default) rather than
    unposed T-pose, ensuring landmark segment vertices align with the actual
    rendered avatar geometry.
    """
    model = load_smpl(gender=gender, num_betas=10)
    body_pose = get_pose_tensor(pose)
    with torch.no_grad():
        out = model(body_pose=body_pose)
        return out.vertices[0].cpu().numpy()


def _vertices_near_segment(vertices: np.ndarray, idx_a: int, idx_b: int, max_dist_m: float) -> list[int]:
    a, b = vertices[idx_a], vertices[idx_b]
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return [idx_a, idx_b]
    t = np.clip((vertices - a) @ ab / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dist = np.linalg.norm(vertices - proj, axis=1)
    return np.where(dist < max_dist_m)[0].tolist()


def zone_vertex_map(
    extractor: MeshMeasurementExtractor,
    gender: str = "neutral",
    pose: str = "standing",
) -> dict[MeasurementZone, list[int]]:
    """Map each SUPPORTED zone to the mesh vertex indices that represent it.

    Only called once per gender/topology/pose (SMPL topology is fixed across
    shape/betas, so these indices are valid for any twin of that gender in that pose).
    """
    mapping: dict[MeasurementZone, list[int]] = {}

    for zone, loop_attr in _LOOP_ZONES.items():
        loop = getattr(extractor, loop_attr)
        idx = set()
        for t in (loop.idx_a1, loop.idx_b1, loop.idx_a2, loop.idx_b2):
            idx.update(int(i) for i in t.tolist())
        mapping[zone] = sorted(idx)

    template_v = _template_vertices(gender, pose=pose)
    max_dist_m = _PAIRED_LANDMARK_BAND_CM / 100.0
    for zone, (attr_a, attr_b) in _PAIRED_LANDMARK_ZONES.items():
        idx_a = getattr(extractor, attr_a)
        idx_b = getattr(extractor, attr_b)
        mapping[zone] = _vertices_near_segment(template_v, idx_a, idx_b, max_dist_m)

    return mapping


@dataclass
class ZoneHeatmapStatus:
    zone: MeasurementZone
    status: Literal["colored", "unsupported_no_vertex_mapping", "not_in_fit_result", "pose_data_unavailable"]
    verdict: Verdict | None = None
    margin_cm: float | None = None
    strain: bool | None = None  # None = not computed for the requested pose (never fabricated)
    vertex_count: int = 0


@dataclass
class ZoneHeatmapResult:
    vertex_colors: np.ndarray  # (6890, 4) uint8
    zones: dict[MeasurementZone, ZoneHeatmapStatus] = field(default_factory=dict)


def build_zone_heatmap(
    fit_result: FitResult,
    size: str,
    pose: str,
    extractor: MeshMeasurementExtractor,
    gender: str = "neutral",
    num_vertices: int = 6890,
    default_color: tuple[int, int, int, int] = COLOR_UNSUPPORTED,
) -> ZoneHeatmapResult:
    """Build the (6890,4) per-vertex color array plus a per-zone status report.

    Reuses fit_result.size_fit(size).zones[...] (the existing, already-tested
    Fit Engine output) as the only source of verdict/margin/strain. This
    function adds no new strain computation -- only a spatial mapping and a
    color lookup.
    """
    size_fit = fit_result.size_fit(size)
    if size_fit is None:
        raise ValueError(f"Size '{size}' not found in this FitResult.")

    vertex_map = zone_vertex_map(extractor, gender=gender, pose=pose)

    colors = np.full((num_vertices, 4), default_color, dtype=np.uint8)
    zone_reports: dict[MeasurementZone, ZoneHeatmapStatus] = {}

    for zone in MeasurementZone:
        zone_fit: ZoneFit | None = size_fit.zones.get(zone)
        if zone_fit is None:
            zone_reports[zone] = ZoneHeatmapStatus(zone=zone, status="not_in_fit_result")
            continue

        color = _VERDICT_COLOR.get(zone_fit.verdict, default_color)
        strain: bool | None = None
        pose_status: Literal["colored", "pose_data_unavailable"] = "colored"
        if pose != "standing":
            pose_fit = zone_fit.poses.get(pose)
            if pose_fit is None:
                pose_status = "pose_data_unavailable"
                strain = None
            else:
                strain = pose_fit.strain
                if strain:
                    color = COLOR_TIGHT

        indices = vertex_map.get(zone)
        if zone in UNSUPPORTED_ZONES or not indices:
            zone_reports[zone] = ZoneHeatmapStatus(
                zone=zone,
                status="unsupported_no_vertex_mapping",
                verdict=zone_fit.verdict,
                margin_cm=zone_fit.margin_cm,
                strain=strain,
                vertex_count=0,
            )
            continue

        colors[indices] = color
        zone_reports[zone] = ZoneHeatmapStatus(
            zone=zone,
            status=pose_status,
            verdict=zone_fit.verdict,
            margin_cm=zone_fit.margin_cm,
            strain=strain,
            vertex_count=len(indices),
        )

    return ZoneHeatmapResult(vertex_colors=colors, zones=zone_reports)
