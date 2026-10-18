"""Demo virtual try-on: a real 3D garment shell fitted to the existing body twin.

Scope (deliberately narrow, Oct 2026 demo slice): the predefined garments in
GARMENTS below -- a crew-neck tee (DEMO_TSHIRT) and fitted full-length trousers
(DEMO_TROUSERS). There is no arbitrary-URL garment ingestion and no generative
garment modelling here.

What this is
------------
A genuine 3D mesh, not an image overlay: the garment is an offset SHELL built
from the twin's own SMPL surface. The shell is a closed, two-sided piece of
geometry (outer face, inner face, stitched rim at neck / hem / both cuffs) that
sits a few millimetres off the skin, so it rotates, self-occludes and catches
light like a solid object in <model-viewer>.

Because the shell is derived from the twin's own posed surface, shoulder, chest,
sleeve and hem placement follow that twin's geometry by construction rather than
by hand-tuned alignment of a foreign mesh.

The body mesh is NEVER modified: the returned body vertices/faces are the exact
arrays the existing SMPL pipeline produces for those betas.

What this is NOT
----------------
Not cloth simulation. There is no drape, no gravity, no collision solve and no
fabric stretch: the offset is a geometric ease field, so the tee reads as a
close-fitting knit tee and cannot show a loose, hanging or bunching garment.
It is a visualization. It computes no fit verdict and no measurement -- those
stay with the Fit Engine (fitcore.fit.engine) and the body-mesh fit heatmap
(garment_heatmap.py), which are unaffected by this module.

Region selection
----------------
Which vertices a garment covers is read off the SMPL model's own linear-blend-
skinning weights (model.lbs_weights, 6890x24) rather than a hand-painted vertex
list, so it is valid for every gender and every set of betas.

Tops (TopGarmentSpec):

- body panel: vertices whose combined weight on pelvis/hips/spine1/spine2/
  spine3/collars exceeds 0.5, cut off below a hem height expressed as a
  fraction of the twin's own pelvis->neck span (so the hem scales with the
  body instead of landing at a fixed number of centimetres).
- sleeves: vertices whose weight on a shoulder joint exceeds 0.5, cut off
  beyond a fraction of that twin's own shoulder->elbow length.
- neck and head joints are excluded, so the neckline falls out of the skinning
  weights at the base of the neck instead of being carved by hand.

Bottoms (BottomGarmentSpec):

- vertices whose combined weight on pelvis/hips/spine1/knees/ankles exceeds
  0.5. Foot joints are excluded, so a full-length leg stops at the ankle rather
  than swallowing the foot.
- the waistband is cut along the twin's own pelvis->neck axis (not world Y, so
  it survives a pose change) at a rise expressed as a fraction of that span.
- each leg is cut at a distance measured ALONG that leg's own hip->knee->ankle
  chain, as a fraction of the chain's length. Every vertex is parameterised
  against whichever leg chain it is nearer to, so the cut follows the limb
  instead of slicing a horizontal plane through it, and one spec gives
  full-length trousers, shorts or anything between.

The left and right legs are separate tubes of geometry joined only at the seat,
because the garment keeps the body's own triangles: the gap between the legs is
never bridged, so a bottom is never a skirt. The result has exactly three
openings -- the waistband and one hem per leg -- and each leg hem is levelled
onto a plane perpendicular to that leg's own axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
import trimesh

from services.geometry.loader import load_smpl
from services.geometry.poses import PoseName, get_pose_tensor

# SMPL kinematic joint indices (fixed by the SMPL skeleton).
_J_PELVIS = 0
_J_NECK = 12
_J_L_SHOULDER, _J_R_SHOULDER = 16, 17
_J_L_ELBOW, _J_R_ELBOW = 18, 19
# pelvis, L/R hip, spine1, spine2, spine3, L/R collar: the tee's body panel.
_BODY_PANEL_JOINTS: tuple[int, ...] = (0, 1, 2, 3, 6, 9, 13, 14)
_SLEEVE_JOINT_PAIRS: tuple[tuple[int, int], ...] = (
    (_J_L_SHOULDER, _J_L_ELBOW),
    (_J_R_SHOULDER, _J_R_ELBOW),
)
# pelvis, L/R hip, spine1, L/R knee, L/R ankle: everything a bottom can cover.
# Foot joints (10, 11) are deliberately absent -- a trouser leg ends at the
# ankle, it does not wrap the foot.
_BOTTOM_JOINTS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8)
# Each leg as its own hip -> knee -> ankle chain, used both to decide which leg
# a vertex belongs to and to measure how far down that leg it sits.
_LEG_CHAINS: tuple[tuple[int, int, int], ...] = ((1, 4, 7), (2, 5, 8))

# A vertex counts as belonging to a region when more than half of its skinning
# weight sits on that region's joints.
_WEIGHT_THRESHOLD = 0.5


@dataclass(frozen=True)
class GarmentShellSpec:
    """Shell parameters every predefined garment shares.

    Lengths are centimetres (repo convention); fractions are dimensionless and
    are taken against the twin's own skeleton so one spec fits any body size.
    Subclasses add the cuts that make a garment a top or a bottom.
    """

    garment_id: str
    name: str
    # The garment's node name in the exported GLB scene.
    node_name: str
    # Target gap between skin and the garment's outer face. This is the "ease"
    # dial: a smaller number reads as a closer-fitting garment.
    ease_cm: float
    # Fabric thickness: the inner face sits this far inside the outer face.
    fabric_thickness_cm: float
    # Never let the outer face come closer to the skin than this, so the
    # garment cannot sink into the body in tight creases.
    min_clearance_cm: float
    # Display material (RGBA 0-1 base color, PBR roughness/metallic).
    base_color: tuple[float, float, float, float]
    roughness: float
    metallic: float
    # Concave-crease handling, see _offset_field.
    curvature_limit: float
    smoothing_iterations: int


@dataclass(frozen=True)
class TopGarmentSpec(GarmentShellSpec):
    """A garment worn on the torso: one body panel plus two sleeves."""

    # Hem height below the pelvis joint, as a fraction of the pelvis->neck span.
    hem_drop_fraction: float = 0.098
    # Sleeve end along the shoulder->elbow axis, as a fraction of its length.
    sleeve_length_fraction: float = 0.55


@dataclass(frozen=True)
class BottomGarmentSpec(GarmentShellSpec):
    """A garment worn on the lower body: a waistband and two separate legs.

    These two fractions are the whole length dial. Shorts, cropped trousers and
    full-length trousers differ only in leg_length_fraction; low-rise versus
    high-rise differs only in waist_rise_fraction. Adding jeans or shorts later
    is a new entry in GARMENTS, not new geometry code.
    """

    # Waistband height above the pelvis joint, measured along the twin's own
    # pelvis->neck axis, as a fraction of that span.
    waist_rise_fraction: float = 0.10
    # Where each leg ends, as a fraction of the distance along that leg's own
    # hip->knee->ankle chain. 1.0 reaches the ankle joint; about 0.42 is a short.
    leg_length_fraction: float = 0.95


DEMO_TSHIRT = TopGarmentSpec(
    garment_id="demo-threads-classic-crew-tee",
    name="Demo Threads Co. Classic Crew Tee",
    node_name="demo_tee",
    hem_drop_fraction=0.098,      # hem a little below the hip, a standard tee length
    sleeve_length_fraction=0.55,  # short sleeve, ending above the elbow
    ease_cm=1.3,
    fabric_thickness_cm=0.3,
    min_clearance_cm=0.5,
    base_color=(0.16, 0.26, 0.42, 1.0),  # muted navy knit
    roughness=0.85,                      # matte cotton jersey, not satin
    metallic=0.0,
    curvature_limit=3.0,
    smoothing_iterations=3,
)

DEMO_TROUSERS = BottomGarmentSpec(
    garment_id="demo-threads-fitted-trousers",
    name="Demo Threads Co. Fitted Trousers",
    node_name="demo_trousers",
    waist_rise_fraction=0.10,   # waistband sitting just above the hip bone
    leg_length_fraction=0.95,   # full length, breaking just above the ankle
    ease_cm=1.1,                # fitted: closer to the skin than the tee
    fabric_thickness_cm=0.35,   # woven twill, a little heavier than jersey
    min_clearance_cm=0.45,
    base_color=(0.20, 0.21, 0.24, 1.0),  # charcoal twill
    roughness=0.80,
    metallic=0.0,
    curvature_limit=3.0,
    smoothing_iterations=3,
)

# The predefined try-on catalog, which the API serves. Adding jeans or shorts
# later is one more entry here (a short is leg_length_fraction around 0.42).
GARMENTS: dict[str, GarmentShellSpec] = {
    DEMO_TSHIRT.garment_id: DEMO_TSHIRT,
    DEMO_TROUSERS.garment_id: DEMO_TROUSERS,
}


@dataclass
class GarmentShell:
    """A built garment shell plus the measurable facts about how it came out."""

    vertices: np.ndarray  # (N, 3) float32 metres, same world space as the body
    faces: np.ndarray     # (M, 3) int32
    spec: GarmentShellSpec
    # Diagnostics, all measured rather than asserted -- the verification test
    # and the CLI both read these.
    stats: dict[str, Any] = field(default_factory=dict)


def _region_masks(
    vertices: np.ndarray,
    joints: np.ndarray,
    lbs_weights: np.ndarray,
    spec: TopGarmentSpec,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-vertex masks for (body panel, sleeves) plus the hem height used.

    See the module docstring for how the regions are defined.
    """
    pelvis_y = float(joints[_J_PELVIS][1])
    neck_y = float(joints[_J_NECK][1])
    span = neck_y - pelvis_y
    if span <= 0:
        raise ValueError("Degenerate skeleton: the neck joint is not above the pelvis joint.")
    hem_y = pelvis_y - spec.hem_drop_fraction * span

    panel_weight = lbs_weights[:, list(_BODY_PANEL_JOINTS)].sum(axis=1)
    body_panel = (panel_weight > _WEIGHT_THRESHOLD) & (vertices[:, 1] >= hem_y)

    sleeves = np.zeros(len(vertices), dtype=bool)
    for shoulder, elbow in _SLEEVE_JOINT_PAIRS:
        a = joints[shoulder]
        b = joints[elbow]
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length <= 1e-6:
            continue
        unit = axis / length
        along = (vertices - a) @ unit
        sleeves |= (lbs_weights[:, shoulder] > _WEIGHT_THRESHOLD) & (
            along <= spec.sleeve_length_fraction * length
        )

    return body_panel, sleeves, hem_y


def _leg_parameterisation(
    vertices: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place every vertex on the nearer leg: (leg index, distance along it, that leg's length).

    Each leg is the polyline hip -> knee -> ankle. A vertex is projected onto
    every segment of both chains; the chain holding the closest projection owns
    it, and the distance along that chain is how far down the leg it sits.
    Measuring along the limb (rather than cutting a horizontal plane) is what
    makes the hem follow the leg in any pose, and it is what the leg-length
    dial is expressed in.
    """
    n_v = len(vertices)
    best_dist = np.full(n_v, np.inf)
    along = np.zeros(n_v)
    leg_of = np.zeros(n_v, dtype=np.int8)
    chain_lengths = np.zeros(len(_LEG_CHAINS))

    for leg_index, chain in enumerate(_LEG_CHAINS):
        travelled = 0.0
        for start, end in zip(chain[:-1], chain[1:]):
            a = joints[start]
            b = joints[end]
            axis = b - a
            length = float(np.linalg.norm(axis))
            if length <= 1e-6:
                continue
            unit = axis / length
            t = np.clip((vertices - a) @ unit, 0.0, length)
            distance = np.linalg.norm(vertices - (a + t[:, None] * unit), axis=1)
            closer = distance < best_dist
            best_dist[closer] = distance[closer]
            along[closer] = travelled + t[closer]
            leg_of[closer] = leg_index
            travelled += length
        chain_lengths[leg_index] = travelled

    if np.any(chain_lengths <= 0):
        raise ValueError("Degenerate skeleton: a leg chain has zero length.")

    return leg_of, along, chain_lengths[leg_of]


def _bottom_region_mask(
    vertices: np.ndarray,
    joints: np.ndarray,
    lbs_weights: np.ndarray,
    spec: BottomGarmentSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex mask for a bottom, plus (leg index, distance along leg, leg length).

    The mask is a weight test for "is this the lower body at all", a waistband
    cut along the spine axis, and a per-leg length cut along the limb. Nothing
    bridges the two legs: the mask keeps the body's own triangles, and the body
    has no triangles spanning the gap between the legs.
    """
    pelvis = joints[_J_PELVIS]
    spine = joints[_J_NECK] - pelvis
    span = float(np.linalg.norm(spine))
    if span <= 0:
        raise ValueError("Degenerate skeleton: the neck joint coincides with the pelvis joint.")
    spine_unit = spine / span

    # Height above the pelvis measured along the body's own spine axis, so a
    # pose change (which tilts the legs, not the torso) does not move the
    # waistband off the waist.
    rise = (vertices - pelvis) @ spine_unit
    waist_limit = spec.waist_rise_fraction * span

    leg_of, along, chain_length = _leg_parameterisation(vertices, joints)
    hem_cut = spec.leg_length_fraction * chain_length

    weight = lbs_weights[:, list(_BOTTOM_JOINTS)].sum(axis=1)
    covered = (weight > _WEIGHT_THRESHOLD) & (rise <= waist_limit) & (along <= hem_cut)

    return covered, leg_of, along, chain_length


def _opening_planes(
    region: trimesh.Trimesh,
    joints: np.ndarray,
    leg_of: np.ndarray,
    source_indices: np.ndarray,
) -> list[tuple[list[int], np.ndarray]]:
    """For each of a bottom's openings, (its boundary ring, its plane normal).

    Same problem _level_hem solves for a tee: a cut can only follow existing
    triangle edges, so every opening comes out as a zigzag about one edge
    length deep (~2 cm on SMPL). The difference is which plane each opening
    belongs on -- the waistband is square to the body's spine axis, and each
    leg hem is square to THAT LEG (its knee->ankle direction), not to the
    floor, so both stay right when the twin is posed.
    """
    loops = _boundary_loops(region)
    if not loops:
        return []

    vertices = np.asarray(region.vertices)
    # The waistband is the loop sitting highest along the spine; the others are
    # leg hems. Using the pelvis->neck axis keeps that true in any pose.
    spine = joints[_J_NECK] - joints[_J_PELVIS]
    spine_unit = spine / float(np.linalg.norm(spine))
    heights = [float(((vertices[loop] - joints[_J_PELVIS]) @ spine_unit).mean()) for loop in loops]
    waistband = int(np.argmax(heights))

    planes: list[tuple[list[int], np.ndarray]] = []
    for index, loop in enumerate(loops):
        if index == waistband:
            planes.append((loop, spine_unit))
            continue
        # Which leg this hem belongs to, and that leg's lower-segment direction.
        leg = int(np.round(leg_of[source_indices[loop]].mean()))
        chain = _LEG_CHAINS[min(max(leg, 0), len(_LEG_CHAINS) - 1)]
        axis = joints[chain[2]] - joints[chain[1]]
        norm = float(np.linalg.norm(axis))
        if norm > 1e-6:
            planes.append((loop, axis / norm))
    return planes


def _flatten_openings(vertices: np.ndarray, planes: list[tuple[list[int], np.ndarray]]) -> int:
    """Slide each opening's ring onto its own flat plane, in place.

    Deliberately applied to the ALREADY-OFFSET shell rather than to the body
    surface it was built from. Levelling the source surface first and then
    offsetting along the old vertex normals can push a rim vertex back through
    the skin (measured: a hem vertex ending up 0.007 cm from the body). Sliding
    the offset ring instead moves it along the limb, roughly tangentially to a
    near-cylindrical surface, so every vertex keeps the clearance it was built
    with.

    Returns the number of vertices moved.
    """
    moved = 0
    for loop, normal in planes:
        centroid = vertices[loop].mean(axis=0)
        vertices[loop] += np.outer((centroid - vertices[loop]) @ normal, normal)
        moved += len(loop)
    return moved


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """The mesh's open boundary rings, as lists of vertex indices.

    A boundary edge belongs to exactly one triangle; the rings are the
    connected components of the graph those edges form. For a tee that is the
    neckline, the hem and two cuffs; for trousers, the waistband and one hem
    per leg.
    """
    counts: dict[tuple[int, int], int] = {}
    for a, b in mesh.edges_sorted:
        key = (int(a), int(b))
        counts[key] = counts.get(key, 0) + 1

    neighbours: dict[int, list[int]] = {}
    for (a, b), count in counts.items():
        if count != 1:
            continue
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in neighbours:
        if start in seen:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(neighbours[node])
        loops.append(component)
    return loops


def _offset_field(surface: trimesh.Trimesh, spec: GarmentShellSpec) -> np.ndarray:
    """Per-vertex outward offset distance (metres) for the garment's outer face.

    A constant normal offset pinches and self-folds inside tight concave creases
    (armpit, the fold under the neck), because neighbouring normals converge.
    Two cheap, deterministic steps keep that under control:

    1. Shrink the offset where the surface is concave. Concavity is the discrete
       uniform Laplacian projected on the vertex normal, scaled by mean edge
       length so the factor is resolution-independent.
    2. Relax the offset surface with a few Taubin iterations (Taubin, not plain
       Laplacian, so the shell does not shrink into the body).

    Then re-impose spec.min_clearance_cm, since smoothing can pull a vertex back
    toward the skin.
    """
    v = np.asarray(surface.vertices, dtype=np.float64)
    normals = np.asarray(surface.vertex_normals, dtype=np.float64)
    edges = surface.edges_unique
    n_v = len(v)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_v, n_v)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    laplacian = (adjacency @ v) / degree[:, None] - v
    concavity = np.einsum("ij,ij->i", laplacian, normals)

    mean_edge = float(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1).mean())
    if mean_edge <= 0:
        raise ValueError("Degenerate garment region: mean edge length is zero.")

    ease_m = spec.ease_cm / 100.0
    limit = 1.0 / (1.0 + spec.curvature_limit * np.maximum(0.0, concavity) / mean_edge)
    offsets = ease_m * limit

    if spec.smoothing_iterations > 0:
        relaxed = trimesh.Trimesh(
            vertices=v + normals * offsets[:, None],
            faces=surface.faces,
            process=False,
        )
        trimesh.smoothing.filter_taubin(relaxed, iterations=spec.smoothing_iterations)
        offsets = np.einsum("ij,ij->i", np.asarray(relaxed.vertices) - v, normals)

    return np.maximum(offsets, spec.min_clearance_cm / 100.0)


def _level_hem(region: trimesh.Trimesh, hem_y: float) -> int:
    """Flatten the hem's sawtooth into a level edge, in place.

    The hem is cut at a height, but the cut can only follow existing triangle
    edges, so the lowest boundary ends up as a zigzag with an amplitude of about
    one edge length (~2 cm on SMPL). Snapping the hem's own boundary vertices
    onto the hem plane costs at most that much displacement and gives the tee a
    straight bottom edge. Only boundary vertices within one mean edge length of
    the hem plane move -- the neckline and the two cuffs are untouched.

    Returns the number of vertices moved.
    """
    boundary = np.unique(region.outline().referenced_vertices)
    if len(boundary) == 0:
        return 0
    edges = region.edges_unique
    mean_edge = float(np.linalg.norm(region.vertices[edges[:, 0]] - region.vertices[edges[:, 1]], axis=1).mean())

    v = np.asarray(region.vertices)
    at_hem = boundary[np.abs(v[boundary, 1] - hem_y) <= mean_edge]
    if len(at_hem) == 0:
        return 0

    moved = v.copy()
    moved[at_hem, 1] = hem_y
    region.vertices = moved
    return int(len(at_hem))


def _stitch_rim(faces: np.ndarray, outer_count: int) -> np.ndarray:
    """Triangles closing the gap between the outer and inner faces.

    Each open boundary edge of the region (the neckline, the hem and the two
    cuffs) appears in exactly one triangle, so it has no opposite-direction
    twin. Every such directed edge gets two triangles bridging to the matching
    inner-face vertices (index + outer_count), which makes the shell closed.
    """
    directed: set[tuple[int, int]] = set()
    for f in faces:
        directed.add((int(f[0]), int(f[1])))
        directed.add((int(f[1]), int(f[2])))
        directed.add((int(f[2]), int(f[0])))

    rim: list[tuple[int, int, int]] = []
    for a, b in directed:
        if (b, a) in directed:
            continue
        rim.append((a, b, b + outer_count))
        rim.append((a, b + outer_count, a + outer_count))

    if not rim:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(rim, dtype=np.int64)


def build_garment_shell(
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    joints: np.ndarray,
    lbs_weights: np.ndarray,
    spec: GarmentShellSpec = DEMO_TSHIRT,
) -> GarmentShell:
    """Build the garment shell for one twin's posed body surface.

    Args:
        body_vertices: (6890, 3) posed body vertices in metres.
        body_faces: (13776, 3) SMPL faces.
        joints: (>=24, 3) SMPL joint positions for the same pose/shape.
        lbs_weights: (6890, 24) SMPL linear-blend-skinning weights.
        spec: Which predefined garment to build.

    Returns:
        A GarmentShell whose vertices live in the SAME coordinate space as
        body_vertices (no grounding or recentring is applied here, so a caller
        can apply one transform to both meshes and keep them aligned).
    """
    v = np.asarray(body_vertices, dtype=np.float64)
    f = np.asarray(body_faces, dtype=np.int64)
    j = np.asarray(joints, dtype=np.float64)
    w = np.asarray(lbs_weights, dtype=np.float64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"body_vertices must have shape (N, 3), got {v.shape}.")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"body_faces must have shape (M, 3), got {f.shape}.")
    if w.shape[0] != v.shape[0]:
        raise ValueError(f"lbs_weights has {w.shape[0]} rows for {v.shape[0]} vertices.")
    if j.shape[0] <= _J_R_ELBOW:
        raise ValueError(
            f"joints must contain at least {_J_R_ELBOW + 1} SMPL joints, got {j.shape[0]}."
        )

    is_bottom = isinstance(spec, BottomGarmentSpec)
    region_stats: dict[str, Any] = {}
    leg_of: np.ndarray | None = None
    below_knee: np.ndarray | None = None
    hem_y = 0.0

    if is_bottom:
        covered, leg_of, along, chain_length = _bottom_region_mask(v, j, w, spec)
        # Past the knee joint, where a trouser's two legs are unambiguously
        # separate tubes (nearer the crotch the body's own surfaces meet, so a
        # gap measurement there would say nothing).
        below_knee = along > 0.5 * chain_length
        region_stats = {
            "kind": "bottom",
            "waist_rise_fraction": spec.waist_rise_fraction,
            "leg_length_fraction": spec.leg_length_fraction,
            "left_leg_vertices": int((covered & (leg_of == 0)).sum()),
            "right_leg_vertices": int((covered & (leg_of == 1)).sum()),
        }
    elif isinstance(spec, TopGarmentSpec):
        body_panel, sleeves, hem_y = _region_masks(v, j, w, spec)
        covered = body_panel | sleeves
        region_stats = {
            "kind": "top",
            "body_panel_vertices": int(body_panel.sum()),
            "sleeve_vertices": int(sleeves.sum()),
        }
    else:
        raise ValueError(
            f"Unsupported garment spec type {type(spec).__name__}: a garment must be a "
            "TopGarmentSpec or a BottomGarmentSpec."
        )

    # Keep only triangles fully inside the covered region, so the garment's
    # boundary follows real mesh edges instead of cutting through triangles.
    # This is also what keeps a bottom's two legs separate: the body has no
    # triangle spanning the gap between the legs, so the garment cannot grow
    # one either.
    keep = covered[f].all(axis=1)
    if not keep.any():
        raise ValueError("Garment region is empty for this body: no fully covered triangle.")

    # Explicit remap rather than remove_unreferenced_vertices(), so the mapping
    # back to body vertex indices stays available for the per-leg hem levelling.
    source_indices = np.unique(f[keep])
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[source_indices] = np.arange(len(source_indices))
    region = trimesh.Trimesh(vertices=v[source_indices], faces=remap[f[keep]], process=False)

    opening_planes: list[tuple[list[int], np.ndarray]] = []
    hem_vertices_levelled = 0
    if is_bottom:
        assert leg_of is not None
        # Worked out here, applied to the offset shell below -- see _flatten_openings.
        opening_planes = _opening_planes(region, j, leg_of, source_indices)
    else:
        hem_vertices_levelled = _level_hem(region, hem_y)

    normals = np.asarray(region.vertex_normals, dtype=np.float64)
    base = np.asarray(region.vertices, dtype=np.float64)
    outer_offsets = _offset_field(region, spec)
    inner_offsets = np.maximum(
        outer_offsets - spec.fabric_thickness_cm / 100.0,
        spec.min_clearance_cm / 200.0,  # the inner face may sit closer, never touching
    )

    outer = base + normals * outer_offsets[:, None]
    inner = base + normals * inner_offsets[:, None]
    if opening_planes:
        hem_vertices_levelled = _flatten_openings(outer, opening_planes)
        _flatten_openings(inner, opening_planes)
    n_outer = len(base)

    faces_outer = region.faces.astype(np.int64)
    faces_inner = region.faces.astype(np.int64)[:, ::-1] + n_outer  # flipped: looks inward
    rim = _stitch_rim(faces_outer, n_outer)

    shell = trimesh.Trimesh(
        vertices=np.vstack([outer, inner]),
        faces=np.vstack([faces_outer, faces_inner, rim]),
        process=False,
    )
    # The shell is closed, so winding can be resolved from the geometry itself.
    trimesh.repair.fix_normals(shell)

    outer_shell = trimesh.Trimesh(vertices=outer, faces=faces_outer, process=False)
    folded = int((np.einsum("ij,ij->i", outer_shell.face_normals, region.face_normals) < 0).sum())

    loops = _boundary_loops(region)

    stats = {
        "garment_id": spec.garment_id,
        "garment_name": spec.name,
        **region_stats,
        "source_vertex_count": int(n_outer),
        "shell_vertex_count": int(len(shell.vertices)),
        "shell_face_count": int(len(shell.faces)),
        "rim_face_count": int(len(rim)),
        "hem_vertices_levelled": hem_vertices_levelled,
        "boundary_loops": len(loops),
        # Every opening's ring size, largest first. A bottom reads as
        # [waistband, hem, hem]; a top as [neck/hem..., cuff, cuff].
        "opening_sizes": sorted((len(loop) for loop in loops), reverse=True),
        "is_watertight": bool(shell.is_watertight),
        "is_winding_consistent": bool(shell.is_winding_consistent),
        "outer_area_m2": float(outer_shell.area),
        "min_clearance_cm": float(outer_offsets.min() * 100.0),
        "mean_clearance_cm": float(outer_offsets.mean() * 100.0),
        "max_clearance_cm": float(outer_offsets.max() * 100.0),
        # Triangles whose normal flipped against the skin after offsetting: the
        # residual concave-crease pinch described in _offset_field. Reported,
        # not hidden.
        "folded_faces": folded,
        "folded_face_ratio": folded / max(1, len(faces_outer)),
    }

    if is_bottom and leg_of is not None and below_knee is not None:
        # How close the two leg shells come to each other past the knee. Real
        # trousers touch at the inner thigh; this says by how much the geometry
        # does, so the limitation is measured rather than assumed.
        region_leg = leg_of[source_indices]
        region_below_knee = below_knee[source_indices]
        left = outer[(region_leg == 0) & region_below_knee]
        right = outer[(region_leg == 1) & region_below_knee]
        if len(left) and len(right):
            gap_cm = float(cKDTree(right).query(left)[0].min() * 100.0)
            stats["leg_shell_min_gap_below_knee_cm"] = gap_cm
            stats["legs_are_separate_below_knee"] = gap_cm > 0.0

    return GarmentShell(
        vertices=np.asarray(shell.vertices, dtype=np.float32),
        faces=np.asarray(shell.faces, dtype=np.int32),
        spec=spec,
        stats=stats,
    )


def build_demo_tryon(
    betas: list[float],
    gender: str = "neutral",
    pose: PoseName = "standing",
    spec: GarmentShellSpec = DEMO_TSHIRT,
) -> tuple[np.ndarray, np.ndarray, GarmentShell]:
    """Posed body mesh plus one predefined garment's shell for a twin's fitted betas.

    Works for either kind of spec: pass a TopGarmentSpec (DEMO_TSHIRT) or a
    BottomGarmentSpec (DEMO_TROUSERS).

    Returns (body_vertices, body_faces, shell). The body arrays are exactly what
    the existing SMPL pipeline produces for these betas -- this adds a garment,
    it does not alter the body shape.
    """
    if not betas:
        raise ValueError("betas must not be empty.")

    model = load_smpl(gender=gender, num_betas=len(betas))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    betas_t = torch.tensor([betas], dtype=torch.float32)
    body_pose = get_pose_tensor(pose)
    with torch.no_grad():
        out = model(betas=betas_t, body_pose=body_pose)
        body_vertices = out.vertices[0].cpu().numpy().astype(np.float32)
        joints = out.joints[0].cpu().numpy().astype(np.float64)
    body_faces = model.faces.astype(np.int32)
    lbs_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)

    shell = build_garment_shell(body_vertices, body_faces, joints, lbs_weights, spec=spec)
    return body_vertices, body_faces, shell


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m services.geometry.tryon",
        description=(
            "Offline demo try-on CLI: build a predefined garment's 3D shell on a twin's fitted "
            "betas and write one GLB holding the body and the garment. No cloud, no LLM call."
        ),
    )
    parser.add_argument(
        "--garment",
        default=DEMO_TSHIRT.garment_id,
        choices=tuple(GARMENTS),
        help="Which predefined garment to build.",
    )
    parser.add_argument(
        "--betas",
        help=(
            "Comma-separated SMPL shape betas, or the path to a twin JSON file with a "
            "\"betas\" key (e.g. one written by python -m services.geometry.twin). "
            "Defaults to the mean body (all zeros)."
        ),
    )
    parser.add_argument("--gender", default="neutral", choices=("neutral", "female", "male"))
    parser.add_argument("--pose", default="standing", choices=("standing", "seated"))
    parser.add_argument("--out", default="tryon.glb", help="Destination .glb path.")
    return parser.parse_args(argv)


def _load_betas(raw: str | None) -> list[float]:
    if not raw:
        return [0.0] * 10
    path = Path(raw)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        betas = data.get("betas")
        if not betas:
            raise ValueError(f"{path} has no \"betas\" key.")
        return [float(b) for b in betas]
    return [float(part) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    # Imported here so the module stays importable (and its pure-geometry
    # functions testable) without the GLB exporter in the import path.
    from services.geometry.exporter import export_tryon_glb

    try:
        betas = _load_betas(args.betas)
        body_vertices, body_faces, shell = build_demo_tryon(
            betas, gender=args.gender, pose=args.pose, spec=GARMENTS[args.garment]
        )
        glb_bytes = export_tryon_glb(
            body_vertices,
            body_faces,
            shell.vertices,
            shell.faces,
            garment_base_color=shell.spec.base_color,
            garment_roughness=shell.spec.roughness,
            garment_metallic=shell.spec.metallic,
            garment_name=shell.spec.node_name,
            output_path=args.out,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = dict(shell.stats)
    report["glb_bytes"] = len(glb_bytes)
    report["output_path"] = str(Path(args.out).resolve())
    report["body_vertices_unchanged"] = int(len(body_vertices))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
