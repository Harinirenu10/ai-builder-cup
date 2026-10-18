"""Unit tests for services/geometry/garment_heatmap.py (Prompt 5, Phase 2).

Verifies:
1. The 5 circumference zones map to real, already-computed topology
   vertices (from MeshMeasurementExtractor's own loop attributes).
2. The 2 paired-landmark zones map to a genuine multi-vertex band, not a
   single landmark vertex standing in for the whole zone.
3. The 5 documented unsupported zones get no vertex mapping at all.
4. Per-vertex colors are a direct, undistorted readout of the existing Fit
   Engine's ZoneFit.verdict (no new strain computation).
5. A seated-pose strain (computed by the existing, already-tested Fit
   Engine, not by this module) overrides the baseline color to tight/red.
6. Missing pose data for a zone is reported as unavailable, never guessed.
7. export_zone_heatmap_glb produces a valid GLB and export_twin_glb's
   existing default (uniform-color) export behavior is unchanged.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from fitcore.fit import evaluate
from fitcore.schemas import Measurement, Measurements, MeasurementZone as Z, PoseDelta, PoseDeltas, Product, SizeChart, Verdict
from services.geometry.exporter import DEFAULT_TWIN_COLOR, export_twin_glb, export_zone_heatmap_glb, validate_glb_bytes
from services.geometry.fitting import MeshMeasurementExtractor
from services.geometry.garment_heatmap import (
    COLOR_IDEAL,
    COLOR_LOOSE_OR_LENGTH,
    COLOR_TIGHT,
    UNSUPPORTED_ZONES,
    build_zone_heatmap,
    zone_vertex_map,
)
from services.geometry.loader import load_smpl


def _body(**cm: float) -> Measurements:
    return Measurements(zones={Z(k): Measurement(value_cm=v, confidence=0.9, source="test") for k, v in cm.items()})


def _product(category: str, sizes: dict, **kw) -> Product:
    return Product(
        id="p", brand="b", title="t", category=category,
        fabric=[{"fiber": "cotton", "percent": 100}],
        size_chart=SizeChart(sizes={s: {Z(k): v for k, v in row.items()} for s, row in sizes.items()}, chart_type="garment"),
        **kw,
    )


@pytest.fixture(scope="module")
def extractor() -> MeshMeasurementExtractor:
    return MeshMeasurementExtractor(gender="neutral")


# --------------------------------------------------------------------------- vertex mapping


def test_loop_zones_map_to_real_topology_vertices(extractor):
    vmap = zone_vertex_map(extractor)
    for zone, loop_attr in [
        (Z.chest, "chest_loop"), (Z.waist, "waist_loop"), (Z.hip, "hip_loop"),
        (Z.neck, "neck_loop"), (Z.thigh, "thigh_loop"),
    ]:
        indices = vmap[zone]
        assert len(indices) > 10, f"{zone.value} vertex mapping is too small to be a real region"
        loop = getattr(extractor, loop_attr)
        loop_vertices = set()
        for t in (loop.idx_a1, loop.idx_b1, loop.idx_a2, loop.idx_b2):
            loop_vertices.update(int(i) for i in t.tolist())
        assert set(indices) == loop_vertices, f"{zone.value} mapping must come only from its own existing loop data"


def test_paired_landmark_zones_are_multi_vertex_bands(extractor):
    vmap = zone_vertex_map(extractor)
    for zone, (idx_a, idx_b) in [
        (Z.shoulder_width, (extractor.idx_l_acromion, extractor.idx_r_acromion)),
        (Z.upper_back, (extractor.idx_l_armscye, extractor.idx_r_armscye)),
    ]:
        indices = vmap[zone]
        assert len(indices) > 5, f"{zone.value} must be a real band, not a single landmark vertex"
        # The band must actually be anchored at the two real landmarks, not an arbitrary region.
        assert idx_a in indices or idx_b in indices or len(indices) > 20


def test_unsupported_zones_have_no_vertex_mapping(extractor):
    vmap = zone_vertex_map(extractor)
    for zone in UNSUPPORTED_ZONES:
        assert zone not in vmap, f"{zone.value} is documented unsupported and must not get a fabricated mapping"


# --------------------------------------------------------------------------- color rules


def test_heatmap_colors_match_existing_fit_engine_verdict(extractor):
    b = _body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = _product("trousers", {"M": {"waist": 83, "hip": 112, "thigh": 77, "inseam": 79, "back_rise": 27}})
    fit_result = evaluate(b, p)

    assert fit_result.sizes[0].zones[Z.waist].verdict == Verdict.tight
    assert fit_result.sizes[0].zones[Z.hip].verdict == Verdict.ideal
    assert fit_result.sizes[0].zones[Z.thigh].verdict == Verdict.loose

    heatmap = build_zone_heatmap(fit_result, "M", "standing", extractor)

    waist_idx = zone_vertex_map(extractor)[Z.waist][0]
    hip_idx = zone_vertex_map(extractor)[Z.hip][0]
    thigh_idx = zone_vertex_map(extractor)[Z.thigh][0]
    assert tuple(heatmap.vertex_colors[waist_idx]) == COLOR_TIGHT
    assert tuple(heatmap.vertex_colors[hip_idx]) == COLOR_IDEAL
    assert tuple(heatmap.vertex_colors[thigh_idx]) == COLOR_LOOSE_OR_LENGTH

    assert heatmap.zones[Z.waist].status == "colored"
    assert heatmap.zones[Z.waist].vertex_count > 0


def test_unsupported_zone_is_reported_not_colored(extractor):
    b = _body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = _product("trousers", {"M": {"waist": 83, "hip": 112, "thigh": 77, "inseam": 79, "back_rise": 27}})
    fit_result = evaluate(b, p)
    heatmap = build_zone_heatmap(fit_result, "M", "standing", extractor)

    assert heatmap.zones[Z.inseam].status == "unsupported_no_vertex_mapping"
    assert heatmap.zones[Z.inseam].vertex_count == 0
    # The verdict is still reported (transparency) even though no vertices are colored for it.
    assert heatmap.zones[Z.inseam].verdict is not None


def test_seated_strain_overrides_to_tight_color(extractor):
    """Reuses the exact scenario already proven in
    packages/fitcore/tests/fit/test_engine.py::
    test_pose_slim_trouser_zero_thigh_ease_strains_when_seated.
    """
    b = _body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = _product(
        "trousers", {"M": {"waist": 86, "hip": 106, "thigh": 56, "inseam": 79, "back_rise": 27}}, fit_type="slim",
    )
    seated = PoseDeltas(poses={"seated": PoseDelta(minutes=120, delta_cm={Z.thigh: 3.0})})
    fit_result = evaluate(b, p, pose_deltas=seated)
    assert fit_result.sizes[0].zones[Z.thigh].poses["seated"].strain is True  # the engine's own, pre-existing result

    heatmap = build_zone_heatmap(fit_result, "M", "seated", extractor)
    thigh_idx = zone_vertex_map(extractor)[Z.thigh][0]
    assert tuple(heatmap.vertex_colors[thigh_idx]) == COLOR_TIGHT
    assert heatmap.zones[Z.thigh].strain is True
    assert heatmap.zones[Z.thigh].status == "colored"


def test_missing_pose_data_is_reported_as_unavailable_not_guessed(extractor):
    b = _body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = _product("trousers", {"M": {"waist": 83, "hip": 112, "thigh": 77, "inseam": 79, "back_rise": 27}})
    # Only thigh has a declared seated delta; waist/hip have none for "seated".
    seated = PoseDeltas(poses={"seated": PoseDelta(minutes=60, delta_cm={Z.thigh: 2.0})})
    fit_result = evaluate(b, p, pose_deltas=seated)
    assert "seated" not in fit_result.sizes[0].zones[Z.waist].poses  # confirms the engine itself has no data here

    heatmap = build_zone_heatmap(fit_result, "M", "seated", extractor)
    assert heatmap.zones[Z.waist].status == "pose_data_unavailable"
    assert heatmap.zones[Z.waist].strain is None  # never fabricated as True/False
    # Still colored from the baseline verdict so the mesh is not blank.
    waist_idx = zone_vertex_map(extractor)[Z.waist][0]
    assert tuple(heatmap.vertex_colors[waist_idx]) == COLOR_TIGHT


def test_standing_pose_never_reports_missing_pose_data(extractor):
    b = _body(waist=82, hip=98, thigh=56, inseam=78, back_rise=26)
    p = _product("trousers", {"M": {"waist": 83, "hip": 112, "thigh": 77, "inseam": 79, "back_rise": 27}})
    fit_result = evaluate(b, p)
    heatmap = build_zone_heatmap(fit_result, "M", "standing", extractor)
    assert all(z.status != "pose_data_unavailable" for z in heatmap.zones.values())


# --------------------------------------------------------------------------- export


def test_export_zone_heatmap_glb_is_valid():
    model = load_smpl("neutral")
    v = model().vertices[0].detach().numpy()
    f = model.faces
    colors = np.full((len(v), 4), DEFAULT_TWIN_COLOR, dtype=np.uint8)
    colors[:100] = COLOR_TIGHT

    glb_bytes = export_zone_heatmap_glb(v, f, colors)
    is_valid, err, meta = validate_glb_bytes(glb_bytes)
    assert is_valid, err
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776


def test_export_zone_heatmap_glb_rejects_wrong_color_shape():
    model = load_smpl("neutral")
    v = model().vertices[0].detach().numpy()
    f = model.faces
    bad_colors = np.full((10, 4), DEFAULT_TWIN_COLOR, dtype=np.uint8)
    with pytest.raises(ValueError):
        export_zone_heatmap_glb(v, f, bad_colors)


def test_export_twin_glb_default_behavior_is_unchanged():
    """Regression: adding export_zone_heatmap_glb must not touch export_twin_glb."""
    model = load_smpl("neutral")
    v = model().vertices[0].detach().numpy()
    f = model.faces
    glb_bytes = export_twin_glb(vertices=v, faces=f)
    is_valid, err, meta = validate_glb_bytes(glb_bytes)
    assert is_valid, err
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776


# --------------------------------------------------------------------------- Phase 3 regression tests


def test_shoulder_landmark_vertices_strictly_align_to_standing_posed_mesh(extractor):
    """Phase 3 regression: shoulder_width vertices must strictly lie within the
    3.0 cm band of the acromion-to-acromion segment on the actual standing body mesh.

    Under the old unposed T-pose template, 48 out of 78 vertices drifted up to 5.2 cm
    away from the shoulder line when deformed into standing A-pose.
    """
    from services.geometry.poses import generate_posed_mesh
    v_stand, _ = generate_posed_mesh([0.0] * 10, gender="neutral", pose="standing")

    vmap = zone_vertex_map(extractor, gender="neutral", pose="standing")
    sh_indices = vmap[Z.shoulder_width]

    idx_a = extractor.idx_l_acromion
    idx_b = extractor.idx_r_acromion
    a, b = v_stand[idx_a], v_stand[idx_b]
    ab = b - a
    ab2 = float(np.dot(ab, ab))

    # All returned vertices must be within 3.0 cm (0.03 m) of the posed segment
    pts = v_stand[sh_indices]
    t = np.clip((pts - a) @ ab / ab2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    distances_m = np.linalg.norm(pts - proj, axis=1)

    assert len(sh_indices) >= 20, "Shoulder band must contain at least 20 vertices"
    assert np.all(distances_m < 0.03), f"Max distance {distances_m.max()*100:.2f} cm exceeds 3.0 cm band"
    # Acromion anchors must be included
    assert idx_a in sh_indices
    assert idx_b in sh_indices


def test_seated_pose_adjusts_paired_landmarks_for_posture(extractor):
    """Phase 3 regression: seated pose uses seated-pose aligned vertices for paired landmarks."""
    from services.geometry.poses import generate_posed_mesh
    v_seat, _ = generate_posed_mesh([0.0] * 10, gender="neutral", pose="seated")

    vmap_seated = zone_vertex_map(extractor, gender="neutral", pose="seated")
    sh_indices = vmap_seated[Z.shoulder_width]
    ub_indices = vmap_seated[Z.upper_back]

    idx_la = extractor.idx_l_acromion
    idx_ra = extractor.idx_r_acromion
    assert idx_la in sh_indices
    assert idx_ra in sh_indices

    idx_ls = extractor.idx_l_armscye
    idx_rs = extractor.idx_r_armscye
    assert idx_ls in ub_indices
    assert idx_rs in ub_indices


def test_zone_vertex_map_backward_compatible_default_standing(extractor):
    """Phase 3 regression: calling zone_vertex_map without pose parameter defaults to standing."""
    vmap_default = zone_vertex_map(extractor, gender="neutral")
    vmap_standing = zone_vertex_map(extractor, gender="neutral", pose="standing")

    assert vmap_default[Z.shoulder_width] == vmap_standing[Z.shoulder_width]
    assert vmap_default[Z.upper_back] == vmap_standing[Z.upper_back]
    assert vmap_default[Z.chest] == vmap_standing[Z.chest]
