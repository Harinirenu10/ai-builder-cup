"""Unit tests for services/geometry/tryon.py (demo virtual try-on garment shell).

Verifies:
1. The covered region is a real multi-thousand-triangle patch selected from the
   SMPL skinning weights, with exactly the four openings a tee has (neckline,
   hem, two cuffs) and no stray disconnected pieces.
2. The garment covers the shoulders, chest and upper arms and stops where a tee
   stops: above the knees, below the hips, short of the elbows and below the
   chin.
3. The garment is a CLOSED 3D shell with real fabric thickness, not a one-sided
   surface and not a flat overlay.
4. Every garment vertex sits strictly outside the skin, so the tee never sinks
   into the body it is worn on.
5. The body mesh is returned bit-for-bit unchanged (same 6890 vertices as the
   existing SMPL pipeline produces for the same betas).
6. It builds for every gender and for extreme betas, not only the mean body.
7. export_tryon_glb writes a valid GLB with BOTH meshes in one scene, under one
   shared grounding transform, and the existing single-mesh exports are
   unaffected.
8. The offline CLI runs without cloud or LLM access and reports its diagnostics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import trimesh

from services.geometry.exporter import (
    DEFAULT_TWIN_COLOR,
    export_tryon_glb,
    export_twin_glb,
    validate_glb_bytes,
)
from services.geometry.poses import generate_posed_mesh
from services.geometry.tryon import (
    DEMO_TROUSERS,
    DEMO_TSHIRT,
    GARMENTS,
    BottomGarmentSpec,
    TopGarmentSpec,
    build_demo_tryon,
    build_garment_shell,
)


MEAN_BETAS = [0.0] * 10


@pytest.fixture(scope="module")
def demo_tryon():
    """Body mesh + demo tee shell for the mean body. Built once: SMPL is slow."""
    return build_demo_tryon(MEAN_BETAS, gender="neutral", pose="standing")


@pytest.fixture(scope="module")
def demo_trousers():
    """Body mesh + demo trouser shell for the mean body. Built once: SMPL is slow."""
    return build_demo_tryon(MEAN_BETAS, gender="neutral", pose="standing", spec=DEMO_TROUSERS)


def test_region_is_a_real_patch_with_four_tee_openings(demo_tryon):
    _, _, shell = demo_tryon
    stats = shell.stats
    # A tee-sized patch of the 13,776-triangle body, not a token sliver.
    assert stats["source_vertex_count"] > 1000
    assert stats["shell_face_count"] > 5000
    # Neckline, hem, left cuff, right cuff.
    assert stats["boundary_loops"] == 4
    # Both halves are present: the body panel is the larger of the two.
    assert stats["sleeve_vertices"] > 100
    assert stats["body_panel_vertices"] > stats["sleeve_vertices"]


def test_garment_covers_a_tees_extent_and_nothing_more(demo_tryon):
    body_vertices, _, shell = demo_tryon
    body_y = body_vertices[:, 1]
    sole_y, crown_y = float(body_y.min()), float(body_y.max())
    stature = crown_y - sole_y

    g_y = shell.vertices[:, 1]
    hem = (float(g_y.min()) - sole_y) / stature
    top = (float(g_y.max()) - sole_y) / stature

    # Hem sits around hip height: below the waist, far above the knee.
    assert 0.50 < hem < 0.62, f"hem at {hem:.2f} of stature"
    # The top of the tee reaches the shoulder/neck base, not over the chin.
    assert 0.78 < top < 0.88, f"shoulder line at {top:.2f} of stature"

    # Sleeves: the garment is wider than the torso (it reaches onto the arms)
    # but stops well short of the hands.
    g_half_width = float(np.abs(shell.vertices[:, 0]).max())
    body_half_width = float(np.abs(body_vertices[:, 0]).max())
    assert g_half_width > 0.2 * body_half_width
    assert g_half_width < 0.6 * body_half_width, "sleeves run too far down the arm"


def test_shell_is_closed_geometry_with_fabric_thickness(demo_tryon):
    _, _, shell = demo_tryon
    mesh = trimesh.Trimesh(vertices=shell.vertices, faces=shell.faces, process=False)

    # A closed solid, not a single-sided sheet and not an image plane.
    assert shell.stats["is_watertight"] is True
    assert shell.stats["is_winding_consistent"] is True
    assert mesh.volume > 0
    # Outer face + inner face + a stitched rim at every opening.
    assert shell.stats["rim_face_count"] > 0
    assert shell.stats["shell_vertex_count"] == 2 * shell.stats["source_vertex_count"]

    # Mean thickness implied by volume/area is close to the spec'd fabric
    # thickness (the two faces are a real distance apart).
    implied_cm = 100.0 * mesh.volume / shell.stats["outer_area_m2"]
    assert 0.5 * DEMO_TSHIRT.fabric_thickness_cm < implied_cm < 2.0 * DEMO_TSHIRT.fabric_thickness_cm


def test_garment_never_sinks_into_the_body(demo_tryon):
    body_vertices, body_faces, shell = demo_tryon
    body = trimesh.Trimesh(vertices=body_vertices, faces=body_faces, process=False)

    # Every garment vertex is outside the skin: using the body's own vertex
    # neighbourhood, the nearest skin vertex is at least a safe fraction of the
    # configured minimum clearance away.
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(np.asarray(body.vertices)).query(shell.vertices)
    assert float(distances.min()) * 100.0 > 0.5 * DEMO_TSHIRT.min_clearance_cm

    # And the reported clearance band matches the spec: never below the floor,
    # and close to the requested ease (the Taubin relaxation in _offset_field can
    # push a convex vertex slightly past the target, never far past it).
    assert shell.stats["min_clearance_cm"] >= DEMO_TSHIRT.min_clearance_cm - 1e-6
    assert shell.stats["max_clearance_cm"] <= 1.25 * DEMO_TSHIRT.ease_cm
    assert shell.stats["mean_clearance_cm"] == pytest.approx(DEMO_TSHIRT.ease_cm, rel=0.25)

    # The concave-crease pinch described in tryon._offset_field stays marginal.
    assert shell.stats["folded_face_ratio"] < 0.03


def test_body_mesh_is_returned_unchanged(demo_tryon):
    body_vertices, body_faces, _ = demo_tryon
    expected_v, expected_f = generate_posed_mesh(MEAN_BETAS, gender="neutral", pose="standing")

    assert body_vertices.shape == (6890, 3)
    assert body_faces.shape == (13776, 3)
    np.testing.assert_array_equal(body_vertices, expected_v)
    np.testing.assert_array_equal(body_faces, expected_f)


@pytest.mark.parametrize(
    "gender,betas",
    [
        ("neutral", [-2.5, 2.0, 0, 0, 0, 0, 0, 0, 0, 0]),  # very slight build
        ("neutral", [3.0, -2.5, 0, 0, 0, 0, 0, 0, 0, 0]),  # much larger build
        ("female", [0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("male", [-1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_builds_for_every_gender_and_extreme_shapes(gender, betas):
    _, _, shell = build_demo_tryon(betas, gender=gender, pose="standing")
    assert shell.stats["is_watertight"] is True
    assert shell.stats["boundary_loops"] == 4
    assert shell.stats["min_clearance_cm"] >= DEMO_TSHIRT.min_clearance_cm - 1e-6
    assert shell.stats["folded_face_ratio"] < 0.03


def test_build_garment_shell_rejects_malformed_input(demo_tryon):
    body_vertices, body_faces, _ = demo_tryon
    joints = np.zeros((24, 3))
    joints[12, 1] = 0.5  # neck above pelvis
    weights = np.zeros((len(body_vertices), 24))

    with pytest.raises(ValueError, match="shape"):
        build_garment_shell(body_vertices[:, :2], body_faces, joints, weights)
    with pytest.raises(ValueError, match="rows"):
        build_garment_shell(body_vertices, body_faces, joints, weights[:10])
    with pytest.raises(ValueError, match="SMPL joints"):
        build_garment_shell(body_vertices, body_faces, joints[:5], weights)
    # All-zero skinning weights cover nothing: reported, not silently empty.
    with pytest.raises(ValueError, match="region is empty"):
        build_garment_shell(body_vertices, body_faces, joints, weights)


def test_export_tryon_glb_holds_both_meshes_in_one_grounded_scene(demo_tryon):
    body_vertices, body_faces, shell = demo_tryon
    glb = export_tryon_glb(
        body_vertices,
        body_faces,
        shell.vertices,
        shell.faces,
        garment_base_color=DEMO_TSHIRT.base_color,
        garment_roughness=DEMO_TSHIRT.roughness,
        garment_metallic=DEMO_TSHIRT.metallic,
        garment_name="demo_tee",
    )
    assert glb[:4] == b"glTF"

    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    assert set(scene.geometry.keys()) == {"body", "demo_tee"}

    body = scene.geometry["body"]
    garment = scene.geometry["demo_tee"]
    assert body.vertices.shape == (6890, 3)
    assert body.faces.shape == (13776, 3)
    assert len(garment.vertices) == shell.stats["shell_vertex_count"]

    # One shared grounding transform: the body is grounded at Y=0, and the
    # garment moved with it instead of being grounded separately.
    assert float(body.vertices[:, 1].min()) == pytest.approx(0.0, abs=1e-5)
    assert float(garment.vertices[:, 1].min()) > 0.4 * float(body.vertices[:, 1].max())
    # The garment stays inside the body's horizontal footprint (it is worn, not adrift).
    assert float(np.abs(garment.vertices[:, 0]).max()) < float(np.abs(body.vertices[:, 0]).max())

    # The garment carries a fabric material rather than the body's vertex colors.
    assert garment.visual.material is not None
    assert garment.visual.material.main_color[:3].tolist() != list(DEFAULT_TWIN_COLOR[:3])


def test_export_tryon_glb_rejects_malformed_meshes(demo_tryon):
    body_vertices, body_faces, shell = demo_tryon
    with pytest.raises(ValueError, match="garment vertices must have shape"):
        export_tryon_glb(body_vertices, body_faces, shell.vertices[:, :2], shell.faces)
    with pytest.raises(ValueError, match="garment faces contain vertex indices out of bounds"):
        export_tryon_glb(body_vertices, body_faces, shell.vertices[:10], shell.faces)
    with pytest.raises(ValueError, match="body vertices contain NaN"):
        bad = body_vertices.copy()
        bad[0, 0] = np.nan
        export_tryon_glb(bad, body_faces, shell.vertices, shell.faces)


def test_existing_single_mesh_twin_export_is_unchanged(demo_tryon):
    """The try-on export is additive: export_twin_glb still produces the same
    one-mesh, 6890-vertex, neutral-grey twin it always did."""
    body_vertices, body_faces, _ = demo_tryon
    glb = export_twin_glb(body_vertices, body_faces)
    is_valid, error, meta = validate_glb_bytes(glb)
    assert is_valid, error
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776


def test_offline_cli_builds_a_glb_and_reports_diagnostics(tmp_path):
    out = tmp_path / "tryon.glb"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.geometry.tryon",
            "--betas",
            "0.4,-0.3,0.1,0,0.2,0,0,0,0,0",
            "--gender",
            "neutral",
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["garment_id"] == DEMO_TSHIRT.garment_id
    assert report["is_watertight"] is True
    assert report["body_vertices_unchanged"] == 6890
    assert out.exists()
    assert out.read_bytes()[:4] == b"glTF"


def test_offline_cli_reports_bad_betas_instead_of_crashing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "services.geometry.tryon", "--betas", "not-a-number"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr


# --------------------------------------------------------------------------- bottoms


def test_trousers_have_a_waistband_and_two_separate_leg_openings(demo_trousers):
    _, _, shell = demo_trousers
    stats = shell.stats
    assert stats["kind"] == "bottom"

    # Exactly three openings: the waistband and one hem per leg. A skirt-like
    # tube would have two; an unclosed leg or a stray patch would give more.
    assert stats["boundary_loops"] == 3
    waistband, hem_a, hem_b = stats["opening_sizes"]
    # The waistband is the largest opening, and the two leg hems are
    # near-identical rings -- which is what "two legs" looks like.
    assert waistband > hem_a
    assert abs(hem_a - hem_b) <= max(2, 0.25 * hem_a)

    # Both legs are really populated, in similar amounts.
    left, right = stats["left_leg_vertices"], stats["right_leg_vertices"]
    assert left > 300 and right > 300
    assert abs(left - right) < 0.25 * (left + right)


def test_trousers_are_two_legs_not_a_skirt(demo_trousers):
    """Nothing bridges the gap between the legs: below the seat no triangle has
    vertices on both sides of the midline, and the two leg shells stay a real
    distance apart."""
    _, _, shell = demo_trousers
    vertices = shell.vertices
    tri = vertices[shell.faces]

    crotch_y = float(np.percentile(vertices[:, 1], 55))  # safely below the seat
    below = (tri[:, :, 1] < crotch_y).all(axis=1)
    spans_midline = (tri[:, :, 0].max(axis=1) > 0.02) & (tri[:, :, 0].min(axis=1) < -0.02)
    assert not bool((below & spans_midline).any()), "a triangle bridges the two legs"

    # And the gap is measured, not assumed.
    assert shell.stats["legs_are_separate_below_knee"] is True
    assert shell.stats["leg_shell_min_gap_below_knee_cm"] > 1.0


def test_trousers_cover_hip_to_ankle_and_stop_there(demo_trousers):
    body_vertices, _, shell = demo_trousers
    body_y = body_vertices[:, 1]
    sole_y, crown_y = float(body_y.min()), float(body_y.max())
    stature = crown_y - sole_y

    g_y = shell.vertices[:, 1]
    hem = (float(g_y.min()) - sole_y) / stature
    waist = (float(g_y.max()) - sole_y) / stature

    # Waistband around the hip/low waist, well below the chest.
    assert 0.55 < waist < 0.68, f"waistband at {waist:.2f} of stature"
    # Full length: the hem reaches the ankle without swallowing the foot.
    assert 0.03 < hem < 0.12, f"hem at {hem:.2f} of stature"


def test_trouser_shell_is_closed_geometry_with_fabric_thickness(demo_trousers):
    _, _, shell = demo_trousers
    mesh = trimesh.Trimesh(vertices=shell.vertices, faces=shell.faces, process=False)

    assert shell.stats["is_watertight"] is True
    assert shell.stats["is_winding_consistent"] is True
    assert mesh.volume > 0
    assert shell.stats["rim_face_count"] > 0
    assert shell.stats["shell_vertex_count"] == 2 * shell.stats["source_vertex_count"]

    implied_cm = 100.0 * mesh.volume / shell.stats["outer_area_m2"]
    assert 0.5 * DEMO_TROUSERS.fabric_thickness_cm < implied_cm
    assert implied_cm < 2.0 * DEMO_TROUSERS.fabric_thickness_cm


def test_trousers_never_sink_into_the_body(demo_trousers):
    body_vertices, _, shell = demo_trousers
    from scipy.spatial import cKDTree

    # The shell's closest point to the skin is its INNER face, whose own floor is
    # min_clearance_cm / 2 (see build_garment_shell). Nothing may come nearer than
    # half of that -- i.e. nothing is inside the body.
    inner_floor_cm = DEMO_TROUSERS.min_clearance_cm / 2.0
    distances, _ = cKDTree(body_vertices).query(shell.vertices)
    assert float(distances.min()) * 100.0 > 0.5 * inner_floor_cm

    assert shell.stats["min_clearance_cm"] >= DEMO_TROUSERS.min_clearance_cm - 1e-6
    assert shell.stats["max_clearance_cm"] <= 1.25 * DEMO_TROUSERS.ease_cm
    assert shell.stats["folded_face_ratio"] < 0.03


def test_trousers_leave_the_body_mesh_unchanged(demo_trousers):
    body_vertices, body_faces, _ = demo_trousers
    expected_v, expected_f = generate_posed_mesh(MEAN_BETAS, gender="neutral", pose="standing")

    assert body_vertices.shape == (6890, 3)
    assert body_faces.shape == (13776, 3)
    np.testing.assert_array_equal(body_vertices, expected_v)
    np.testing.assert_array_equal(body_faces, expected_f)


@pytest.mark.parametrize(
    "gender,betas,pose",
    [
        ("neutral", [-2.5, 2.0, 0, 0, 0, 0, 0, 0, 0, 0], "standing"),
        ("neutral", [3.0, -2.5, 0, 0, 0, 0, 0, 0, 0, 0], "standing"),
        ("female", [0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0], "standing"),
        ("male", [-1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0], "standing"),
        # A pose change tilts the legs; the waistband is cut along the spine and
        # each hem along its own leg, so both have to survive it.
        ("neutral", [0.0] * 10, "seated"),
    ],
)
def test_trousers_build_for_every_gender_shape_and_pose(gender, betas, pose):
    _, _, shell = build_demo_tryon(betas, gender=gender, pose=pose, spec=DEMO_TROUSERS)
    assert shell.stats["is_watertight"] is True
    assert shell.stats["boundary_loops"] == 3
    assert shell.stats["legs_are_separate_below_knee"] is True
    assert shell.stats["min_clearance_cm"] >= DEMO_TROUSERS.min_clearance_cm - 1e-6
    assert shell.stats["folded_face_ratio"] < 0.03


def test_leg_length_is_an_adjustable_dial():
    """Shorts and cropped trousers are the same geometry code with one number
    changed -- which is what makes adding jeans or shorts later a spec entry,
    not new geometry. Every length still closes properly."""
    hems = []
    for fraction in (0.42, 0.70, 0.95):
        spec = replace(
            DEMO_TROUSERS,
            leg_length_fraction=fraction,
            garment_id=f"test-bottom-{fraction}",
        )
        _, _, shell = build_demo_tryon(MEAN_BETAS, gender="neutral", spec=spec)
        assert shell.stats["is_watertight"] is True
        assert shell.stats["boundary_loops"] == 3
        assert shell.stats["leg_length_fraction"] == fraction
        hems.append(float(shell.vertices[:, 1].min()))

    # A longer leg reaches lower, every time.
    assert hems == sorted(hems, reverse=True), f"hem heights not monotonic: {hems}"


def test_ease_is_an_adjustable_dial():
    """A looser spec really does sit further off the skin."""
    loose = replace(DEMO_TROUSERS, ease_cm=2.2, garment_id="test-bottom-loose")
    _, _, fitted_shell = build_demo_tryon(MEAN_BETAS, gender="neutral", spec=DEMO_TROUSERS)
    _, _, loose_shell = build_demo_tryon(MEAN_BETAS, gender="neutral", spec=loose)
    assert loose_shell.stats["mean_clearance_cm"] > fitted_shell.stats["mean_clearance_cm"] + 0.3
    assert loose_shell.stats["is_watertight"] is True


def test_catalog_holds_both_demo_garments_and_their_kinds():
    assert set(GARMENTS) == {DEMO_TSHIRT.garment_id, DEMO_TROUSERS.garment_id}
    assert isinstance(DEMO_TSHIRT, TopGarmentSpec)
    assert isinstance(DEMO_TROUSERS, BottomGarmentSpec)
    # Distinct GLB node names, so one scene could hold both without a clash.
    assert DEMO_TSHIRT.node_name != DEMO_TROUSERS.node_name


def test_export_tryon_glb_holds_body_and_trousers_in_one_grounded_scene(demo_trousers):
    body_vertices, body_faces, shell = demo_trousers
    glb = export_tryon_glb(
        body_vertices,
        body_faces,
        shell.vertices,
        shell.faces,
        garment_base_color=DEMO_TROUSERS.base_color,
        garment_roughness=DEMO_TROUSERS.roughness,
        garment_metallic=DEMO_TROUSERS.metallic,
        garment_name=DEMO_TROUSERS.node_name,
    )
    assert glb[:4] == b"glTF"

    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    assert set(scene.geometry.keys()) == {"body", DEMO_TROUSERS.node_name}

    body = scene.geometry["body"]
    garment = scene.geometry[DEMO_TROUSERS.node_name]
    assert body.vertices.shape == (6890, 3)
    assert len(garment.vertices) == shell.stats["shell_vertex_count"]

    # One shared grounding transform: the body is grounded at Y=0 and the
    # trousers moved with it, landing on the body's lower half.
    stature = float(body.vertices[:, 1].max())
    assert float(body.vertices[:, 1].min()) == pytest.approx(0.0, abs=1e-5)
    assert float(garment.vertices[:, 1].max()) < 0.75 * stature
    assert float(garment.vertices[:, 1].min()) < 0.2 * stature
    assert float(np.abs(garment.vertices[:, 0]).max()) < float(np.abs(body.vertices[:, 0]).max())


def test_offline_cli_builds_the_trousers_too(tmp_path):
    out = tmp_path / "trousers.glb"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.geometry.tryon",
            "--garment",
            DEMO_TROUSERS.garment_id,
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["garment_id"] == DEMO_TROUSERS.garment_id
    assert report["kind"] == "bottom"
    assert report["boundary_loops"] == 3
    assert report["is_watertight"] is True
    assert report["body_vertices_unchanged"] == 6890
    assert out.read_bytes()[:4] == b"glTF"


def test_tee_behaviour_is_unchanged_by_the_bottoms_work(demo_tryon):
    """The tee still reports what it did before bottoms existed: a top, four
    openings (neckline, hem, two cuffs), watertight, same clearance floor."""
    _, _, shell = demo_tryon
    stats = shell.stats
    assert stats["kind"] == "top"
    assert stats["boundary_loops"] == 4
    assert stats["is_watertight"] is True
    assert stats["body_panel_vertices"] > stats["sleeve_vertices"] > 100
    assert stats["min_clearance_cm"] >= DEMO_TSHIRT.min_clearance_cm - 1e-6
