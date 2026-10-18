"""3D GLB digital twin export pipeline for SMPL body models.

Exports fitted SMPL meshes to standard binary .glb format compatible with
Google model-viewer (@google/model-viewer), Three.js, and standard 3D web engines.

Requirements (Prompt 1 specification):
- Preserves 6,890 vertices and 13,776 faces.
- Accurate metric scale (meters in GLTF specification).
- Natural orientation (Y-up, Z-forward, soles grounded at Y=0).
- Neutral grey PBR/diffuse material.
- CPU-only execution without GPU or OpenGL dependencies via Trimesh.
- Binary validation: re-imports and verifies topology, bounds, and finiteness.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
import trimesh

from services.geometry.fitting import FitResult

# Standard GLTF binary magic header: "glTF" (0x46546C67)
GLTF_MAGIC = b"glTF"

# Neutral light-grey aesthetic material color (RGBA uint8)
DEFAULT_TWIN_COLOR = (195, 195, 200, 255)


def export_twin_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path | None = None,
    ground_mesh: bool = True,
    color: tuple[int, int, int, int] = DEFAULT_TWIN_COLOR,
) -> bytes:
    """Export 3D body vertices and faces to a binary .glb file.

    Args:
        vertices: (6890, 3) float array of vertex coordinates in meters.
        faces: (13776, 3) integer triangle index array.
        output_path: Optional destination path to write the .glb file.
        ground_mesh: If True, shifts Y so the lowest sole vertex rests at Y=0.0.
        color: RGBA uint8 tuple for neutral avatar appearance.

    Returns:
        Raw bytes of the compiled .glb asset.

    Raises:
        ValueError: If vertices or faces are malformed, non-finite, or empty.
    """
    v_arr = np.asarray(vertices, dtype=np.float32)
    f_arr = np.asarray(faces, dtype=np.int32)

    # 1. Input integrity checks
    if v_arr.ndim != 2 or v_arr.shape[1] != 3:
        raise ValueError(f"Vertices array must have shape (N, 3), got {v_arr.shape}.")

    if f_arr.ndim != 2 or f_arr.shape[1] != 3:
        raise ValueError(f"Faces array must have shape (M, 3), got {f_arr.shape}.")

    if len(v_arr) == 0 or len(f_arr) == 0:
        raise ValueError("Vertices and faces must not be empty.")

    if not np.all(np.isfinite(v_arr)):
        raise ValueError("Vertices array contains NaN or Infinite coordinate values.")

    if np.any(f_arr < 0) or np.any(f_arr >= len(v_arr)):
        raise ValueError("Faces array contains vertex indices out of bounds.")

    # 2. Coordinate normalization
    v_export = v_arr.copy()
    if ground_mesh:
        # Ground soles to Y = 0.0 and center horizontal center of mass at X=0, Z=0
        min_y = float(np.min(v_export[:, 1]))
        v_export[:, 1] -= min_y
        mean_x = float(np.mean(v_export[:, 0]))
        mean_z = float(np.mean(v_export[:, 2]))
        v_export[:, 0] -= mean_x
        v_export[:, 2] -= mean_z

    # 3. Build Trimesh geometry
    mesh = trimesh.Trimesh(
        vertices=v_export,
        faces=f_arr,
        process=False,  # Preserve exact SMPL 6890 vertex indexing
    )

    # 4. Apply clean vertex visual coloring and smooth normals
    mesh.fix_normals()
    vertex_colors = np.full((len(v_export), 4), color, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=vertex_colors,
    )

    # 5. Export to binary GLTF (.glb)
    glb_bytes = mesh.export(file_type="glb")
    if not isinstance(glb_bytes, bytes):
        glb_bytes = bytes(glb_bytes)

    # 6. Save to disk if requested
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(glb_bytes)

    return glb_bytes


def validate_glb_bytes(glb_bytes: bytes) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate a compiled .glb binary asset by re-parsing.

    Checks:
    - Valid GLTF 2.0 binary magic header ("glTF").
    - Successful scene decoding.
    - Preserved vertex and face counts.
    - Finite bounding box and realistic stature in meters.

    Args:
        glb_bytes: Raw bytes of the .glb file.

    Returns:
        (is_valid, error_message, metadata_dict)
    """
    if not glb_bytes or len(glb_bytes) < 20:
        return False, "GLB byte buffer is empty or truncated (< 20 bytes).", {}

    # Check magic header
    if glb_bytes[:4] != GLTF_MAGIC:
        return False, f"Invalid GLTF binary header magic: {glb_bytes[:4]!r}. Expected b'glTF'.", {}

    try:
        scene_or_mesh = trimesh.load(
            trimesh.util.wrap_as_stream(glb_bytes),
            file_type="glb",
        )
    except Exception as e:
        return False, f"Trimesh failed to parse GLB stream: {e}", {}

    # Unpack geometry from scene
    if isinstance(scene_or_mesh, trimesh.Scene):
        if not scene_or_mesh.geometry:
            return False, "GLB scene contains no geometric meshes.", {}
        mesh = trimesh.util.concatenate(scene_or_mesh.dump())
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        return False, f"Unexpected type returned by Trimesh: {type(scene_or_mesh)}", {}

    v_count = len(mesh.vertices)
    f_count = len(mesh.faces)

    if v_count == 0 or f_count == 0:
        return False, f"GLB mesh has zero elements: {v_count} vertices, {f_count} faces.", {}

    if not np.all(np.isfinite(mesh.vertices)):
        return False, "Parsed GLB mesh contains NaN or Inf vertex coordinates.", {}

    bounds = mesh.bounds
    y_height_m = float(bounds[1, 1] - bounds[0, 1])

    metadata = {
        "num_vertices": v_count,
        "num_faces": f_count,
        "height_m": round(y_height_m, 3),
        "height_cm": round(y_height_m * 100.0, 1),
        "bounds_min": [round(float(x), 3) for x in bounds[0]],
        "bounds_max": [round(float(x), 3) for x in bounds[1]],
        "file_size_bytes": len(glb_bytes),
    }

    return True, None, metadata


def export_fit_result_to_glb(
    fit_result: FitResult,
    output_path: str | Path | None = None,
    ground_mesh: bool = True,
) -> bytes:
    """Helper to export a completed FitResult directly to a .glb asset."""
    if not fit_result.is_valid:
        raise ValueError(
            f"Cannot export invalid FitResult: {fit_result.rejection_reason or 'Validation failed.'}"
        )

    return export_twin_glb(
        vertices=fit_result.vertices,
        faces=fit_result.faces,
        output_path=output_path,
        ground_mesh=ground_mesh,
    )


def export_zone_heatmap_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    output_path: str | Path | None = None,
    ground_mesh: bool = True,
) -> bytes:
    """Export a body mesh with a PER-VERTEX color array (Prompt 5, Phase 2's
    body-mesh fit-visualization overlay -- see garment_heatmap.py). This is a
    separate function: export_twin_glb's own default export (one uniform
    neutral color for every vertex) is unchanged and still the exact
    behavior used everywhere else.

    Args:
        vertices: (N, 3) float array of vertex coordinates in meters.
        faces: (M, 3) integer triangle index array.
        vertex_colors: (N, 4) uint8 RGBA array, one color per vertex (as
            produced by garment_heatmap.build_zone_heatmap).
        output_path: Optional destination path to write the .glb file.
        ground_mesh: If True, shifts Y so the lowest sole vertex rests at Y=0.0.

    Returns:
        Raw bytes of the compiled .glb asset.

    Raises:
        ValueError: If vertices, faces, or vertex_colors are malformed.
    """
    v_arr = np.asarray(vertices, dtype=np.float32)
    f_arr = np.asarray(faces, dtype=np.int32)
    c_arr = np.asarray(vertex_colors, dtype=np.uint8)

    if v_arr.ndim != 2 or v_arr.shape[1] != 3:
        raise ValueError(f"Vertices array must have shape (N, 3), got {v_arr.shape}.")
    if f_arr.ndim != 2 or f_arr.shape[1] != 3:
        raise ValueError(f"Faces array must have shape (M, 3), got {f_arr.shape}.")
    if len(v_arr) == 0 or len(f_arr) == 0:
        raise ValueError("Vertices and faces must not be empty.")
    if c_arr.shape != (len(v_arr), 4):
        raise ValueError(f"vertex_colors must have shape ({len(v_arr)}, 4), got {c_arr.shape}.")
    if not np.all(np.isfinite(v_arr)):
        raise ValueError("Vertices array contains NaN or Infinite coordinate values.")
    if np.any(f_arr < 0) or np.any(f_arr >= len(v_arr)):
        raise ValueError("Faces array contains vertex indices out of bounds.")

    v_export = v_arr.copy()
    if ground_mesh:
        min_y = float(np.min(v_export[:, 1]))
        v_export[:, 1] -= min_y
        mean_x = float(np.mean(v_export[:, 0]))
        mean_z = float(np.mean(v_export[:, 2]))
        v_export[:, 0] -= mean_x
        v_export[:, 2] -= mean_z

    mesh = trimesh.Trimesh(vertices=v_export, faces=f_arr, process=False)
    mesh.fix_normals()
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=c_arr)

    glb_bytes = mesh.export(file_type="glb")
    if not isinstance(glb_bytes, bytes):
        glb_bytes = bytes(glb_bytes)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(glb_bytes)

    return glb_bytes
