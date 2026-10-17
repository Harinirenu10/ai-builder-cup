"""Procedural stand-in body mesh and fit heatmaps, written as binary glTF (.glb) with numpy only.

NOT SMPL. The SMPL model files are license-gated and the geometry service is not built, so the demo persona's
"twin" is a lofted mannequin whose girths follow the persona's (simulated) measurements. Every demo screen that
shows it says so. The files are static assets served by Firebase Hosting; nothing waits on the geometry service.

    python seed/glb.py samples/twin.json out_dir        # writes twin.glb
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

SEGMENTS = 40            # points around each ring
NEUTRAL = (0.78, 0.79, 0.82)
VERDICT_RGB = {          # fit verdicts, as colours a colour-blind viewer can still tell apart by lightness too
    "ideal": (0.20, 0.62, 0.42),
    "tight": (0.86, 0.28, 0.20),
    "short": (0.90, 0.58, 0.15),
    "loose": (0.25, 0.47, 0.85),
    "long": (0.45, 0.35, 0.80),
}

# fraction of standing height at which each zone sits (stylised, not anatomical)
ZONE_HEIGHT = {"shoulder_width": 0.815, "chest": 0.72, "waist": 0.625, "hip": 0.53, "thigh": 0.44, "garment_length": 0.60,
               "back_rise": 0.50, "inseam": 0.25, "torso_length": 0.66, "sleeve_length": 0.70, "upper_back": 0.74, "neck": 0.87}


def _ellipse_radii(circumference_cm: float, depth_ratio: float) -> tuple[float, float]:
    """Half-width and half-depth (metres) of an ellipse with the given circumference."""
    # Ramanujan-free: for a = width radius, b = depth radius, C ~ 2*pi*sqrt((a^2+b^2)/2)
    mean = circumference_cm / 100.0 / (2 * np.pi)
    a = mean * np.sqrt(2 / (1 + depth_ratio**2))
    return float(a), float(a * depth_ratio)


def _profile(z: dict[str, float], height_cm: float) -> list[tuple[float, float, float]]:
    """(y fraction of height, circumference cm, depth ratio) from feet to crown."""
    chest, waist, hip = z["chest"], z["waist"], z["hip"]
    thigh = z.get("thigh", hip * 0.57)
    shoulders = max(chest * 1.02, z.get("shoulder_width", 42.0) * 2.35)
    knee = thigh * 0.66
    calf = knee * 0.9
    ankle = knee * 0.6
    neck = z.get("neck", 36.0)
    # legs are lofted as one tube, so the lower half is stylised: two legs side by side read as one wider column
    return [
        (0.00, ankle * 1.9, 0.55), (0.04, ankle * 1.9, 0.50), (0.12, calf * 1.9, 0.55), (0.26, knee * 1.9, 0.55),
        (0.40, thigh * 1.85, 0.62), (0.48, hip * 0.97, 0.70), (0.53, hip, 0.72), (0.585, (hip + waist) / 2, 0.70),
        (0.625, waist, 0.70), (0.67, (waist + chest) / 2, 0.72), (0.72, chest, 0.74), (0.775, (chest + shoulders) / 2, 0.66),
        (0.815, shoulders, 0.52), (0.845, neck * 1.9, 0.78), (0.865, neck, 0.95), (0.89, neck * 0.98, 0.95),
        # head
        (0.905, 54.0, 0.92), (0.935, 57.0, 0.95), (0.965, 54.0, 0.92), (0.985, 40.0, 0.90), (1.00, 6.0, 0.90),
    ]


def _loft(rings: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """rings: (y, centre_x, half_width, half_depth) -> vertices, triangle indices (closed at both ends)."""
    th = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    verts = []
    for y, cx, a, b in rings:
        verts.append(np.stack([cx + a * np.cos(th), np.full_like(th, y), b * np.sin(th)], axis=1))
    n = len(rings)
    faces = []
    for i in range(n - 1):
        for j in range(SEGMENTS):
            j2 = (j + 1) % SEGMENTS
            p, q, r, s = i * SEGMENTS + j, i * SEGMENTS + j2, (i + 1) * SEGMENTS + j, (i + 1) * SEGMENTS + j2
            faces += [(p, r, q), (q, r, s)]
    v = np.concatenate(verts)
    # cap both ends with a centre vertex
    for ring_i, flip in ((0, True), (n - 1, False)):
        y, cx, _, _ = rings[ring_i]
        c = len(v)
        v = np.vstack([v, [cx, y, 0.0]])
        for j in range(SEGMENTS):
            j2 = (j + 1) % SEGMENTS
            a_, b_ = ring_i * SEGMENTS + j, ring_i * SEGMENTS + j2
            faces.append((c, b_, a_) if flip else (c, a_, b_))
    return v.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def body_mesh(zones_cm: dict[str, float], height_cm: float = 172.0) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (metres, Y up, feet at 0) and faces of the stand-in body: torso/legs/head plus two arms."""
    h = height_cm / 100.0
    rings = []
    for yf, circ, depth in _profile(zones_cm, height_cm):
        a, b = _ellipse_radii(circ, depth)
        rings.append((yf * h, 0.0, a, b))
    vb, fb = _loft(rings)

    shoulder_half = max(zones_cm.get("shoulder_width", 42.0), 36.0) / 200.0
    sleeve = zones_cm.get("sleeve_length", 58.0) / 100.0
    arm_top = 0.80 * h
    parts_v, parts_f, offset = [vb], [fb], len(vb)
    for side in (-1.0, 1.0):
        x0 = side * (shoulder_half + 0.035)
        arm = []
        for t in np.linspace(0.0, 1.0, 7):
            y = arm_top - t * sleeve * 1.02
            x = x0 + side * 0.05 * np.sin(t * np.pi / 2)
            r = 0.047 - 0.018 * t
            arm.append((y, x, r, r * 0.95))
        va, fa = _loft(arm)
        parts_v.append(va)
        parts_f.append(fa + offset)
        offset += len(va)
    return np.concatenate(parts_v), np.concatenate(parts_f)


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.where(length == 0, 1, length)).astype(np.float32)


def heat_colors(v: np.ndarray, height_cm: float, verdicts: dict[str, str] | None) -> np.ndarray:
    """Per-vertex RGBA. Neutral grey when verdicts is None; otherwise each zone tints a soft band around its height."""
    base = np.tile(np.array(NEUTRAL, dtype=np.float32), (len(v), 1))
    if not verdicts:
        return np.concatenate([base, np.ones((len(v), 1), np.float32)], axis=1)
    h = height_cm / 100.0
    weight_sum = np.zeros(len(v), dtype=np.float32)
    tint = np.zeros_like(base)
    for zone, verdict in verdicts.items():
        if zone not in ZONE_HEIGHT or verdict not in VERDICT_RGB:
            continue
        centre = ZONE_HEIGHT[zone] * h
        # sleeves tint the arms only; every other zone tints the torso and legs
        is_arm = np.abs(v[:, 0]) > 0.17
        if zone == "sleeve_length":
            w = np.where(is_arm, 1.0, 0.0)
        elif zone == "shoulder_width":
            w = np.exp(-(((v[:, 1] - centre) / 0.045) ** 2)) * (1 + 0.5 * is_arm)
        else:
            w = np.exp(-(((v[:, 1] - centre) / 0.06) ** 2)) * np.where(is_arm, 0.0, 1.0)
        tint += w[:, None].astype(np.float32) * np.array(VERDICT_RGB[verdict], dtype=np.float32)
        weight_sum += w.astype(np.float32)
    mix = np.clip(weight_sum, 0.0, 1.0)[:, None]
    avg = np.where(weight_sum[:, None] > 0, tint / np.maximum(weight_sum[:, None], 1e-6), base)
    rgb = base * (1 - mix) + avg * mix
    return np.concatenate([rgb, np.ones((len(v), 1), np.float32)], axis=1)


def write_glb(v: np.ndarray, f: np.ndarray, colors: np.ndarray, path: Path | None = None) -> bytes:
    """Binary glTF 2.0 with positions, normals, vertex colours and indices; a white PBR material."""
    v = v.astype(np.float32)
    n = _vertex_normals(v, f)
    c = colors.astype(np.float32)
    idx = f.astype(np.uint32).reshape(-1)
    blobs = [v.tobytes(), n.tobytes(), c.tobytes(), idx.tobytes()]
    offsets, pos = [], 0
    for b in blobs:
        offsets.append(pos)
        pos += len(b)  # every blob is a multiple of 4 bytes already
    binary = b"".join(blobs)
    mn, mx = v.min(axis=0).tolist(), v.max(axis=0).tolist()
    doc = {
        "asset": {"version": "2.0", "generator": "measured-fit seed/glb.py (stand-in body, not SMPL)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "stand_in_body"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.85}, "doubleSided": True}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(blobs[3]), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": len(v), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(v), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    out = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out)
    return out


def read_glb_header(data: bytes) -> dict:
    """Parse a .glb we wrote (used by tests): returns the JSON chunk."""
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("not a valid glb")
    jlen, jtype = struct.unpack("<II", data[12:20])
    if jtype != 0x4E4F534A:
        raise ValueError("first chunk is not JSON")
    return json.loads(data[20:20 + jlen])


def make_twin_glb(zones_cm: dict[str, float], height_cm: float, verdicts: dict[str, str] | None = None, path: Path | None = None) -> bytes:
    v, f = body_mesh(zones_cm, height_cm)
    return write_glb(v, f, heat_colors(v, height_cm, verdicts), path)


if __name__ == "__main__":
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    zones = {k: val["value_cm"] for k, val in json.loads(src.read_text(encoding="utf-8"))["zones"].items()}
    make_twin_glb(zones, 172.0, None, out_dir / "twin.glb")
    print(f"wrote {out_dir / 'twin.glb'}")
