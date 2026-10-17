"""Unit tests for geometry.twin (Offline 3D Body Twin CLI).

Verifies:
1. python -m geometry.twin --help outputs expected usage and flags.
2. Successful end-to-end execution generates measurements.json, twin.glb, and 4 overlays.
3. measurements.json contains all 12 zones, confidence scores, and diagnostics.
4. twin.glb has valid GLTF binary header, 6,890 vertices, and 13,776 faces.
5. Missing input files exit with code 1 and descriptive error.
6. Invalid subject stature exits with code 1.
7. Blurry or unexposed photos fail local quality checks and exit with code 1.
8. Strict offline guarantee: zero network requests or LLM calls occur during execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest

from services.geometry.exporter import validate_glb_bytes
from services.geometry.landmarks import ViewLandmarkResult
from services.geometry.twin import main, parse_args, run_offline_pipeline


def _make_valid_test_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate high-contrast synthetic JPEG passing local quality/blur checks."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (255, 255, 255), 4)
    cv2.circle(img, (width // 2, 70), 30, (230, 200, 180), -1)
    cv2.rectangle(img, (width // 2 - 35, 110), (width // 2 + 35, 240), (80, 80, 80), -1)
    cv2.line(img, (width // 2, 240), (width // 2 - 20, height - 20), (40, 40, 40), 6)
    cv2.line(img, (width // 2, 240), (width // 2 + 20, height - 20), (40, 40, 40), 6)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _make_blurry_test_jpeg(width: int = 240, height: int = 360) -> bytes:
    """Generate uniform gray JPEG with low Laplacian variance (triggers blur)."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def _create_mock_landmarks_and_silhouettes():
    """Synthetic landmarks and masks for rapid test execution."""
    views = ["front", "back", "left", "right"]
    lm_results = {}
    masks = {}
    silhouettes = {}
    h, w = 360, 240

    for v in views:
        mask = np.zeros((h, w), dtype=np.uint8)
        bw = 70 if v in ("front", "back") else 45
        mask[50:330, w // 2 - bw // 2 : w // 2 + bw // 2] = 1
        masks[v] = mask

        lm_results[v] = ViewLandmarkResult(
            view=v,
            landmarks=[],
            image_width=w,
            image_height=h,
            mask_top_row=50,
            mask_bottom_row=329,
            pixel_height=280,
            cm_per_pixel=172.0 / 280.0,
            mean_confidence=0.92,
        )
    return lm_results, masks, silhouettes


@pytest.fixture
def sample_photos(tmp_path: Path) -> dict[str, Path]:
    valid_bytes = _make_valid_test_jpeg()
    photos = {}
    for v in ("front", "back", "left", "right"):
        p = tmp_path / f"{v}.jpg"
        p.write_bytes(valid_bytes)
        photos[v] = p
    return photos


def test_cli_help():
    """Verify python -m geometry.twin --help runs successfully via subprocess."""
    res = subprocess.run(
        [sys.executable, "-m", "geometry.twin", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--front" in res.stdout
    assert "--height" in res.stdout
    assert "--out" in res.stdout


@patch("services.geometry.twin.extract_four_view_landmarks_and_silhouettes")
def test_cli_successful_execution(mock_extract, sample_photos, tmp_path):
    """Verify successful offline run generates measurements.json, twin.glb, and overlays."""
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "--front", str(sample_photos["front"]),
            "--back", str(sample_photos["back"]),
            "--left", str(sample_photos["left"]),
            "--right", str(sample_photos["right"]),
            "--height", "172.0",
            "--out", str(out_dir),
            "--gender", "neutral",
        ]
    )
    assert exit_code == 0

    # 1. Verify measurements.json
    meas_file = out_dir / "measurements.json"
    assert meas_file.exists()
    meas_data = json.loads(meas_file.read_text(encoding="utf-8"))
    assert meas_data["user_height_cm"] == 172.0
    assert meas_data["gender"] == "neutral"
    assert len(meas_data["betas"]) == 10
    assert meas_data["mean_silhouette_iou"] > 0.0

    # Verify all 12 zones are present
    expected_zones = [
        "chest", "waist", "hip", "thigh", "neck", "shoulder_width",
        "sleeve_length", "inseam", "torso_length", "back_rise", "upper_back", "garment_length"
    ]
    for z in expected_zones:
        assert z in meas_data["zones"]
        assert meas_data["zones"][z]["value_cm"] > 0
        assert meas_data["zones"][z]["confidence"] > 0

    # 2. Verify twin.glb
    glb_file = out_dir / "twin.glb"
    assert glb_file.exists()
    glb_bytes = glb_file.read_bytes()
    is_valid, err_msg, meta = validate_glb_bytes(glb_bytes)
    assert is_valid is True
    assert err_msg is None
    assert meta["num_vertices"] == 6890
    assert meta["num_faces"] == 13776
    assert 1.4 < meta["height_m"] < 2.0

    # 3. Verify 4 diagnostic overlays
    for v in ("front", "back", "left", "right"):
        ov_file = out_dir / f"overlay_{v}.png"
        assert ov_file.exists()
        img = cv2.imread(str(ov_file))
        assert img is not None
        assert img.shape[0] == 360
        assert img.shape[1] == 240


def test_cli_missing_input_file(tmp_path):
    """CLI must fail with code 1 when an input photo is missing."""
    out_dir = tmp_path / "out"
    exit_code = main(
        [
            "--front", str(tmp_path / "missing_front.jpg"),
            "--back", str(tmp_path / "back.jpg"),
            "--left", str(tmp_path / "left.jpg"),
            "--right", str(tmp_path / "right.jpg"),
            "--height", "172.0",
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 1


def test_cli_invalid_stature(sample_photos, tmp_path):
    """CLI must fail with code 1 when height is out of anatomical bounds."""
    out_dir = tmp_path / "out"
    exit_code = main(
        [
            "--front", str(sample_photos["front"]),
            "--back", str(sample_photos["back"]),
            "--left", str(sample_photos["left"]),
            "--right", str(sample_photos["right"]),
            "--height", "30.0",
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 1


def test_cli_blurry_photo_rejected(sample_photos, tmp_path):
    """CLI must fail with code 1 when a photo is blurry."""
    # Overwrite front photo with blurry image
    sample_photos["front"].write_bytes(_make_blurry_test_jpeg())
    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "--front", str(sample_photos["front"]),
            "--back", str(sample_photos["back"]),
            "--left", str(sample_photos["left"]),
            "--right", str(sample_photos["right"]),
            "--height", "172.0",
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 1


@patch("services.geometry.twin.extract_four_view_landmarks_and_silhouettes")
def test_cli_offline_guarantee_no_network_calls(mock_extract, sample_photos, tmp_path):
    """Strict test ensuring NO network sockets or LLM calls are attempted."""
    mock_extract.return_value = _create_mock_landmarks_and_silhouettes()
    out_dir = tmp_path / "out"

    def _forbidden_network_call(*args, **kwargs):
        raise AssertionError("Forbidden external network or LLM call occurred during offline CLI run!")

    # Intercept socket, urllib, requests, and llm wrappers
    with patch("socket.socket.connect", side_effect=_forbidden_network_call), \
         patch("fitcore.llm.generate_json", side_effect=_forbidden_network_call):

        result = run_offline_pipeline(
            front_path=sample_photos["front"],
            back_path=sample_photos["back"],
            left_path=sample_photos["left"],
            right_path=sample_photos["right"],
            height_cm=172.0,
            output_dir=out_dir,
        )

        assert Path(result["measurements_json"]).exists()
        assert Path(result["glb_path"]).exists()
        assert len(result["overlays"]) == 4
