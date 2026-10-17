"""Unit tests for services/geometry/landmarks.py.

Verifies:
1. Landmark data models and retrieval helpers.
2. Scale calibration (cm_per_pixel) from segmentation masks.
3. Silhouette horizontal row width calculations.
4. Error handling for missing poses, incomplete landmark counts, and empty masks.
5. Injected mock landmarker for deterministic offline testing.
6. 4-view orchestration across front, back, left, right.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from services.geometry.landmarks import (
    IncompleteLandmarksError,
    LandmarkDetectionError,
    UnreliableLandmarksError,
    MediaPipePoseExtractor,
    PoseLandmarkPoint,
    ViewLandmarkResult,
    ViewSilhouette,
    calibrate_scale,
    clean_segmentation_mask,
    extract_four_view_landmarks_and_silhouettes,
    extract_row_width,
)



def _create_mock_landmark(idx: int, x: float = 0.5, y: float = 0.5, z: float = 0.0, vis: float = 0.9):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = vis
    lm.presence = 1.0
    return lm


def _create_mock_33_landmarks(width: int = 400, height: int = 800) -> list[MagicMock]:
    """Create 33 simulated landmarks in realistic vertical proportions."""
    landmarks = []
    for i in range(33):
        # Y spans from head (0.1) to feet (0.9)
        norm_y = 0.1 + (i / 33.0) * 0.8
        norm_x = 0.5 + (0.05 if i % 2 == 0 else -0.05)
        landmarks.append(_create_mock_landmark(i, x=norm_x, y=norm_y))
    return landmarks


def _create_mock_mask_array(width: int = 400, height: int = 800, top: int = 80, bottom: int = 720) -> np.ndarray:
    """Create a simulated binary person mask with a rectangular silhouette."""
    mask = np.zeros((height, width), dtype=np.float32)
    # Person silhouette between top and bottom rows, width 100 pixels in center
    center_x = width // 2
    mask[top:bottom, center_x - 50:center_x + 50] = 0.9  # > 0.5 probability
    return mask


def test_calibrate_scale_valid():
    """Verify scale calibration calculates correct cm_per_pixel."""
    mask = np.zeros((1000, 500), dtype=np.uint8)
    # Subject occupies rows 100 to 899 (pixel height = 800)
    mask[100:900, 200:300] = 1

    top, bottom, px_h, cm_per_px = calibrate_scale(mask, user_height_cm=180.0)
    assert top == 100
    assert bottom == 899
    assert px_h == 800
    assert abs(cm_per_px - (180.0 / 800.0)) < 1e-5


def test_calibrate_scale_empty_mask_raises():
    """Empty mask with no detected person must raise LandmarkDetectionError."""
    empty_mask = np.zeros((500, 500), dtype=np.uint8)
    with pytest.raises(LandmarkDetectionError, match="No person detected"):
        calibrate_scale(empty_mask, user_height_cm=175.0)


def test_calibrate_scale_too_small_raises():
    """Mask with implausibly small height must raise LandmarkDetectionError."""
    tiny_mask = np.zeros((500, 500), dtype=np.uint8)
    tiny_mask[200:220, 240:260] = 1  # only 20 pixels high
    with pytest.raises(LandmarkDetectionError, match="Extracted person height too small"):
        calibrate_scale(tiny_mask, user_height_cm=175.0)


def test_extract_row_width():
    """Verify horizontal width extraction at row y."""
    mask = np.zeros((500, 400), dtype=np.uint8)
    # Row 200 has silhouette from column 150 to 249 (100 pixels)
    mask[200, 150:250] = 1

    cm_per_px = 0.25  # 1 pixel = 0.25 cm
    left, right, px_w, w_cm = extract_row_width(mask, y=200, cm_per_pixel=cm_per_px)
    assert left == 150
    assert right == 249
    assert px_w == 100
    assert abs(w_cm - 25.0) < 1e-4

    # Row with no mask content returns 0
    left0, right0, px_w0, w_cm0 = extract_row_width(mask, y=50, cm_per_pixel=cm_per_px)
    assert px_w0 == 0
    assert w_cm0 == 0.0


def test_extract_row_width_x_bounds_exclude_contamination_outside_window():
    """Regression test for the x_min/x_max bounding mechanism added to fix
    hair/hand contamination in body measurements: a row containing a narrow
    true silhouette plus a much wider contaminating blob (e.g. hair, or an
    arm/hand resting against the torso) must, when bounded to a window around
    the narrow region, return only the narrow region's width -- not the full
    contaminated extent.
    """
    mask = np.zeros((500, 400), dtype=np.uint8)
    # Wide contamination spans columns 50-349 (300px); the true silhouette is
    # a narrower sub-region, columns 150-249 (100px), within that same row.
    mask[200, 50:350] = 1
    cm_per_px = 0.25

    # Unbounded: the wide contamination dominates.
    _, _, px_w_unbounded, _ = extract_row_width(mask, y=200, cm_per_pixel=cm_per_px)
    assert px_w_unbounded == 300

    # Bounded to a window around the true silhouette: contamination excluded.
    left, right, px_w, w_cm = extract_row_width(
        mask, y=200, cm_per_pixel=cm_per_px, x_min=150.0, x_max=249.0
    )
    assert left == 150
    assert right == 249
    assert px_w == 100
    assert abs(w_cm - 25.0) < 1e-4

    # A window entirely outside the mask content returns 0, not a crash.
    left_e, right_e, px_w_e, w_cm_e = extract_row_width(
        mask, y=200, cm_per_pixel=cm_per_px, x_min=360.0, x_max=390.0
    )
    assert px_w_e == 0
    assert w_cm_e == 0.0


def test_extractor_with_mock_landmarker():
    """Verify MediaPipePoseExtractor correctly parses landmarks and thresholded mask."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()

    mock_landmarks = _create_mock_33_landmarks()
    mock_detection.pose_landmarks = [mock_landmarks]

    # Segmentation mask mock
    mock_mask_obj = MagicMock()
    mock_mask_obj.numpy_view.return_value = _create_mock_mask_array(width=400, height=800, top=100, bottom=700)
    mock_detection.segmentation_masks = [mock_mask_obj]

    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker)
    dummy_img = np.zeros((800, 400, 3), dtype=np.uint8)

    res, binary_mask, sil = extractor.extract(dummy_img, view="front", user_height_cm=175.0)

    # 1. Verify landmarks
    assert len(res.landmarks) == 33
    assert res.get("LEFT_SHOULDER") is not None
    assert res.get("RIGHT_HIP") is not None
    assert res.get(11).name == "LEFT_SHOULDER"

    # 2. Verify binary mask thresholding at 0.5
    assert binary_mask.shape == (800, 400)
    assert binary_mask.dtype == np.uint8
    assert binary_mask[400, 200] == 1
    assert binary_mask[10, 10] == 0

    # 3. Verify scale calibration
    assert res.pixel_height == 600
    assert abs(res.cm_per_pixel - (175.0 / 600.0)) < 1e-4


    # 4. Verify silhouette profile
    assert isinstance(sil, ViewSilhouette)
    assert len(sil.profile_widths) >= 3


def test_extractor_missing_pose_raises():
    """If landmarker detects 0 poses, LandmarkDetectionError must be raised."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()
    mock_detection.pose_landmarks = []  # No poses
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker)
    dummy_img = np.zeros((600, 400, 3), dtype=np.uint8)

    with pytest.raises(LandmarkDetectionError, match="No human pose detected"):
        extractor.extract(dummy_img, view="front", user_height_cm=170.0)


def test_extractor_incomplete_landmarks_raises():
    """If landmarker returns fewer than 33 points, IncompleteLandmarksError must be raised."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()
    mock_detection.pose_landmarks = [_create_mock_33_landmarks()[:25]]  # Only 25 landmarks
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker)
    dummy_img = np.zeros((600, 400, 3), dtype=np.uint8)

    with pytest.raises(IncompleteLandmarksError, match="Incomplete landmarks extracted"):
        extractor.extract(dummy_img, view="back", user_height_cm=170.0)


def test_extract_four_view_landmarks_and_silhouettes():
    """Verify orchestration of all 4 orthogonal views."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()
    mock_detection.pose_landmarks = [_create_mock_33_landmarks()]
    mock_mask = MagicMock()
    mock_mask.numpy_view.return_value = _create_mock_mask_array()
    mock_detection.segmentation_masks = [mock_mask]
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker)

    views = {
        "front": np.zeros((800, 400, 3), dtype=np.uint8),
        "back": np.zeros((800, 400, 3), dtype=np.uint8),
        "left": np.zeros((800, 400, 3), dtype=np.uint8),
        "right": np.zeros((800, 400, 3), dtype=np.uint8),
    }

    results, masks, silhouettes = extract_four_view_landmarks_and_silhouettes(
        views=views,
        user_height_cm=178.0,
        extractor=extractor,
    )

    for v in ("front", "back", "left", "right"):
        assert v in results
        assert v in masks
        assert v in silhouettes
        assert len(results[v].landmarks) == 33
        assert masks[v].shape == (800, 400)


def test_extract_four_view_missing_key_raises():
    """Missing any of the 4 required views raises KeyError."""
    extractor = MediaPipePoseExtractor(landmarker=MagicMock())
    incomplete_views = {
        "front": np.zeros((800, 400, 3), dtype=np.uint8),
        "back": np.zeros((800, 400, 3), dtype=np.uint8),
    }
    with pytest.raises(KeyError, match="Missing required view: 'left'"):
        extract_four_view_landmarks_and_silhouettes(
            views=incomplete_views,
            user_height_cm=175.0,
            extractor=extractor,
        )


def test_clean_segmentation_mask_removes_stray_noise():
    """Verify clean_segmentation_mask eliminates small floating blobs and retains main body."""
    mask = np.zeros((600, 400), dtype=np.uint8)
    # Main person body (large area > 10,000 pixels)
    mask[100:500, 150:250] = 1

    # Stray noise pixels and tiny blobs in corners (area < 50 pixels)
    mask[10, 10] = 1
    mask[15, 20:25] = 1
    mask[590, 390] = 1

    cleaned = clean_segmentation_mask(mask, min_area=150)
    # Stray noise pixels must be removed
    assert cleaned[10, 10] == 0
    assert cleaned[15, 22] == 0
    assert cleaned[590, 390] == 0

    # Main body must be preserved
    assert cleaned[300, 200] == 1
    assert cleaned[105:495, 155:245].all() == 1


def test_calibrate_scale_ignores_stray_pixels():
    """Verify calibrate_scale filters out isolated noise pixels and computes height from the true body."""
    mask = np.zeros((1000, 500), dtype=np.uint8)
    # Main body from rows 100 to 899 (800 pixels)
    mask[100:900, 200:300] = 1

    # Single stray pixel at row 2 and row 998
    mask[2, 250] = 1
    mask[998, 250] = 1

    top, bottom, px_h, cm_per_px = calibrate_scale(mask, user_height_cm=180.0, allow_cropped=True)
    # Scale should be derived from the body, not the stray pixels at 2 and 998
    assert top == 100
    assert bottom == 899
    assert px_h == 800


def test_calibrate_scale_detects_cropped_body():
    """Verify calibrate_scale raises LandmarkDetectionError when body is sliced at frame edges."""
    mask = np.zeros((800, 400), dtype=np.uint8)
    # Body spans from row 0 to row 799 (touches top and bottom borders)
    mask[0:800, 150:250] = 1

    with pytest.raises(LandmarkDetectionError, match="Subject appears vertically cropped"):
        calibrate_scale(mask, user_height_cm=175.0, allow_cropped=False)


def test_extractor_unreliable_critical_landmark_raises():
    """Verify extract raises UnreliableLandmarksError if critical landmarks have low visibility/presence."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()

    mock_landmarks = _create_mock_33_landmarks()
    # Degrade LEFT_SHOULDER (index 11) visibility to 0.15 (< 0.35 threshold)
    mock_landmarks[11].visibility = 0.15
    mock_landmarks[11].presence = 0.20

    mock_detection.pose_landmarks = [mock_landmarks]
    mock_mask = MagicMock()
    mock_mask.numpy_view.return_value = _create_mock_mask_array()
    mock_detection.segmentation_masks = [mock_mask]
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker, min_critical_visibility=0.35)
    dummy_img = np.zeros((800, 400, 3), dtype=np.uint8)

    with pytest.raises(UnreliableLandmarksError, match="insufficient visibility"):
        extractor.extract(dummy_img, view="front", user_height_cm=175.0)


def test_profile_view_passes_with_only_near_ankle_reliable():
    """Regression test: in a true side-profile photo, the far leg is partially
    self-occluded by the body, so MediaPipe reports low confidence for the far
    ankle even on a sharp, correctly-posed photo (confirmed with real photos:
    near-ankle ~0.85, far-ankle ~0.10). Requiring both ankles in a 'left'/'right'
    view is geometrically unsatisfiable, so at least one reliable ankle must be
    enough there.
    """
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()

    mock_landmarks = _create_mock_33_landmarks()
    mock_landmarks[27].visibility = 0.10  # LEFT_ANKLE: occluded far leg
    mock_landmarks[27].presence = 1.0
    mock_landmarks[28].visibility = 0.85  # RIGHT_ANKLE: visible near leg
    mock_landmarks[28].presence = 1.0

    mock_detection.pose_landmarks = [mock_landmarks]
    mock_mask = MagicMock()
    mock_mask.numpy_view.return_value = _create_mock_mask_array()
    mock_detection.segmentation_masks = [mock_mask]
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker, min_critical_visibility=0.35)
    dummy_img = np.zeros((800, 400, 3), dtype=np.uint8)

    result, _, _ = extractor.extract(dummy_img, view="left", user_height_cm=175.0)
    assert result.view == "left"


def test_profile_view_raises_when_both_ankles_unreliable():
    """Both ankles unreliable in a profile view must still raise (there is no
    visible ankle to fall back on at all)."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()

    mock_landmarks = _create_mock_33_landmarks()
    mock_landmarks[27].visibility = 0.10  # LEFT_ANKLE
    mock_landmarks[27].presence = 1.0
    mock_landmarks[28].visibility = 0.05  # RIGHT_ANKLE
    mock_landmarks[28].presence = 1.0

    mock_detection.pose_landmarks = [mock_landmarks]
    mock_mask = MagicMock()
    mock_mask.numpy_view.return_value = _create_mock_mask_array()
    mock_detection.segmentation_masks = [mock_mask]
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker, min_critical_visibility=0.35)
    dummy_img = np.zeros((800, 400, 3), dtype=np.uint8)

    with pytest.raises(UnreliableLandmarksError, match="Neither ankle"):
        extractor.extract(dummy_img, view="right", user_height_cm=175.0)


def test_front_view_still_requires_both_ankles_reliable():
    """Front/back views show both feet side-by-side with no occlusion, so the
    strict both-ankles-required check must still apply there (unlike profile
    views)."""
    mock_landmarker = MagicMock()
    mock_detection = MagicMock()

    mock_landmarks = _create_mock_33_landmarks()
    mock_landmarks[27].visibility = 0.10  # LEFT_ANKLE: unreliable even in front view
    mock_landmarks[27].presence = 1.0
    mock_landmarks[28].visibility = 0.90  # RIGHT_ANKLE: reliable

    mock_detection.pose_landmarks = [mock_landmarks]
    mock_mask = MagicMock()
    mock_mask.numpy_view.return_value = _create_mock_mask_array()
    mock_detection.segmentation_masks = [mock_mask]
    mock_landmarker.detect.return_value = mock_detection

    extractor = MediaPipePoseExtractor(landmarker=mock_landmarker, min_critical_visibility=0.35)
    dummy_img = np.zeros((800, 400, 3), dtype=np.uint8)

    with pytest.raises(UnreliableLandmarksError, match="insufficient visibility"):
        extractor.extract(dummy_img, view="front", user_height_cm=175.0)


def test_model_asset_path_is_absolute_and_cwd_independent():
    """Verify model asset path resolves to an absolute path regardless of current directory."""
    extractor = MediaPipePoseExtractor()
    assert extractor.model_asset_path.is_absolute()
    assert extractor.model_asset_path.name == "pose_landmarker_heavy.task"
    assert extractor.model_asset_path.exists()

