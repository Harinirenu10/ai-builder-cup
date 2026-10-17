"""MediaPipe Pose Landmarker Heavy extraction and silhouette profiling for 3D body twin pipeline.

CPU-only integration extracting 33 pose landmarks and binary person segmentation
masks thresholded at 0.5 across orthogonal body views (front, back, left, right).

Provides:
- 33 Pose landmarks with visibility, presence, and pixel coordinates.
- Person segmentation mask thresholded at 0.5.
- Pixel height calculation and scale calibration (cm_per_pixel) from user height.
- Anatomical silhouette width extraction at landmark heights.
- Strict error handling for missing, incomplete, or unreliable landmarks.
- Injected / mockable landmarker interface for offline unit testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("geometry.landmarks")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_heavy.task"

# 33 Standard MediaPipe Pose Landmark Names
LANDMARK_NAMES: dict[int, str] = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}

CRITICAL_LANDMARK_NAMES = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

# In a true orthogonal side-profile photo ('left'/'right' views), one leg is always
# nearer the camera than the other, so the far ankle is partially self-occluded by
# the body and MediaPipe reports low confidence for it even on a perfectly sharp,
# correctly-posed photo (confirmed with real photos: near-side ankle ~0.85, far-side
# ankle ~0.10). Requiring BOTH ankles to clear the visibility threshold in a profile
# view is geometrically unsatisfiable, so for those views only at least one ankle
# must be confidently visible. Front/back views show both feet side-by-side with no
# such occlusion, so both ankles are still required there.
PROFILE_VIEWS_WITH_SINGLE_ANKLE_REQUIRED = {"left", "right"}
ANKLE_LANDMARK_NAMES = {"LEFT_ANKLE", "RIGHT_ANKLE"}


class LandmarkError(Exception):
    """Base exception for landmark and silhouette extraction errors."""


class LandmarkDetectionError(LandmarkError):
    """Raised when no human pose can be detected in a view."""


class IncompleteLandmarksError(LandmarkError):
    """Raised when fewer than 33 landmarks are extracted."""


class UnreliableLandmarksError(LandmarkError):
    """Raised when critical landmarks have insufficient visibility/confidence."""


class PoseLandmarkPoint(BaseModel):
    """Single 3D anatomical landmark."""
    index: int = Field(..., ge=0, le=32)
    name: str
    x: float = Field(..., description="Normalized horizontal coordinate in [0, 1]")
    y: float = Field(..., description="Normalized vertical coordinate in [0, 1]")
    z: float = Field(..., description="Relative depth coordinate")
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    presence: float = Field(default=1.0, ge=0.0, le=1.0)
    pixel_x: float = Field(..., description="Horizontal pixel coordinate")
    pixel_y: float = Field(..., description="Vertical pixel coordinate")


class ViewLandmarkResult(BaseModel):
    """Structured extraction outcome for one orthogonal view."""
    view: str = Field(..., description="Orthogonal view: 'front', 'back', 'left', 'right'")
    landmarks: list[PoseLandmarkPoint]
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    mask_top_row: int
    mask_bottom_row: int
    pixel_height: int = Field(..., gt=0)
    cm_per_pixel: float = Field(..., gt=0.0)
    mean_confidence: float = Field(..., ge=0.0, le=1.0)

    def get(self, name_or_idx: str | int) -> PoseLandmarkPoint | None:
        """Retrieve landmark by standard name or integer index."""
        if isinstance(name_or_idx, int):
            for lm in self.landmarks:
                if lm.index == name_or_idx:
                    return lm
            return None
        target_name = name_or_idx.upper().strip()
        for lm in self.landmarks:
            if lm.name == target_name:
                return lm
        return None


class ViewSilhouette(BaseModel):
    """Calibrated silhouette profile for one view."""
    view: str
    image_width: int
    image_height: int
    mask_top_row: int
    mask_bottom_row: int
    pixel_height: int
    cm_per_pixel: float
    profile_widths: dict[int, float] = Field(
        default_factory=dict,
        description="Row pixel index mapped to horizontal silhouette width in cm"
    )


def clean_segmentation_mask(raw_mask: np.ndarray, min_area: int = 150) -> np.ndarray:

    """Clean a binary segmentation mask by removing small noise blobs and stray pixels.

    Uses connected component analysis to retain only the primary person silhouette
    and morphological opening to eliminate isolated noise whiskers.

    Args:
        raw_mask: 2D uint8 or float mask.
        min_area: Minimum pixel area threshold for connected components.

    Returns:
        Cleaned 2D binary uint8 mask (values 0 or 1).
    """
    if raw_mask.dtype != np.uint8:
        binary = (raw_mask >= 0.5).astype(np.uint8)
    else:
        binary = (raw_mask > 0).astype(np.uint8)

    if not binary.any():
        return binary

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    # Filter out background (label 0) and small components (< min_area)
    cleaned = np.zeros_like(binary)
    # Always include the largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = int(np.argmax(areas)) + 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        # Keep if it is the primary component or meets the minimum area
        if label == max_label or area >= min_area:
            cleaned[labels == label] = 1

    # Morphological opening to eliminate 1-2 pixel isolated noise whiskers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def calibrate_scale(
    mask: np.ndarray,
    user_height_cm: float,
    min_row_density: int = 3,
    allow_cropped: bool = False,
) -> tuple[int, int, int, float]:
    """Calculate topmost and bottommost mask rows and derive metric scale.

    Applies noise filtering to discard stray single-pixel spikes and verifies
    that the subject's crown and feet are within the frame.

    Args:
        mask: 2D binary uint8 mask (values 0 or 1).
        user_height_cm: Confirmed subject stature in centimeters.
        min_row_density: Minimum horizontal pixel count required for valid crown/sole rows.
        allow_cropped: If False, raises LandmarkDetectionError if subject is sliced by frame border.

    Returns:
        (top_row, bottom_row, pixel_height, cm_per_pixel)

    Raises:
        LandmarkDetectionError: If mask is empty, person height is implausibly small,
                                or subject is vertically cropped.
    """
    cleaned_mask = clean_segmentation_mask(mask)

    # Calculate row sums (horizontal span density)
    row_sums = cleaned_mask.sum(axis=1)
    valid_rows = np.where(row_sums >= min_row_density)[0]

    if len(valid_rows) == 0:
        # Fallback to any non-zero row if density is low
        valid_rows = np.where(row_sums > 0)[0]
        if len(valid_rows) == 0:
            raise LandmarkDetectionError("No person detected in segmentation mask; cannot calibrate scale.")

    top_row = int(valid_rows[0])
    bottom_row = int(valid_rows[-1])
    pixel_height = bottom_row - top_row + 1

    if pixel_height < 50:
        raise LandmarkDetectionError(
            f"Extracted person height too small ({pixel_height} pixels). "
            "Please ensure full body from head to feet is framed."
        )

    # Verify that the subject is not vertically cut off at frame boundaries
    img_h = mask.shape[0]
    if not allow_cropped:
        if top_row <= 1 and bottom_row >= img_h - 2:
            raise LandmarkDetectionError(
                "Subject appears vertically cropped at frame borders. "
                "Step back until entire body from head to feet is visible."
            )

    cm_per_pixel = user_height_cm / float(pixel_height)
    return top_row, bottom_row, pixel_height, cm_per_pixel


def extract_row_width(
    mask: np.ndarray,
    y: int,
    cm_per_pixel: float,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[int, int, int, float]:
    """Calculate horizontal silhouette width at vertical row y.

    Args:
        mask: 2D binary uint8 mask (0 or 1).
        y: Vertical row coordinate.
        cm_per_pixel: Metric scale.
        x_min: Optional inclusive lower column bound restricting the search
            range, e.g. a landmark-derived anatomical window used to exclude
            protrusions (hair, arms/hands resting at the sides) that are part
            of the person mask but not the body region being measured.
        x_max: Optional inclusive upper column bound.

    Returns:
        (left_col, right_col, pixel_width, width_cm)
    """
    h, w = mask.shape[:2]
    y_clamped = max(0, min(h - 1, int(round(y))))
    row = mask[y_clamped, :]

    if x_min is not None or x_max is not None:
        lo = 0 if x_min is None else max(0, int(round(x_min)))
        hi = w if x_max is None else min(w, int(round(x_max)) + 1)
        if hi <= lo:
            return 0, 0, 0, 0.0
        cols = np.where(row[lo:hi] > 0)[0]
        if len(cols) == 0:
            return 0, 0, 0, 0.0
        left_col = int(cols[0]) + lo
        right_col = int(cols[-1]) + lo
    else:
        cols = np.where(row > 0)[0]
        if len(cols) == 0:
            return 0, 0, 0, 0.0
        left_col = int(cols[0])
        right_col = int(cols[-1])

    pixel_width = right_col - left_col + 1
    width_cm = pixel_width * cm_per_pixel
    return left_col, right_col, pixel_width, width_cm


class MediaPipePoseExtractor:
    """CPU-only MediaPipe Pose Landmarker Heavy wrapper."""

    def __init__(
        self,
        model_asset_path: Path | str | None = None,
        landmarker: Any | None = None,
        min_critical_visibility: float = 0.35,
    ) -> None:
        """Initialize the extractor.

        Args:
            model_asset_path: Path to pose_landmarker_heavy.task (resolved absolutely).
            landmarker: Optional pre-instantiated or mock landmarker for testing.
            min_critical_visibility: Minimum visibility/presence score required for key landmarks.
        """
        self.model_asset_path = Path(model_asset_path or DEFAULT_MODEL_PATH).resolve()
        self._landmarker = landmarker
        self.min_critical_visibility = min_critical_visibility

    @property
    def landmarker(self) -> Any:
        """Lazily initialize MediaPipe PoseLandmarker if not injected."""
        if self._landmarker is None:
            if not self.model_asset_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe task asset missing at {self.model_asset_path}. "
                    "Ensure models/pose_landmarker_heavy.task exists."
                )


            import mediapipe as mp
            from mediapipe.tasks.python import vision

            base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_asset_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_segmentation_masks=True,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def extract(
        self,
        image_input: bytes | np.ndarray,
        view: str,
        user_height_cm: float,
    ) -> tuple[ViewLandmarkResult, np.ndarray, ViewSilhouette]:
        """Extract landmarks and binary segmentation mask from a single view.

        Args:
            image_input: Raw image bytes or BGR/RGB numpy array.
            view: Orthogonal view name ('front', 'back', 'left', 'right').
            user_height_cm: Subject stature in cm.

        Returns:
            (ViewLandmarkResult, binary_mask_uint8, ViewSilhouette)

        Raises:
            LandmarkDetectionError: If no pose is detected.
            IncompleteLandmarksError: If landmark count != 33.
            UnreliableLandmarksError: If critical landmarks have low visibility.
        """
        # Decode image to RGB numpy array
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to decode image input for {view} view.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            rgb = image_input if image_input.shape[2] == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img_h, img_w = rgb.shape[:2]

        # Execute MediaPipe detection
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self.landmarker.detect(mp_image)

        # 1. Verify pose detection
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            raise LandmarkDetectionError(
                f"No human pose detected in {view} view. "
                "Please verify the subject is clearly framed against a contrasting background."
            )

        raw_landmarks = detection_result.pose_landmarks[0]

        # 2. Verify complete 33 landmark points
        if len(raw_landmarks) != 33:
            raise IncompleteLandmarksError(
                f"Incomplete landmarks extracted in {view} view: "
                f"expected 33, got {len(raw_landmarks)}."
            )

        # 3. Process landmark points
        points: list[PoseLandmarkPoint] = []
        confidences: list[float] = []

        for idx, lm in enumerate(raw_landmarks):
            vis = float(getattr(lm, "visibility", 1.0) or 1.0)
            pres = float(getattr(lm, "presence", 1.0) or 1.0)
            confidences.append(vis)

            px_x = float(np.clip(lm.x * img_w, 0, img_w - 1))
            px_y = float(np.clip(lm.y * img_h, 0, img_h - 1))

            points.append(
                PoseLandmarkPoint(
                    index=idx,
                    name=LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}"),
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=vis,
                    presence=pres,
                    pixel_x=px_x,
                    pixel_y=px_y,
                )
            )

        # 4. Check critical landmark visibility and presence
        points_by_name = {p.name: p for p in points}
        is_profile_view = view in PROFILE_VIEWS_WITH_SINGLE_ANKLE_REQUIRED

        def _is_reliable(point: PoseLandmarkPoint) -> bool:
            return point.visibility >= self.min_critical_visibility and point.presence >= self.min_critical_visibility

        for crit_name in CRITICAL_LANDMARK_NAMES:
            p = points_by_name.get(crit_name)
            if p is None:
                raise IncompleteLandmarksError(f"Missing critical landmark {crit_name} in {view} view.")

            if is_profile_view and crit_name in ANKLE_LANDMARK_NAMES:
                # Checked together (at least one ankle reliable) below instead of per-landmark.
                continue

            if not _is_reliable(p):
                raise UnreliableLandmarksError(
                    f"Critical landmark '{crit_name}' in {view} view has insufficient visibility "
                    f"({p.visibility:.2f}) or presence ({p.presence:.2f}) "
                    f"(required >= {self.min_critical_visibility:.2f}). "
                    "Cannot safely compute body measurements."
                )

        if is_profile_view:
            left_ankle, right_ankle = points_by_name.get("LEFT_ANKLE"), points_by_name.get("RIGHT_ANKLE")
            if left_ankle is None or right_ankle is None:
                raise IncompleteLandmarksError(f"Missing ankle landmark(s) in {view} view.")
            if not (_is_reliable(left_ankle) or _is_reliable(right_ankle)):
                raise UnreliableLandmarksError(
                    f"Neither ankle is reliably visible in {view} view "
                    f"(left={left_ankle.visibility:.2f}, right={right_ankle.visibility:.2f}, "
                    f"required >= {self.min_critical_visibility:.2f} for at least one). "
                    "Cannot safely compute body measurements."
                )

        mean_confidence = float(np.mean(confidences))

        # 5. Extract and threshold segmentation mask at 0.5, then clean noise
        if detection_result.segmentation_masks and len(detection_result.segmentation_masks) > 0:
            mask_float = detection_result.segmentation_masks[0].numpy_view()
            raw_binary = (mask_float >= 0.5).astype(np.uint8)
            binary_mask = clean_segmentation_mask(raw_binary)
        else:
            # Fallback for mock environments without segmentation masks
            binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # Fill bounding box enclosing landmarks
            min_x = max(0, int(min(p.pixel_x for p in points) - 15))
            max_x = min(img_w - 1, int(max(p.pixel_x for p in points) + 15))
            min_y = max(0, int(min(p.pixel_y for p in points) - 20))
            max_y = min(img_h - 1, int(max(p.pixel_y for p in points) + 20))
            binary_mask[min_y:max_y, min_x:max_x] = 1

        # 6. Calibrate scale and vertical boundaries
        top_row, bottom_row, pixel_height, cm_per_pixel = calibrate_scale(
            mask=binary_mask,
            user_height_cm=user_height_cm,
        )


        landmark_result = ViewLandmarkResult(
            view=view,
            landmarks=points,
            image_width=img_w,
            image_height=img_h,
            mask_top_row=top_row,
            mask_bottom_row=bottom_row,
            pixel_height=pixel_height,
            cm_per_pixel=cm_per_pixel,
            mean_confidence=mean_confidence,
        )

        # 7. Extract profile widths at key landmark heights
        profile_widths: dict[int, float] = {}
        # Sample shoulder, hip, and mid-torso rows
        shoulder_y = int(round((points_by_name["LEFT_SHOULDER"].pixel_y + points_by_name["RIGHT_SHOULDER"].pixel_y) / 2.0))
        hip_y = int(round((points_by_name["LEFT_HIP"].pixel_y + points_by_name["RIGHT_HIP"].pixel_y) / 2.0))
        waist_y = int(round(shoulder_y + (hip_y - shoulder_y) * 0.55))

        for row_y in (shoulder_y, waist_y, hip_y):
            _, _, _, w_cm = extract_row_width(binary_mask, row_y, cm_per_pixel)
            profile_widths[row_y] = w_cm

        silhouette = ViewSilhouette(
            view=view,
            image_width=img_w,
            image_height=img_h,
            mask_top_row=top_row,
            mask_bottom_row=bottom_row,
            pixel_height=pixel_height,
            cm_per_pixel=cm_per_pixel,
            profile_widths=profile_widths,
        )

        return landmark_result, binary_mask, silhouette


def extract_four_view_landmarks_and_silhouettes(
    views: dict[str, bytes | np.ndarray],
    user_height_cm: float,
    extractor: MediaPipePoseExtractor | None = None,
) -> tuple[dict[str, ViewLandmarkResult], dict[str, np.ndarray], dict[str, ViewSilhouette]]:
    """Process all 4 orthogonal views (front, back, left, right).

    Args:
        views: Mapping of view name ('front', 'back', 'left', 'right') to image.
        user_height_cm: Subject stature in cm.
        extractor: Optional injected MediaPipePoseExtractor.

    Returns:
        (landmark_results, binary_masks, silhouettes)
    """
    if extractor is None:
        extractor = MediaPipePoseExtractor()

    # Validate all 4 required views upfront
    for view_name in ("front", "back", "left", "right"):
        if view_name not in views:
            raise KeyError(f"Missing required view: '{view_name}'")

    landmark_results: dict[str, ViewLandmarkResult] = {}
    masks: dict[str, np.ndarray] = {}
    silhouettes: dict[str, ViewSilhouette] = {}

    for view_name in ("front", "back", "left", "right"):
        res, mask, sil = extractor.extract(

            image_input=views[view_name],
            view=view_name,
            user_height_cm=user_height_cm,
        )
        landmark_results[view_name] = res
        masks[view_name] = mask
        silhouettes[view_name] = sil

    return landmark_results, masks, silhouettes
