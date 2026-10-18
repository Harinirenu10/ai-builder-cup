"""Domain schemas for garment flat-lay photo measurement.

A garment is photographed flat on a contrasting surface next to a full A4
sheet (210 x 297 mm), which is used as the only available physical scale
reference in the frame.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from fitcore.schemas import MeasurementZone


class GarmentCategory(str, Enum):
    """Supported garment categories for flat-lay measurement."""
    tops = "tops"
    kurtas = "kurtas"
    bottoms = "bottoms"


# Category-specific keypoint vocabularies. Tops and kurtas share the same
# construction landmarks (both are upper-body garments measured flat).
TOP_KEYPOINT_NAMES: tuple[str, ...] = (
    "left_armpit",
    "right_armpit",
    "left_shoulder_seam",
    "right_shoulder_seam",
    "back_neck_point",
    "hem_left",
    "hem_right",
    "hem_center",
    "left_cuff_outer",
    "left_cuff_inner",
)

BOTTOM_KEYPOINT_NAMES: tuple[str, ...] = (
    "waist_left",
    "waist_right",
    "back_waist_center",
    "crotch",
    "hip_left",
    "hip_right",
    "left_hem_inner",
    "left_hem_outer",
)

CATEGORY_KEYPOINTS: dict[GarmentCategory, tuple[str, ...]] = {
    GarmentCategory.tops: TOP_KEYPOINT_NAMES,
    GarmentCategory.kurtas: TOP_KEYPOINT_NAMES,
    GarmentCategory.bottoms: BOTTOM_KEYPOINT_NAMES,
}

QUALITY_FLAG_NAMES: tuple[str, ...] = (
    "folded_or_overlapping",
    "not_flat",
    "partly_out_of_frame",
)


class KeypointXY(BaseModel):
    """A single garment keypoint, normalized to a 0-1000 coordinate space
    over the rectified (A4-scaled) canvas, independent of actual pixel size.
    """
    x: float = Field(..., ge=0.0, le=1000.0)
    y: float = Field(..., ge=0.0, le=1000.0)


class GarmentQualityFlags(BaseModel):
    """Quality flags describing conditions that reduce measurement trust."""
    folded_or_overlapping: bool = Field(default=False, description="Garment is folded or overlapping itself")
    not_flat: bool = Field(default=False, description="Garment is not laid fully flat (wrinkled/bunched)")
    partly_out_of_frame: bool = Field(default=False, description="Part of the garment is outside the photo frame")


class GeminiGarmentKeypoints(BaseModel):
    """Structured Gemini Flash response: category-specific normalized keypoints plus quality flags."""
    keypoints: dict[str, KeypointXY] = Field(
        default_factory=dict,
        description="Map of keypoint name to normalized (0-1000) coordinate on the rectified image",
    )
    quality: GarmentQualityFlags = Field(default_factory=GarmentQualityFlags)
    notes: str = Field(default="", description="Short objective note about the garment photo, no body commentary")


class A4DetectionResult(BaseModel):
    """Outcome of locating the A4 reference sheet in the source photo."""
    a4_found: bool = Field(..., description="Whether a valid A4-ratio quadrilateral was located")
    a4_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence the detected quad is an A4 sheet")
    skew_degrees: float = Field(default=0.0, description="Rotation of the detected A4 sheet relative to horizontal, in degrees")


# Measurement computation reference: documents exactly which two keypoints
# (by name) define each measurement, in centimeters, on the rectified image.
GARMENT_ZONE_DEFINITIONS: dict[str, dict[str, Any]] = {
    # tops / kurtas
    "chest_flat": {"category": "tops", "from": "left_armpit", "to": "right_armpit",
                   "method": "Horizontal flat width across the chest at armpit level."},
    "chest_circ": {"category": "tops", "derived_from": "chest_flat", "factor": 2.0,
                   "method": "2 x chest_flat (flat-lay captures half the closed body circumference)."},
    "shoulder_width": {"category": "tops", "from": "left_shoulder_seam", "to": "right_shoulder_seam",
                        "method": "Straight-line distance between the two shoulder seam points."},
    "sleeve_length": {"category": "tops", "from": "left_shoulder_seam", "to": "left_cuff_outer",
                       "method": "Distance from the shoulder seam to the outer cuff edge along the sleeve."},
    "garment_length": {"category": "tops", "from": "back_neck_point", "to": "hem_center",
                        "method": "Distance from the back neck point to the center hem."},
    "hem_width": {"category": "tops", "from": "hem_left", "to": "hem_right",
                  "method": "Flat width across the hemline."},
    # bottoms
    "waist_flat": {"category": "bottoms", "from": "waist_left", "to": "waist_right",
                   "method": "Flat width across the waistband."},
    "waist_circ": {"category": "bottoms", "derived_from": "waist_flat", "factor": 2.0,
                   "method": "2 x waist_flat (flat-lay captures half the closed waistband circumference)."},
    "hip_flat": {"category": "bottoms", "from": "hip_left", "to": "hip_right",
                 "method": "Flat width across the hip line."},
    "inseam": {"category": "bottoms", "from": "crotch", "to": "left_hem_inner",
               "method": "Distance from the crotch point to the inner hem edge."},
    "leg_opening": {"category": "bottoms", "from": "left_hem_inner", "to": "left_hem_outer",
                    "method": "Flat width across the leg opening/hem."},
}


class GarmentFlatMeasurements(BaseModel):
    """Deterministic flat-lay measurements in centimeters, with confidence per field.

    Only fields relevant to the garment's category are populated; the rest are None.
    See GARMENT_ZONE_DEFINITIONS for the exact keypoint pair and formula behind each field.
    """
    chest_flat_cm: float | None = None
    chest_circ_cm: float | None = None
    shoulder_width_cm: float | None = None
    sleeve_length_cm: float | None = None
    garment_length_cm: float | None = None
    hem_width_cm: float | None = None

    waist_flat_cm: float | None = None
    waist_circ_cm: float | None = None
    hip_flat_cm: float | None = None
    inseam_cm: float | None = None
    leg_opening_cm: float | None = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall measurement confidence")

    def to_measurement_zones(self) -> dict[MeasurementZone, float]:
        """Map the subset of fields that correspond to a MeasurementZone, in cm.

        hip_circ is not an exposed field (not requested by spec for bottoms), but the
        body's hip zone is a closed circumference, so it is approximated the same way
        as chest/waist: 2 x hip_flat. This mirrors the explicit chest_circ/waist_circ
        convention and is the only way to compare a bottoms garment against a body twin's
        hip measurement.
        """
        out: dict[MeasurementZone, float] = {}
        if self.chest_circ_cm is not None:
            out[MeasurementZone.chest] = self.chest_circ_cm
        if self.shoulder_width_cm is not None:
            out[MeasurementZone.shoulder_width] = self.shoulder_width_cm
        if self.sleeve_length_cm is not None:
            out[MeasurementZone.sleeve_length] = self.sleeve_length_cm
        if self.garment_length_cm is not None:
            out[MeasurementZone.garment_length] = self.garment_length_cm
        if self.waist_circ_cm is not None:
            out[MeasurementZone.waist] = self.waist_circ_cm
        if self.hip_flat_cm is not None:
            out[MeasurementZone.hip] = self.hip_flat_cm * 2.0
        if self.inseam_cm is not None:
            out[MeasurementZone.inseam] = self.inseam_cm
        return out


class GarmentMeasurement(BaseModel):
    """Firestore document for garments/{garment_id}."""
    id: str
    user_id: str
    category: GarmentCategory
    measurements: GarmentFlatMeasurements
    keypoints: dict[str, KeypointXY]
    keypoints_source: Literal["gemini", "user_corrected"]
    quality: GarmentQualityFlags
    a4: A4DetectionResult
    rectified_image_url: str | None = None
    overlay_image_url: str | None = None
    created_at: str


class FitPrefsDocument(BaseModel):
    """Firestore document for users/{uid}/fit_prefs/{category}.

    Preferred ease per zone, computed as garment_cm - body_twin_cm, for zones
    present on both the garment and the user's body twin.
    """
    user_id: str
    category: GarmentCategory
    garment_id: str
    twin_id: str
    ease_cm: dict[MeasurementZone, float] = Field(default_factory=dict)
    created_at: str


class LabelCheck(str, Enum):
    matches = "matches"
    smaller_than_label = "smaller_than_label"
    larger_than_label = "larger_than_label"


class FitsTwinVerdict(str, Enum):
    fits = "fits"
    tight = "tight"
    loose = "loose"
    unknown = "unknown"


class UnboxingResult(BaseModel):
    """Response for POST /orders/{order_id}/unboxing."""
    order_id: str
    garment_id: str
    product_id: str
    ordered_size_label: str
    tolerance_cm: float
    label_check: LabelCheck
    zone_diffs_cm: dict[MeasurementZone, float] = Field(
        default_factory=dict, description="measured_cm - ordered_label_cm per zone"
    )
    fits_twin: FitsTwinVerdict
