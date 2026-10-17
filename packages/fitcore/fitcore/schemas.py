"""Domain schemas for measured fit."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class MeasurementZone(str, Enum):
    """Supported anatomic and garment measurement zones."""
    chest = "chest"
    waist = "waist"
    hip = "hip"
    shoulder_width = "shoulder_width"
    sleeve_length = "sleeve_length"
    torso_length = "torso_length"
    garment_length = "garment_length"
    inseam = "inseam"
    thigh = "thigh"
    neck = "neck"
    back_rise = "back_rise"
    upper_back = "upper_back"


ZONE_CALCULATION_METHODS: dict[MeasurementZone, dict[str, str]] = {
    MeasurementZone.chest: {
        "type": "circumference",
        "method": "Closed horizontal planar contour slice at maximum thoracic girth (nipple line / lower axillary hollow).",
    },
    MeasurementZone.waist: {
        "type": "circumference",
        "method": "Closed horizontal planar contour slice at minimum torso perimeter between lower ribs and iliac crest.",
    },
    MeasurementZone.hip: {
        "type": "circumference",
        "method": "Closed horizontal planar contour slice at maximum gluteal / greater trochanter perimeter.",
    },
    MeasurementZone.shoulder_width: {
        "type": "linear",
        "method": "3D Euclidean surface distance across back between left and right acromion / deltoid apex vertices.",
    },
    MeasurementZone.sleeve_length: {
        "type": "linear",
        "method": "Contour path from acromion along lateral arm, over flexed elbow to distal styloid process of radius (wrist).",
    },
    MeasurementZone.torso_length: {
        "type": "linear",
        "method": "Vertical surface distance along dorsal midline from C7 cervical vertebra to horizontal waist plane.",
    },
    MeasurementZone.garment_length: {
        "type": "linear",
        "method": "Vertical reference distance along posterior midline from cervical vertebra base (collar) to hemline plane.",
    },
    MeasurementZone.inseam: {
        "type": "linear",
        "method": "Surface geodesic path along medial leg from perineal / crotch junction down to lower border of medial malleolus (inner ankle).",
    },
    MeasurementZone.thigh: {
        "type": "circumference",
        "method": "Closed cross-sectional slice orthogonal to thigh axis 5 cm below gluteal fold at maximal upper leg perimeter.",
    },
    MeasurementZone.neck: {
        "type": "circumference",
        "method": "Closed cross-sectional slice around base of neck directly superior to suprasternal notch and C7 vertebra.",
    },
    MeasurementZone.back_rise: {
        "type": "linear",
        "method": "Midline surface distance along sacrum from posterior waist center point down to crotch intersection.",
    },
    MeasurementZone.upper_back: {
        "type": "linear",
        "method": "Transverse horizontal distance across upper back between posterior armscye (armpit fold) crease landmarks.",
    },
}


class Measurement(BaseModel):
    """Single zone measurement in metric units (centimeters)."""
    value_cm: float = Field(..., gt=0, description="Measurement value in centimeters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    source: str = Field(..., description="Source of the measurement, e.g. '3d_twin', 'manual_tape', 'flat_lay'")


class Measurements(BaseModel):
    """Collection of measurements indexed by measurement zone."""
    zones: dict[MeasurementZone, Measurement] = Field(
        default_factory=dict,
        description="Zone-to-measurement mapping in centimeters"
    )

    def get_cm(self, zone: MeasurementZone) -> float | None:
        """Helper to get measurement value in cm if present."""
        m = self.zones.get(zone)
        return m.value_cm if m else None


class ChartType(str, Enum):
    body = "body"
    garment = "garment"


class SizeChart(BaseModel):
    """Brand or merchant size chart mapping size labels to zone dimensions in cm."""
    sizes: dict[str, dict[MeasurementZone, float]] = Field(
        ...,
        description="Mapping from size label (e.g. 'M', '32') to zone-value_cm pairs"
    )
    chart_type: Literal["body", "garment"] = Field(
        default="garment",
        description="Whether values represent garment dimensions or intended body dimensions"
    )
    size_system: str = Field(
        default="ALPHA",
        description="Size system designation, e.g. 'ALPHA', 'US_NUMERIC', 'EU'"
    )


class Product(BaseModel):
    """Ecommerce garment product metadata and size specification."""
    id: str = Field(..., description="Unique product identifier")
    brand: str = Field(..., description="Brand name")
    title: str = Field(..., description="Product title")
    size_chart: SizeChart | None = Field(default=None, description="Extracted or merchant-provided size chart")
    material_composition: dict[str, float] = Field(
        default_factory=dict,
        description="Fabric breakdown, e.g. {'cotton': 0.98, 'elastane': 0.02}"
    )
    description: str | None = Field(default=None, description="Product description text")


class FitResult(BaseModel):
    """Fit recommendation and verdict placeholder."""
    recommended_size: str = Field(..., description="Recommended size label, e.g. 'M'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence score")
    verdict: str = Field(..., description="Objective description of fit focused on garment ease and dimensions")
    zone_deltas: dict[MeasurementZone, float] = Field(
        default_factory=dict,
        description="Ease delta in cm (garment_cm - body_cm) per zone"
    )
    details: str | None = Field(default=None, description="Additional fit notes describing garment cut")


# --- Phase 1: 3D Twin & Body Measurement Pipeline Domain Models ---


class JobStatus(str, Enum):
    """Lifecycle status states for body twin generation jobs."""
    checking = "checking"
    measuring = "measuring"
    fitting = "fitting"
    refining = "refining"
    done = "done"
    needs_retake = "needs_retake"
    rejected = "rejected"
    failed = "failed"


class BodyTwinJobRequest(BaseModel):
    """Request payload for 3D body twin reconstruction from 4 orthogonal photos."""
    user_id: str = Field(..., description="Unique user identifier owning this twin")
    height_cm: float = Field(..., gt=50.0, lt=260.0, description="Confirmed subject height in centimeters")
    gender: Literal["neutral", "female", "male"] = Field(default="neutral", description="SMPL model base template")
    age_18_confirmed: bool = Field(default=False, description="User confirmation of 18+ age requirement")
    opt_in_keep_front_photo: bool = Field(default=False, description="Optional opt-in consent to retain front photo only")
    photo_front: str = Field(..., description="Front orthogonal photo (data URL, storage URI, or base64)")
    photo_back: str = Field(..., description="Back orthogonal photo")
    photo_left: str = Field(..., description="Left profile orthogonal photo")
    photo_right: str = Field(..., description="Right profile orthogonal photo")


class CriticView(str, Enum):
    """Viewpoint for critic feedback."""
    front = "front"
    back = "back"
    left = "left"
    right = "right"


class CriticRegion(str, Enum):
    """Supported anatomical silhouette regions for critique."""
    shoulders = "shoulders"
    chest = "chest"
    waist = "waist"
    hips = "hips"
    thighs = "thighs"
    arms = "arms"
    legs = "legs"
    torso_length = "torso_length"
    leg_length = "leg_length"


class CriticIssue(str, Enum):
    """Allowed geometry issue types."""
    too_wide = "too_wide"
    too_narrow = "too_narrow"
    too_long = "too_long"
    too_short = "too_short"
    ok = "ok"


class CriticCause(str, Enum):
    """Attributed causes for visual discrepancy."""
    model_error = "model_error"
    loose_clothing = "loose_clothing"
    pose = "pose"
    occlusion = "occlusion"
    hair = "hair"


class CriticRegionFeedback(BaseModel):
    """Structured feedback on a specific body region from a specific view."""
    view: CriticView = Field(..., description="Orthogonal photo viewpoint")
    region: CriticRegion = Field(..., description="Anatomical silhouette region")
    issue: CriticIssue = Field(..., description="Identified geometry issue or 'ok'")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score from 0.0 to 1.0")
    cause: CriticCause = Field(..., description="Attributed cause of the silhouette discrepancy")
    trust: float = Field(..., ge=0.0, le=1.0, description="Critic trust/confidence in this assessment (0.0 to 1.0)")


class CriticReport(BaseModel):
    """Multimodal critique report returned by Gemini in the refinement loop."""
    regions: list[CriticRegionFeedback] = Field(
        default_factory=list,
        description="Region-by-region silhouette critique"
    )
    stop: bool = Field(
        default=False,
        description="Whether the critic recommends stopping refinement (fit is satisfactory or cannot improve)"
    )
    summary: str = Field(
        ...,
        description="Neutral, objective, shopper-readable summary describing the fit and silhouette"
    )


class CriticRoundResult(BaseModel):
    """Structured verdict produced by Gemini Flash during multimodal critic review."""
    round_number: int = Field(..., ge=1, le=5, description="Refinement round index (1 to 5)")
    suggested_adjustments: dict[MeasurementZone, float] = Field(
        default_factory=dict,
        description="Suggested zone circumference/length adjustments in centimeters"
    )
    suggested_beta_delta: list[float] = Field(
        default_factory=list,
        description="Suggested directional updates to the first 10 SMPL shape parameters"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in silhouette critique")
    notes: str = Field(
        ...,
        description="Objective feedback describing garment and silhouette geometry without body commentary"
    )


class RoundLogDocument(BaseModel):
    """Document representing a single refinement round stored in jobs/{job_id}/rounds/{round_num}."""
    round: int = Field(..., ge=1, le=5)
    mean_iou_before: float = Field(..., ge=0.0, le=1.0)
    mean_iou_after: float = Field(..., ge=0.0, le=1.0)
    iou_delta: float = Field(..., description="Change in mean silhouette IoU (after - before)")
    accepted: bool = Field(..., description="True if iou_delta >= +0.002, False if rejected and rolled back")
    step_size: float = Field(default=1.0, description="Current gradient/optimization step size (halved on rollback)")
    zone_adjustments: dict[MeasurementZone, float] = Field(default_factory=dict)
    per_view_iou: dict[str, float] = Field(default_factory=dict, description="Per-view IoU metrics")
    overlay_urls: dict[str, str] = Field(default_factory=dict, description="Storage URLs to generated overlays")
    critic_notes: str = Field(default="")
    critic_latency_ms: int = Field(default=0)
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class JobDocument(BaseModel):
    """Firestore document representing twin fitting job state at jobs/{job_id}."""
    id: str = Field(..., description="Unique job identifier")
    user_id: str = Field(..., description="User ID owner")
    status: JobStatus = Field(default=JobStatus.checking)
    current_step: str | None = Field(default=None)
    error_code: str | None = Field(default=None, description="e.g. BLUR_DETECTED, AGE_CONSENT_REQUIRED")
    error_message: str | None = Field(default=None)
    height_cm: float = Field(..., gt=50.0, lt=260.0)
    gender: Literal["neutral", "female", "male"] = Field(default="neutral")
    opt_in_keep_front_photo: bool = Field(default=False)
    created_at: str = Field(...)
    updated_at: str = Field(...)
    twin_id: str | None = Field(default=None)


class TwinDocument(BaseModel):
    """Firestore document representing a fitted 3D avatar at twins/{twin_id}."""
    id: str = Field(..., description="Unique twin identifier")
    user_id: str = Field(..., description="User ID owner")
    gender: Literal["neutral", "female", "male"]
    height_cm: float = Field(..., gt=50.0, lt=260.0)
    betas: list[float] = Field(default_factory=list, description="Optimized 10 SMPL shape parameters")
    measurements: Measurements = Field(..., description="All zone measurements in cm with confidence")
    glb_storage_url: str | None = Field(default=None, description="Cloud Storage URI for rendered .glb avatar")
    final_mean_iou: float | None = Field(default=None, ge=0.0, le=1.0, description="Final silhouette IoU achieved across 4 views")
    created_at: str = Field(...)

