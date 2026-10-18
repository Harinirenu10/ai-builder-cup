"""Domain schemas for measured fit. All lengths are centimeters internally."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

from fitcore.units import inches_to_cm


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


class Category(str, Enum):
    tshirt = "tshirt"
    shirt = "shirt"
    kurta = "kurta"
    dress = "dress"
    blouse = "blouse"
    trousers = "trousers"
    jeans = "jeans"
    jacket = "jacket"


class FitType(str, Enum):
    slim = "slim"
    regular = "regular"
    relaxed = "relaxed"
    oversized = "oversized"


class StretchClass(str, Enum):
    none = "none"
    low = "low"
    high = "high"


class SizeChart(BaseModel):
    """Brand or merchant size chart mapping size labels to zone dimensions.

    Sizes should be listed smallest to largest; ties in the fit engine go to the earlier size.
    Values given in inches are converted to centimeters on validation.
    """
    sizes: dict[str, dict[MeasurementZone, float]] = Field(
        ...,
        description="Mapping from size label (e.g. 'M', '32') to zone-value pairs"
    )
    chart_type: Literal["body", "garment"] = Field(
        default="garment",
        description="Whether values represent garment dimensions or intended body dimensions"
    )
    size_system: str = Field(
        default="ALPHA",
        description="Size system designation, e.g. 'ALPHA', 'US_NUMERIC', 'EU'"
    )
    unit: Literal["cm", "in"] = Field(
        default="cm",
        description="Unit of the supplied values; normalised to 'cm' after validation"
    )
    original_unit: Literal["cm", "in"] | None = Field(
        default=None, description="Unit the source page used, kept for display; values are already cm"
    )
    type_evidence: str | None = Field(default=None, description="Label text that decided body vs garment, e.g. 'To fit chest'")
    flags: list[str] = Field(default_factory=list, description="Quality flags, e.g. 'non_monotonic:chest'")

    @model_validator(mode="after")
    def _normalise_to_cm(self) -> "SizeChart":
        if self.unit == "in":
            self.original_unit = self.original_unit or "in"
            self.sizes = {
                size: {zone: inches_to_cm(v, precision=4) for zone, v in row.items()}
                for size, row in self.sizes.items()
            }
            self.unit = "cm"
        return self


class FabricPart(BaseModel):
    """One fiber and its share of the fabric."""
    fiber: str = Field(..., description="Fiber name, e.g. 'cotton', 'rayon', 'elastane'")
    percent: float = Field(..., gt=0, le=100, description="Share of the fabric in percent")


class Product(BaseModel):
    """Ecommerce garment product metadata and size specification."""
    id: str = Field(..., description="Unique product identifier")
    brand: str = Field(..., description="Brand name")
    title: str = Field(..., description="Product title")
    category: Category = Field(..., description="Garment category")
    fit_type: FitType = Field(default=FitType.regular, description="Intended cut")
    fabric: list[FabricPart] = Field(default_factory=list, description="Fiber breakdown with percentages")
    stretch_class: StretchClass = Field(default=StretchClass.none, description="Fabric stretch")
    pre_shrunk: bool | None = Field(default=None, description="True, False, or None when unknown")
    size_chart: SizeChart | None = Field(default=None, description="Extracted or merchant-provided size chart")
    material_composition: dict[str, float] = Field(
        default_factory=dict,
        description="Legacy fabric breakdown as fractions, e.g. {'cotton': 0.98, 'elastane': 0.02}; used when fabric is empty"
    )
    description: str | None = Field(default=None, description="Product description text")
    pre_shrunk_evidence: str | None = Field(default=None, description="Short page snippet behind pre_shrunk")
    source_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fabric_from_legacy(self) -> "Product":
        if not self.fabric and self.material_composition:
            total = sum(self.material_composition.values())
            scale = 100.0 if total <= 1.5 else 1.0
            self.fabric = [
                FabricPart(fiber=k, percent=v * scale)
                for k, v in self.material_composition.items()
                if v > 0
            ]
        return self


# --------------------------------------------------------------------------- optional fit inputs


class FitPreferences(BaseModel):
    """Personal ease offsets (cm) per category and zone, learned from the shopper's best garments.

    The offset is added to the computed ease: a positive offset makes a garment read roomier.
    """
    offsets_cm: dict[Category, dict[MeasurementZone, float]] = Field(default_factory=dict)


class ReviewAdjustments(BaseModel):
    """Per-zone cm corrections derived from reviews.

    Positive means the garment runs smaller than its chart (ease is reduced by this amount).
    """
    cm: dict[MeasurementZone, float] = Field(default_factory=dict)


class ReviewSignal(BaseModel):
    """A zone-level signal read from product reviews."""
    zone: MeasurementZone
    direction: Literal["smaller", "larger"] = Field(
        ..., description="'smaller': the garment runs smaller or shorter than the chart; 'larger': the opposite"
    )
    strength: Literal["slight", "moderate", "strong"] = "moderate"
    context: Literal["long_torso", "short_torso"] | None = Field(
        default=None, description="Only applies to shoppers matching this context"
    )


class PoseDelta(BaseModel):
    """Body circumference increase (cm) per zone in one pose, and time spent in that pose."""
    minutes: float = Field(default=0.0, ge=0)
    delta_cm: dict[MeasurementZone, float] = Field(default_factory=dict)


class PoseDeltas(BaseModel):
    poses: dict[str, PoseDelta] = Field(default_factory=dict, description="e.g. {'seated': PoseDelta(...)}")


# --------------------------------------------------------------------------- results


class Verdict(str, Enum):
    tight = "tight"
    ideal = "ideal"
    loose = "loose"
    short = "short"
    long = "long"


class PoseFit(BaseModel):
    ease_cm: float
    strain: bool
    minutes: float


class ZoneFit(BaseModel):
    zone: MeasurementZone
    hard: bool = Field(description="True when the zone is hard to alter")
    path: Literal["garment", "body"] = Field(description="Which size-chart path produced the ease")
    ease_now_cm: float
    ease_after_wash_cm: float
    shrink_cm: float = 0.0
    target_min_cm: float
    target_ideal_cm: float
    target_max_cm: float
    verdict: Verdict
    margin_cm: float = Field(description="Distance outside the acceptable range now; negative below, positive above, 0 inside")
    verdict_after_wash: Verdict
    margin_after_wash_cm: float
    poses: dict[str, PoseFit] = Field(default_factory=dict)
    penalty: float = 0.0


class SizeFit(BaseModel):
    size: str
    zones: dict[MeasurementZone, ZoneFit]
    penalty: float
    score: float = Field(description="0-100, higher is a better match")


class Alteration(BaseModel):
    zone: MeasurementZone
    action: str
    cm: float
    text: str


class FitWarning(BaseModel):
    kind: Literal["wash", "pose", "size", "data"]
    message: str
    zone: MeasurementZone | None = None


class FitResult(BaseModel):
    """Fit recommendation for one product and one body."""
    category: Category
    fit_type: FitType
    sizes: list[SizeFit]
    recommended_size: str | None = Field(
        default=None, description="Best size, or None when a hard zone fails in every size"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alterations: list[Alteration] = Field(default_factory=list)
    warnings: list[FitWarning] = Field(default_factory=list)
    explanation_facts: list[str] = Field(default_factory=list)

    def size_fit(self, size: str) -> SizeFit | None:
        return next((s for s in self.sizes if s.size == size), None)


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


class OrderDocument(BaseModel):
    """Firestore document representing a placed order at orders/{order_id}.

    Minimal shape: enough to look up the ordered size specification on the
    Product record for unboxing comparison. Extend as order flows are built out.
    """
    id: str = Field(..., description="Unique order identifier")
    user_id: str = Field(..., description="User ID owner of the order")
    product_id: str = Field(..., description="Ordered Product id")
    size_label: str = Field(..., description="Ordered size label, e.g. 'M', must match a Product.size_chart key")
    status: str = Field(default="placed", description="Order lifecycle status")
    created_at: str = Field(...)


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
    saved: bool = Field(default=False, description="Whether the owner has explicitly confirmed/kept this twin via the Save action")
    saved_at: str | None = Field(default=None, description="ISO timestamp of the explicit save action, or None if never saved")


class TailorLines(BaseModel):
    """Text lines of a tailor card in one language."""
    title: str
    size_line: str
    alteration_lines: list[str]
    measurement_lines: list[str]


class TailorCard(BaseModel):
    size: str
    alterations: list[Alteration]
    measurements: dict[MeasurementZone, float] = Field(description="Only what a tailor needs, in cm")
    lines: dict[str, TailorLines] = Field(description="Language code -> text lines")
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- business view (verdicts only)


class BizZoneVerdict(BaseModel):
    zone: MeasurementZone
    verdict: Verdict
    verdict_after_wash: Verdict


class BizAlteration(BaseModel):
    zone: MeasurementZone
    action: str


class BizVerdict(BaseModel):
    """What a business may see: the size verdict and per-zone verdicts. No measurements, ease or cm values."""
    category: Category
    recommended_size: str | None = None
    confidence: float = 0.0
    zones: list[BizZoneVerdict] = Field(default_factory=list)
    alterations: list[BizAlteration] = Field(default_factory=list)
    warning_kinds: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- reviews read from a product page


class ReviewerContext(BaseModel):
    """What a reviewer said about themselves, when they said it."""
    height_cm: float | None = None
    torso: Literal["long", "short"] | None = None
    note: str | None = Field(default=None, description="Short note, e.g. 'bought M, 5ft 10'")


class ZoneSignal(BaseModel):
    zone: MeasurementZone
    direction: Literal["runs_small", "runs_large", "short", "long", "tight", "loose"]
    strength: float = Field(ge=0.0, le=1.0)
    mention_count: int = Field(ge=1)
    reviewer_context: list[ReviewerContext] = Field(default_factory=list)


class ReviewSignals(BaseModel):
    """Signals from review text only. Counts come from reviews actually read; empty when there were none."""
    reviews_read: int = Field(default=0, ge=0)
    zone_signals: list[ZoneSignal] = Field(default_factory=list)
    shrinkage_mentions: int = Field(default=0, ge=0)
    color_mismatch_mentions: int = Field(default=0, ge=0)
    sheerness_mentions: int = Field(default=0, ge=0)


# --------------------------------------------------------------------------- Prompt 7: Wardrobe & Planner Schemas

Circle = Literal["family", "office", "college", "friends", "other"]


class CircleProposal(BaseModel):
    circle: Circle = "other"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = Field(default="")


class BoundingBox(BaseModel):
    ymin: int = Field(ge=0, le=1000)
    xmin: int = Field(ge=0, le=1000)
    ymax: int = Field(ge=0, le=1000)
    xmax: int = Field(ge=0, le=1000)


class DetectedGarment(BaseModel):
    box: list[int] = Field(
        ...,
        description="Bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000",
    )
    category: Category = Field(default=Category.shirt)
    color: str = Field(default="unknown")
    pattern: str = Field(default="solid")
    formality: int = Field(default=3, ge=1, le=5)
    occasions: list[str] = Field(default_factory=list)
    season: str = Field(default="all")
    ethnic_or_western: str = Field(default="western")


class IngestDetections(BaseModel):
    items: list[DetectedGarment] = Field(default_factory=list)


class WardrobeItem(BaseModel):
    id: str
    user_id: str
    category: Category
    color: str
    pattern: str
    formality: int = Field(ge=1, le=5)
    occasions: list[str] = Field(default_factory=list)
    season: str = "all"
    ethnic_or_western: str = "western"
    image_url: str | None = None
    phash: str | None = None
    created_at: str


class CalendarEvent(BaseModel):
    id: str
    title: str
    start_time: str
    end_time: str | None = None
    location: str | None = None
    attendees: list[str] = Field(default_factory=list)
    circle: Circle = "other"
    description: str | None = None


class WearLogEntry(BaseModel):
    id: str
    user_id: str
    item_id: str
    event_id: str | None = None
    circle: Circle
    date: str
    logged_at: str


class RepeatCheck(BaseModel):
    circle: Circle
    last_worn_with_circle_on: str | None = None
    chip_text: str | None = None


class EventPlan(BaseModel):
    event_id: str
    title: str
    dress_code: str
    outfit_item_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    pack_list: list[str] = Field(default_factory=list)
    repeat_check: RepeatCheck


class DayPlan(BaseModel):
    date: str
    events: list[EventPlan] = Field(default_factory=list)
    cached: bool = False
