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
