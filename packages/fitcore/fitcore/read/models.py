"""Lenient models for what the LLM returns. Python (normalize.py) turns them into Product and ReviewSignals."""

from typing import Literal

from pydantic import BaseModel, Field

from fitcore.schemas import Category, FitType, MeasurementZone, StretchClass


class RawFabric(BaseModel):
    fiber: str
    percent: float


class RawMeasure(BaseModel):
    label: str = Field(description="Measurement name exactly as printed, e.g. 'To fit chest'")
    value_min: float
    value_max: float | None = Field(default=None, description="Upper end when the cell is a range like 38-40")


class RawSize(BaseModel):
    label: str
    measurements: list[RawMeasure] = Field(default_factory=list)


class ExtractedProduct(BaseModel):
    title: str
    brand: str | None = None
    category: Category | None = Field(default=None, description="null when the item is not one of the supported garments")
    fit_type: FitType | None = None
    fabric: list[RawFabric] = Field(default_factory=list)
    stretch_class: StretchClass | None = None
    pre_shrunk: Literal["true", "false", "unknown"] = "unknown"
    pre_shrunk_evidence: str | None = None
    size_system: Literal["IN", "UK", "US", "EU", "JP", "KR", "CN", "letter"] | None = None
    chart_type: Literal["body", "garment", "unknown"] = "unknown"
    chart_type_evidence: str | None = Field(default=None, description="The label or sentence that decided body vs garment")
    unit: Literal["cm", "in", "unknown"] = "unknown"
    circumference_basis: Literal["full", "flat", "unknown"] = Field(
        default="unknown", description="'flat' when the garment chest/waist/hip are measured flat (half the circumference)"
    )
    sizes: list[RawSize] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)


class RawReviewerContext(BaseModel):
    height_cm: float | None = None
    torso: Literal["long", "short"] | None = None
    note: str | None = None


class RawZoneSignal(BaseModel):
    zone: MeasurementZone
    direction: Literal["runs_small", "runs_large", "short", "long", "tight", "loose"]
    strength: float = 0.5
    mention_count: int = 1
    reviewer_context: list[RawReviewerContext] = Field(default_factory=list)


class ExtractedReviews(BaseModel):
    reviews_read: int = Field(default=0, description="How many separate reviews appear in the text provided")
    zone_signals: list[RawZoneSignal] = Field(default_factory=list)
    shrinkage_mentions: int = 0
    color_mismatch_mentions: int = 0
    sheerness_mentions: int = 0


class PageExtraction(BaseModel):
    product: ExtractedProduct
    reviews: ExtractedReviews = Field(default_factory=ExtractedReviews)
