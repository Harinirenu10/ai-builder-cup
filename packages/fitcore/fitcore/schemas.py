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
    """Fit recommendation and verdict placeholder (to be extended in later prompts)."""
    recommended_size: str = Field(..., description="Recommended size label, e.g. 'M'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence score")
    verdict: str = Field(..., description="Objective description of fit focused on garment ease and dimensions")
    zone_deltas: dict[MeasurementZone, float] = Field(
        default_factory=dict,
        description="Ease delta in cm (garment_cm - body_cm) per zone"
    )
    details: str | None = Field(default=None, description="Additional fit notes describing garment cut")
