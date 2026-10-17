"""Request and response models for the business-facing endpoints.

FitQueryResponse is the whole privacy contract of the fit query: a size, a confidence, per-zone verdict words,
an optional garment-side hint and a guarantee flag. It has no field for any body measurement or shape parameter,
and tests/test_fit_query_privacy.py fails if one is ever added.
"""

from typing import Literal

from pydantic import BaseModel, Field

from fitcore.schemas import Category, FabricPart, FitType, MeasurementZone, SizeChart, StretchClass, Verdict


class FitQueryProduct(BaseModel):
    """What the merchant knows about a garment. Charts may be in cm or inches (set size_chart.unit)."""
    size_chart: SizeChart
    category: Category
    fit_type: FitType = FitType.regular
    fabric: list[FabricPart] = Field(default_factory=list)
    stretch_class: StretchClass = StretchClass.none
    pre_shrunk: bool | None = None
    ref: str | None = Field(default=None, max_length=64, description="Merchant's own product id, logged instead of a hash if given")


class FitQueryRequest(BaseModel):
    product: FitQueryProduct
    consent_token: str = Field(min_length=20)


class ZoneVerdict(BaseModel):
    zone: MeasurementZone
    verdict: Verdict


class FitQueryResponse(BaseModel):
    recommended_size: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    zone_verdicts: list[ZoneVerdict]
    alteration_hint: str | None
    guarantee_eligible: bool


class ConsentRequest(BaseModel):
    merchant: str = Field(description="The merchant's publishable key")
    origin: str = Field(description="Origin of the store page that opened the popup")


class ConsentResponse(BaseModel):
    consent_id: str
    token: str
    expires_at: str
    merchant_name: str
    origin: str


class ConsentSummary(BaseModel):
    consent_id: str
    merchant_name: str
    scope: str
    created_at: str
    revoked_at: str | None


class MerchantPublic(BaseModel):
    name: str
    origin_allowed: bool


class GiftCreate(BaseModel):
    product: FitQueryProduct
    title: str = Field(default="a gift", max_length=120, description="What the gift is, shown to the recipient")
    note: str | None = Field(default=None, max_length=300)


class GiftCreated(BaseModel):
    id: str
    recipient_path: str
    expires_at: str


class GiftGiverView(BaseModel):
    """What the giver can ever see: the size and how sure we are. Nothing else about the recipient."""
    id: str
    status: Literal["pending", "approved", "declined", "expired"]
    recommended_size: str | None = None
    confidence: float | None = None


class GiftRecipientView(BaseModel):
    id: str
    status: Literal["pending", "approved", "declined", "expired"]
    title: str
    note: str | None
    category: Category
    can_approve: bool
    shares: str = "The giver will see only the recommended size and how confident we are. No measurements, photos or body details."
