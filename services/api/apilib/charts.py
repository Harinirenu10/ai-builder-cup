"""Size chart studio: measure flat-lay photos, check the grading, publish a chart, and put it on the store."""

import base64
import binascii
import hashlib
import html
import time
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from apilib.auth import RateLimiter, current_uid, err, get_limiter
from apilib.fitquery import check_origin, enforce_rate_limit, merchant_by_key, now_iso
from apilib.returns import desk_merchant
from apilib.stores import Store, get_store
from fitcore.chart.grading import GradingFlag, apply_templates, check_grading, explain_flags, order_labels
from fitcore.chart.review_compare import ReviewNote, compare_reviews
from fitcore.garment.geometry import MeasureError
from fitcore.garment.measure import GarmentMeasurement, measure_garment
from fitcore.schemas import Category, FitType, MeasurementZone as Z, ReviewSignals, SizeChart

router = APIRouter()
MAX_IMAGE_BYTES = 8 * 1024 * 1024
INCH = 2.54


# ----------------------------------------------------------------------------- models


class MeasureIn(BaseModel):
    category: Category
    size_label: str = Field(min_length=1, max_length=16)
    image: str = Field(description="Base64 of one flat-lay photo with an A4 sheet (a data: URL is accepted)")

    @field_validator("image")
    @classmethod
    def _check(cls, v: str) -> str:
        raw = v.split(",", 1)[1] if v.startswith("data:") else v
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image must be base64") from exc
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image is empty or larger than 8 MB")
        return raw


class ZoneOut(BaseModel):
    value_cm: float
    confidence: float


class MeasureOut(BaseModel):
    size_label: str
    category: Category
    zones: dict[Z, ZoneOut]
    flags: list[str]
    unsupported: list[Z]
    overlay_png_base64: str
    seconds: float


class GradingIn(BaseModel):
    category: Category
    sizes: dict[str, dict[Z, float]] = Field(min_length=1)
    explain: bool = Field(default=False, description="Ask the flash model to word each flag (otherwise a template does)")
    product_url: str | None = Field(default=None, description="The product page the reader already read; its reviews and old chart are compared")
    old_chart: SizeChart | None = None
    review_signals: ReviewSignals | None = None


class GradingOut(BaseModel):
    flags: list[GradingFlag]
    review_notes: list[ReviewNote]
    review_source: Literal["none", "product_page", "request"] = "none"


class PublishIn(BaseModel):
    category: Category
    name: str = Field(min_length=1, max_length=120)
    product_ref: str | None = Field(default=None, max_length=64)
    fit_type: FitType | None = None
    sizes: dict[str, dict[Z, float]] = Field(min_length=1)
    source: Literal["measured", "edited"] = "measured"
    confidences: dict[str, dict[Z, float]] | None = None


class ChartOut(BaseModel):
    id: str
    merchant_id: str
    product_ref: str | None
    name: str
    category: Category
    fit_type: FitType | None
    chart_type: Literal["garment"] = "garment"
    sizes: dict[str, dict[Z, float]]
    source: str
    version: int
    status: str
    live: bool
    created_by: str
    created_at: str
    grading_flags: list[GradingFlag]


class ExportOut(BaseModel):
    json_chart: dict[str, Any] = Field(serialization_alias="json")
    html: str
    unit: Literal["cm", "in"]


class StoreChart(BaseModel):
    chart_id: str
    name: str
    version: int
    chart_type: Literal["garment"] = "garment"
    unit: Literal["cm"] = "cm"
    sizes: dict[str, dict[Z, float]]
    published_at: str


# ----------------------------------------------------------------------------- dependencies (tests replace these)


def get_measurer() -> Callable[[bytes, Category], GarmentMeasurement]:
    return lambda image, category: measure_garment(image, category)


def get_flag_explainer() -> Callable[[list[GradingFlag]], list[GradingFlag]]:
    return explain_flags


# ----------------------------------------------------------------------------- measure


@router.post("/biz/charts/measure", response_model=MeasureOut)
def measure(body: MeasureIn, m: dict = Depends(desk_merchant), measurer: Callable = Depends(get_measurer)) -> MeasureOut:
    """Measure one flat-lay photo. The studio calls this once per size, all at the same time."""
    started = time.monotonic()
    try:
        result: GarmentMeasurement = measurer(base64.b64decode(body.image), body.category)
    except MeasureError as exc:
        raise err(422, exc.code, exc.message) from exc
    except HTTPException:
        raise
    except Exception as exc:  # the landmark model or network failed
        raise err(502, "measuring_failed", "The photo could not be measured right now. Please try again.") from exc
    return MeasureOut(size_label=body.size_label, category=body.category,
                      zones={z: ZoneOut(value_cm=v.value_cm, confidence=v.confidence) for z, v in result.zones.items()},
                      flags=result.flags, unsupported=result.unsupported, overlay_png_base64=base64.b64encode(result.overlay_png).decode(),
                      seconds=round(time.monotonic() - started, 2))


# ----------------------------------------------------------------------------- grading check and reviews


def product_page_context(store: Store, url: str) -> tuple[SizeChart | None, ReviewSignals | None]:
    """The old chart and ReviewSignals the reader stored for this URL (products/{sha256(url)})."""
    doc = store.get("products", hashlib.sha256(url.strip().encode("utf-8")).hexdigest())
    resp = (doc or {}).get("response") or {}
    product, reviews = resp.get("product") or {}, resp.get("reviews")
    chart = SizeChart.model_validate(product["size_chart"]) if product.get("size_chart") else None
    return chart, ReviewSignals.model_validate(reviews) if reviews else None


@router.post("/biz/charts/grading-check", response_model=GradingOut)
def grading_check(body: GradingIn, m: dict = Depends(desk_merchant), store: Store = Depends(get_store),
                  explainer: Callable = Depends(get_flag_explainer)) -> GradingOut:
    flags = check_grading(body.category, body.sizes)
    flags = explainer(flags) if body.explain else apply_templates(flags)
    old, reviews, source = body.old_chart, body.review_signals, "request" if (body.old_chart or body.review_signals) else "none"
    if body.product_url and not (old or reviews):
        old, reviews = product_page_context(store, body.product_url)
        source = "product_page" if (old or reviews) else "none"
    return GradingOut(flags=flags, review_notes=compare_reviews(body.sizes, old, reviews), review_source=source)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- publish, list, export


def _own(store: Store, m: dict[str, Any], chart_id: str) -> dict[str, Any]:
    c = store.get("charts", chart_id)
    if not c or c["merchant_id"] != m["id"]:
        raise err(404, "not_found", "No such chart.")
    return c


@router.post("/biz/charts", response_model=ChartOut)
def publish(body: PublishIn, uid: str = Depends(current_uid), m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ChartOut:
    """Save the chart as published (a new version for the product). The grading flags at publish time are kept with it."""
    for label, row in body.sizes.items():
        if not row or any(v <= 0 or v > 400 for v in row.values()):
            raise err(422, "bad_values", f"Size {label} needs measurements between 0 and 400 cm.")
    flags = check_grading(body.category, body.sizes)
    existing = [c for c in store.find("charts", merchant_id=m["id"]) if body.product_ref and c.get("product_ref") == body.product_ref]
    ordered = {k: {z.value: v for z, v in body.sizes[k].items()} for k in order_labels(list(body.sizes))}
    doc = {"merchant_id": m["id"], "product_ref": body.product_ref, "name": body.name, "category": body.category.value,
           "fit_type": body.fit_type.value if body.fit_type else None, "chart_type": "garment", "sizes": ordered, "source": body.source,
           "version": 1 + max([c["version"] for c in existing], default=0), "status": "published", "live": False, "created_by": uid,
           "created_at": now_iso(), "grading_flags": [f.model_dump(mode="json") for f in flags],
           "confidences": {k: {z.value: v for z, v in row.items()} for k, row in (body.confidences or {}).items()}}
    cid = store.add("charts", doc)
    return ChartOut.model_validate({**doc, "id": cid})


@router.get("/biz/charts", response_model=list[ChartOut])
def list_charts(m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> list[ChartOut]:
    rows = sorted(store.find("charts", merchant_id=m["id"]), key=lambda c: c["created_at"], reverse=True)
    return [ChartOut.model_validate(c) for c in rows]


@router.get("/biz/charts/{chart_id}", response_model=ChartOut)
def get_chart(chart_id: str, m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ChartOut:
    return ChartOut.model_validate(_own(store, m, chart_id))


ZONE_TITLES = {Z.chest: "Chest", Z.waist: "Waist", Z.hip: "Hip", Z.shoulder_width: "Shoulder", Z.sleeve_length: "Sleeve length", Z.garment_length: "Length",
               Z.thigh: "Thigh", Z.inseam: "Inseam", Z.neck: "Neck", Z.back_rise: "Back rise", Z.upper_back: "Upper back"}
ZONE_ORDER = [Z.chest, Z.waist, Z.hip, Z.shoulder_width, Z.sleeve_length, Z.garment_length, Z.thigh, Z.inseam, Z.neck, Z.back_rise, Z.upper_back]


def convert(sizes: dict[str, dict[str, float]], unit: str) -> dict[str, dict[str, float]]:
    k = 1.0 if unit == "cm" else 1.0 / INCH
    return {s: {z: round(v * k, 1) for z, v in row.items()} for s, row in sizes.items()}


def html_table(name: str, sizes: dict[str, dict[str, float]], unit: str) -> str:
    zones = [z for z in ZONE_ORDER if any(z.value in row for row in sizes.values())]
    head = "".join(f"<th>{html.escape(ZONE_TITLES[z])} ({unit})</th>" for z in zones)
    rows = "".join("<tr><th scope=\"row\">" + html.escape(s) + "</th>" + "".join(f"<td>{sizes[s][z.value]:g}</td>" if z.value in sizes[s] else "<td></td>" for z in zones) + "</tr>"
                   for s in sizes)
    return (f'<table class="size-chart"><caption>{html.escape(name)}: garment measurements, laid flat, in {unit}</caption>'
            f'<thead><tr><th scope="col">Size</th>{head}</tr></thead><tbody>{rows}</tbody></table>')


@router.get("/biz/charts/{chart_id}/export", response_model=ExportOut, response_model_by_alias=True)
def export(chart_id: str, unit: Literal["cm", "in"] = "cm", m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ExportOut:
    c = _own(store, m, chart_id)
    sizes = convert(c["sizes"], unit)
    return ExportOut(json_chart={"name": c["name"], "product_ref": c["product_ref"], "category": c["category"], "chart_type": "garment", "unit": unit,
                                 "version": c["version"], "sizes": sizes}, html=html_table(c["name"], sizes, unit), unit=unit)


# ----------------------------------------------------------------------------- put it on the store


def _set_live(store: Store, m: dict[str, Any], c: dict[str, Any], live: bool) -> dict[str, Any]:
    if not c.get("product_ref"):
        raise err(422, "no_product_ref", "Give the chart a product reference (for example dtc-shirt) before putting it on the store.")
    if live:
        for other in store.find("charts", merchant_id=m["id"]):
            if other.get("product_ref") == c["product_ref"] and other["id"] != c["id"] and other.get("live"):
                store.update("charts", other["id"], {"live": False})
    store.update("charts", c["id"], {"live": live, "live_at": now_iso() if live else None})
    return store.get("charts", c["id"])  # type: ignore[return-value]


@router.post("/biz/charts/{chart_id}/use-on-store", response_model=ChartOut)
def use_on_store(chart_id: str, m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ChartOut:
    """Make this the chart the mock store shows (and the size widget uses) for the product. Any other live chart for it is switched off."""
    return ChartOut.model_validate(_set_live(store, m, _own(store, m, chart_id), True))


@router.post("/biz/charts/{chart_id}/remove-from-store", response_model=ChartOut)
def remove_from_store(chart_id: str, m: dict = Depends(desk_merchant), store: Store = Depends(get_store)) -> ChartOut:
    return ChartOut.model_validate(_set_live(store, m, _own(store, m, chart_id), False))


@router.get("/store/charts/{product_ref}", response_model=StoreChart, responses={204: {"description": "No chart has been put on the store for this product"}})
def store_chart(product_ref: str, x_merchant_key: str | None = Header(default=None), origin: str | None = Header(default=None),
                store: Store = Depends(get_store), limiter: RateLimiter = Depends(get_limiter)) -> StoreChart | Response:
    """What a store page asks for: the live chart for a product, or 204 (empty, not an error) when the seller has not put one on the store."""
    m = merchant_by_key(store, x_merchant_key)
    check_origin(m, origin, required=False)
    enforce_rate_limit(m, limiter)
    live = [c for c in store.find("charts", merchant_id=m["id"]) if c.get("product_ref") == product_ref and c.get("live")]
    if not live:
        return Response(status_code=204)
    c = max(live, key=lambda c: c["version"])
    return StoreChart(chart_id=c["id"], name=c["name"], version=c["version"], sizes=c["sizes"], published_at=c.get("live_at") or c["created_at"])
