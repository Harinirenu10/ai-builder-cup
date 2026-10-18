"""Garment flat-lay photo measurement endpoints: measure, fit-prefs, unboxing.

Follows the same no-header-auth convention as services/api/main.py's twin
endpoints (user_id supplied directly in the request), since no auth layer
exists yet in this repo. Ownership is enforced by comparing the stored
document's user_id against the caller-supplied user_id.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from apilib.stores import Store, get_store
from fitcore.fit import evaluate as fit_engine_evaluate
from fitcore.garment_photo.pipeline import A4NotFoundError, measure_garment_photo
from fitcore.garment_photo.schemas import (
    FitPrefsDocument,
    FitsTwinVerdict,
    GarmentCategory,
    GarmentFlatMeasurements,
    GarmentMeasurement,
    KeypointXY,
    LabelCheck,
    UnboxingResult,
)
from fitcore.schemas import (
    Category,
    FitType,
    Measurements,
    MeasurementZone,
    Product,
    SizeChart,
    Verdict,
)
from services.geometry.cloud_client import get_firestore_client, get_storage_client

logger = logging.getLogger("api.garment_routes")

router = APIRouter()

DEFAULT_TOLERANCE_CM = 1.5

# fits_twin is evaluated by the real fitcore.fit engine (packages/fitcore/fitcore/fit/engine.py),
# not a local heuristic. The one measured garment is wrapped as a single-size synthetic
# Product.size_chart (chart_type="garment", matching evaluate()'s "garment - body" ease formula)
# and compared against the twin via fit_engine_evaluate(); see _fits_twin_verdict below.
_SYNTHETIC_SIZE_LABEL = "measured"

# GarmentCategory (this module's photo-keypoint vocabulary: tops/kurtas/bottoms) has no 1:1
# mapping to fitcore.fit's Category (its per-garment-type zone/ease config keys). Chosen as the
# closest match whose zones_by_category (fitcore/config/zones.yaml) overlaps what this pipeline
# actually measures for that silhouette: tops -> shirt (chest, waist, shoulder_width,
# sleeve_length, garment_length), kurtas -> kurta (exact name match), bottoms -> trousers
# (waist, hip, thigh, inseam, back_rise -- jeans has an identical zone set, trousers is the more
# generic default). Revisit if GarmentCategory ever gains finer-grained categories.
GARMENT_CATEGORY_TO_FIT_CATEGORY: dict[GarmentCategory, Category] = {
    GarmentCategory.tops: Category.shirt,
    GarmentCategory.kurtas: Category.kurta,
    GarmentCategory.bottoms: Category.trousers,
}

# fitcore.fit.Verdict has 5 values (tight/ideal/loose for circumferences, short/ideal/long for
# lengths); FitsTwinVerdict only distinguishes fits/tight/loose. short behaves like tight (not
# enough room) and long like loose (too much room) for this coarser signal.
FIT_VERDICT_TO_FITS_TWIN: dict[Verdict, FitsTwinVerdict] = {
    Verdict.tight: FitsTwinVerdict.tight,
    Verdict.short: FitsTwinVerdict.tight,
    Verdict.ideal: FitsTwinVerdict.fits,
    Verdict.loose: FitsTwinVerdict.loose,
    Verdict.long: FitsTwinVerdict.loose,
}


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _require_garment(fs: Any, garment_id: str, user_id: str) -> dict[str, Any]:
    garment = fs.get_garment(garment_id)
    if not garment:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found.")
    if garment.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Garment does not belong to this user.")
    return garment


def _require_twin(fs: Any, twin_id: str, user_id: str) -> dict[str, Any]:
    twin = fs.get_twin(twin_id)
    if not twin:
        raise HTTPException(status_code=404, detail=f"Twin '{twin_id}' not found.")
    if twin.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Body twin does not belong to this user.")
    return twin


def _garment_measurement_zones(garment: dict[str, Any]) -> dict[MeasurementZone, float]:
    measurements = GarmentFlatMeasurements(**garment["measurements"])
    return measurements.to_measurement_zones()


def _twin_measurement_zones(twin: dict[str, Any]) -> dict[MeasurementZone, float]:
    measurements = Measurements(**twin["measurements"])
    return {zone: m.value_cm for zone, m in measurements.zones.items()}


@router.post("/garment/measure")
async def measure_garment(
    image: UploadFile,
    category: str = Form(...),
    user_id: str = Form(...),
    corrected_keypoints: str | None = Form(default=None),
) -> dict[str, Any]:
    """Measure a garment from a flat-lay photo next to an A4 sheet."""
    try:
        garment_category = GarmentCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported category '{category}'. Expected one of: "
            f"{', '.join(c.value for c in GarmentCategory)}.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    parsed_corrections: dict[str, KeypointXY] | None = None
    if corrected_keypoints:
        try:
            raw = json.loads(corrected_keypoints)
            parsed_corrections = {name: KeypointXY(**v) for name, v in raw.items()}
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid corrected_keypoints JSON: {exc}")

    try:
        result = measure_garment_photo(
            image_bytes=image_bytes,
            category=garment_category,
            corrected_keypoints=parsed_corrections,
        )
    except A4NotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    garment_id = f"garment_{uuid.uuid4().hex[:12]}"
    st = get_storage_client()
    fs = get_firestore_client()

    rectified_url = st.upload_bytes(
        destination_path=f"users/{user_id}/garments/{garment_id}/rectified.png",
        data=result.rectified_image_png,
        content_type="image/png",
    )
    overlay_url = st.upload_bytes(
        destination_path=f"users/{user_id}/garments/{garment_id}/overlay.png",
        data=result.overlay_image_png,
        content_type="image/png",
    )

    garment_doc = GarmentMeasurement(
        id=garment_id,
        user_id=user_id,
        category=result.category,
        measurements=result.measurements,
        keypoints=result.keypoints,
        keypoints_source=result.keypoints_source,  # type: ignore[arg-type]
        quality=result.quality,
        a4=result.a4,
        rectified_image_url=rectified_url,
        overlay_image_url=overlay_url,
        created_at=_iso_now(),
    )
    fs.create_garment(garment_id, garment_doc.model_dump())

    return {
        "garment_id": garment_id,
        "category": result.category.value,
        "measurements": result.measurements.model_dump(),
        "keypoints": {k: v.model_dump() for k, v in result.keypoints.items()},
        "keypoints_source": result.keypoints_source,
        "quality": result.quality.model_dump(),
        "a4": result.a4.model_dump(),
        "rectified_image_url": rectified_url,
        "overlay_image_url": overlay_url,
    }


class FitPrefsFromGarmentRequest(BaseModel):
    user_id: str
    garment_id: str
    twin_id: str


@router.post("/me/fit-prefs/from-garment")
def fit_prefs_from_garment(request: FitPrefsFromGarmentRequest) -> dict[str, Any]:
    """Compute preferred ease (garment_cm - body_cm) per compatible zone and save it."""
    fs = get_firestore_client()
    garment = _require_garment(fs, request.garment_id, request.user_id)
    twin = _require_twin(fs, request.twin_id, request.user_id)

    garment_zones = _garment_measurement_zones(garment)
    twin_zones = _twin_measurement_zones(twin)

    ease_cm: dict[MeasurementZone, float] = {}
    for zone, garment_cm in garment_zones.items():
        body_cm = twin_zones.get(zone)
        if body_cm is not None:
            ease_cm[zone] = round(garment_cm - body_cm, 2)

    category = GarmentCategory(garment["category"])
    prefs_doc = FitPrefsDocument(
        user_id=request.user_id,
        category=category,
        garment_id=request.garment_id,
        twin_id=request.twin_id,
        ease_cm=ease_cm,
        created_at=_iso_now(),
    )
    fs.set_fit_prefs(request.user_id, category.value, prefs_doc.model_dump())

    return {
        "category": category.value,
        "ease_cm": {zone.value: val for zone, val in ease_cm.items()},
        "saved_path": f"users/{request.user_id}/fit_prefs/{category.value}",
    }


class UnboxingRequest(BaseModel):
    user_id: str
    garment_id: str
    sku: str | None = Field(
        default=None, description="Which item in the order to unbox; required when the order has more than one item"
    )
    twin_id: str | None = None
    tolerance_cm: float = Field(default=DEFAULT_TOLERANCE_CM, gt=0.0)


def _label_check(zone_diffs: dict[MeasurementZone, float], tolerance_cm: float) -> LabelCheck:
    out_of_tolerance = {z: d for z, d in zone_diffs.items() if abs(d) > tolerance_cm}
    if not out_of_tolerance:
        return LabelCheck.matches
    mean_diff = sum(out_of_tolerance.values()) / len(out_of_tolerance)
    return LabelCheck.larger_than_label if mean_diff > 0 else LabelCheck.smaller_than_label


def _fits_twin_verdict(
    category: GarmentCategory,
    garment_zones: dict[MeasurementZone, float],
    twin_measurements: Measurements,
) -> FitsTwinVerdict:
    """Ask the real fitcore.fit engine, not a local heuristic.

    Wraps the one measured garment as a single-size synthetic Product.size_chart
    (chart_type="garment", so evaluate() uses the same garment_cm - body_cm ease
    formula this endpoint already uses for label_check) and reads back each zone's
    Verdict from the engine's own ease-range/stretch-allowance/body_ref config.
    """
    if not garment_zones:
        return FitsTwinVerdict.unknown

    product = Product(
        id="garment-photo:synthetic",
        brand="measured-fit",
        title=f"{category.value} (measured from photo)",
        category=GARMENT_CATEGORY_TO_FIT_CATEGORY[category],
        fit_type=FitType.regular,
        size_chart=SizeChart(sizes={_SYNTHETIC_SIZE_LABEL: garment_zones}, chart_type="garment"),
    )
    try:
        result = fit_engine_evaluate(twin_measurements, product)
    except ValueError:
        # No zone the engine tracks for this category overlapped what was measured.
        return FitsTwinVerdict.unknown

    size_fit = result.size_fit(_SYNTHETIC_SIZE_LABEL)
    if size_fit is None or not size_fit.zones:
        return FitsTwinVerdict.unknown

    votes = [FIT_VERDICT_TO_FITS_TWIN[zf.verdict] for zf in size_fit.zones.values()]
    tight, loose, fits = votes.count(FitsTwinVerdict.tight), votes.count(FitsTwinVerdict.loose), votes.count(FitsTwinVerdict.fits)
    top = max(
        (FitsTwinVerdict.tight, tight), (FitsTwinVerdict.loose, loose), (FitsTwinVerdict.fits, fits), key=lambda kv: kv[1]
    )
    return top[0]


def _select_order_item(order: dict[str, Any], order_id: str, sku: str | None) -> dict[str, Any]:
    items = order.get("items") or []
    if sku:
        item = next((it for it in items if it.get("sku") == sku), None)
        if item is None:
            raise HTTPException(status_code=422, detail=f"Order '{order_id}' has no item with sku '{sku}'.")
        return item
    if len(items) == 1:
        return items[0]
    raise HTTPException(
        status_code=422, detail=f"Order '{order_id}' has {len(items)} items; specify 'sku' to pick one."
    )


@router.post("/orders/{order_id}/unboxing")
def unboxing(order_id: str, request: UnboxingRequest, store: Store = Depends(get_store)) -> dict[str, Any]:
    """Compare a measured garment against the ordered size label and (optionally) the user's body twin.

    Orders and their product size-chart snapshots live in the Store abstraction used by the rest
    of the API (apilib/stores.py, the same documents apilib/orders.py creates at checkout), not in
    Firestore -- the order's product_snapshots[sku].size_chart is already the exact size chart the
    shopper bought against, so no separate product lookup is needed.
    """
    fs = get_firestore_client()

    order = store.get("orders", order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found.")
    if order.get("uid") != request.user_id:
        raise HTTPException(status_code=403, detail="Order does not belong to this user.")

    item = _select_order_item(order, order_id, request.sku)
    sku = item["sku"]

    garment = _require_garment(fs, request.garment_id, request.user_id)

    snapshot = (order.get("product_snapshots") or {}).get(sku)
    size_chart = snapshot.get("size_chart") if snapshot else None
    ordered_size_label = item["size"]
    if not size_chart or ordered_size_label not in size_chart.get("sizes", {}):
        raise HTTPException(
            status_code=422,
            detail=f"Order '{order_id}' item '{sku}' has no size chart entry for ordered size "
            f"'{ordered_size_label}'.",
        )
    ordered_spec_raw = size_chart["sizes"][ordered_size_label]
    ordered_spec = {MeasurementZone(z): v for z, v in ordered_spec_raw.items()}

    garment_zones = _garment_measurement_zones(garment)

    zone_diffs: dict[MeasurementZone, float] = {}
    for zone, ordered_cm in ordered_spec.items():
        measured_cm = garment_zones.get(zone)
        if measured_cm is not None:
            zone_diffs[zone] = round(measured_cm - ordered_cm, 2)

    label_check = _label_check(zone_diffs, request.tolerance_cm)

    fits_twin = FitsTwinVerdict.unknown
    if request.twin_id:
        twin = _require_twin(fs, request.twin_id, request.user_id)
        twin_measurements = Measurements(**twin["measurements"])
        fits_twin = _fits_twin_verdict(GarmentCategory(garment["category"]), garment_zones, twin_measurements)

    result = UnboxingResult(
        order_id=order_id,
        garment_id=request.garment_id,
        product_id=sku,
        ordered_size_label=ordered_size_label,
        tolerance_cm=request.tolerance_cm,
        label_check=label_check,
        zone_diffs_cm=zone_diffs,
        fits_twin=fits_twin,
    )
    return result.model_dump()
