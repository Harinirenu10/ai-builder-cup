"""Tests for services/api/garment_routes.py: measure, fit-prefs, unboxing."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import apilib.stores as stores_module
from apilib.stores import MemoryStore
from fitcore.schemas import MeasurementZone
from services.api.main import app
from services.geometry.cloud_client import (
    InMemoryFirestoreClient,
    InMemoryStorageClient,
    set_cloud_clients,
)

client = TestClient(app)

CORRECTED_TOP_KEYPOINTS = {
    "left_armpit": {"x": 300, "y": 300},
    "right_armpit": {"x": 700, "y": 300},
    "left_shoulder_seam": {"x": 320, "y": 200},
    "right_shoulder_seam": {"x": 680, "y": 200},
    "back_neck_point": {"x": 500, "y": 150},
    "hem_center": {"x": 500, "y": 800},
    "left_cuff_outer": {"x": 250, "y": 500},
}


@pytest.fixture
def mock_cloud():
    fs = InMemoryFirestoreClient()
    st = InMemoryStorageClient()
    set_cloud_clients(firestore_client=fs, storage_client=st)
    return fs, st


@pytest.fixture
def mock_store(monkeypatch):
    """Orders/products for unboxing live in apilib's Store (same abstraction apilib/orders.py
    writes to at checkout), not Firestore -- see services/api/garment_routes.py::unboxing."""
    store = MemoryStore()
    monkeypatch.setattr(stores_module, "_store", store)
    return store


def _make_a4_jpeg_bytes() -> bytes:
    img = np.full((1600, 1600, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (600, 766), (255, 255, 255), -1)  # 400x566 ~= A4 ratio
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def _measure_garment(user_id: str = "user_1") -> dict:
    resp = client.post(
        "/garment/measure",
        data={
            "category": "tops",
            "user_id": user_id,
            "corrected_keypoints": json.dumps(CORRECTED_TOP_KEYPOINTS),
        },
        files={"image": ("shirt.jpg", _make_a4_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_measure_garment_with_corrected_keypoints_skips_gemini(mock_cloud):
    fs, st = mock_cloud
    data = _measure_garment()

    assert data["keypoints_source"] == "user_corrected"
    assert data["category"] == "tops"
    assert data["measurements"]["chest_flat_cm"] > 0
    assert data["measurements"]["chest_circ_cm"] == pytest.approx(data["measurements"]["chest_flat_cm"] * 2)
    assert data["a4"]["a4_found"] is True
    assert data["rectified_image_url"].startswith("gs://")
    assert data["overlay_image_url"].startswith("gs://")

    garment = fs.get_garment(data["garment_id"])
    assert garment is not None
    assert garment["user_id"] == "user_1"


def test_measure_garment_missing_a4_returns_ui_safe_error(mock_cloud):
    plain_img = np.full((800, 800, 3), 60, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", plain_img)
    assert ok

    resp = client.post(
        "/garment/measure",
        data={"category": "tops", "user_id": "user_1"},
        files={"image": ("shirt.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Place a full A4 sheet next to the garment."


def test_measure_garment_rejects_unknown_category(mock_cloud):
    resp = client.post(
        "/garment/measure",
        data={"category": "dresses", "user_id": "user_1"},
        files={"image": ("shirt.jpg", _make_a4_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "Unsupported category" in resp.json()["detail"]


def _make_twin(fs, twin_id: str, user_id: str, chest_cm: float) -> None:
    fs.create_twin(
        twin_id,
        {
            "id": twin_id,
            "user_id": user_id,
            "measurements": {
                "zones": {
                    "chest": {"value_cm": chest_cm, "confidence": 0.9, "source": "3d_twin"},
                }
            },
        },
    )


def test_fit_prefs_from_garment_computes_ease(mock_cloud):
    fs, _ = mock_cloud
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _make_twin(fs, "twin_1", "user_1", chest_cm=chest_circ - 5.0)

    resp = client.post(
        "/me/fit-prefs/from-garment",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "twin_id": "twin_1"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ease_cm"]["chest"] == pytest.approx(5.0, abs=0.05)
    assert fs.get_fit_prefs("user_1", "tops") is not None


def test_fit_prefs_rejects_garment_owned_by_other_user(mock_cloud):
    fs, _ = mock_cloud
    garment_data = _measure_garment(user_id="user_1")
    _make_twin(fs, "twin_1", "user_2", chest_cm=90.0)

    resp = client.post(
        "/me/fit-prefs/from-garment",
        json={"user_id": "user_2", "garment_id": garment_data["garment_id"], "twin_id": "twin_1"},
    )
    assert resp.status_code == 403


def _seed_order(
    store,
    order_id: str,
    chest_cm_for_size: float | None,
    sku: str = "sku_shirt",
    size_label: str = "M",
    user_id: str = "user_1",
    extra_items: list[dict] | None = None,
) -> None:
    """Seed an order the way apilib/orders.py::create_order actually writes one:
    a Store document with uid, multi-item 'items', and a per-sku product_snapshots
    carrying the size chart the shopper bought against."""
    size_chart = (
        None
        if chest_cm_for_size is None
        else {"sizes": {size_label: {"chest": chest_cm_for_size}}, "chart_type": "garment", "size_system": "ALPHA"}
    )
    items = [{"sku": sku, "title": "Classic Shirt", "qty": 1, "size": size_label}]
    if extra_items:
        items.extend(extra_items)
    store.set(
        "orders",
        order_id,
        {
            "uid": user_id,
            "items": items,
            "product_snapshots": {sku: {"title": "Classic Shirt", "category": "shirt", "fit_type": "regular", "size_chart": size_chart}},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )


def test_unboxing_matches_label_within_tolerance(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ)  # exact match

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label_check"] == "matches"
    assert data["zone_diffs_cm"]["chest"] == pytest.approx(0.0, abs=0.05)
    assert data["product_id"] == "sku_shirt"  # the sku, since orders are now multi-item
    assert data["fits_twin"] == "unknown"  # no twin_id supplied


def test_unboxing_detects_larger_than_label(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ - 3.0)  # garment measures 3cm bigger

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "tolerance_cm": 1.5},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label_check"] == "larger_than_label"
    assert data["zone_diffs_cm"]["chest"] == pytest.approx(3.0, abs=0.05)


def test_unboxing_fits_twin_verdict_ideal_via_real_fit_engine(mock_cloud, mock_store):
    """fits_twin now comes from fitcore.fit.evaluate(), not a local heuristic.

    shirt/regular/chest ease range is [10, 16, 22] cm (fitcore/config/ease.yaml) with
    stretch_allowance_cm.none = 0.0, so an ease of exactly 16cm (the table's own ideal)
    must classify as Verdict.ideal -> FitsTwinVerdict.fits.
    """
    fs, _ = mock_cloud
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ)
    _make_twin(fs, "twin_1", "user_1", chest_cm=chest_circ - 16.0)

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "twin_id": "twin_1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["fits_twin"] == "fits"


def test_unboxing_fits_twin_verdict_tight_via_real_fit_engine(mock_cloud, mock_store):
    """Ease of 5cm is below the shirt/regular/chest range's min (10cm) -> Verdict.tight."""
    fs, _ = mock_cloud
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ)
    _make_twin(fs, "twin_1", "user_1", chest_cm=chest_circ - 5.0)

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "twin_id": "twin_1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["fits_twin"] == "tight"


def test_unboxing_fits_twin_verdict_loose_via_real_fit_engine(mock_cloud, mock_store):
    """Ease of 30cm is above the shirt/regular/chest range's max (22cm) -> Verdict.loose."""
    fs, _ = mock_cloud
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ)
    _make_twin(fs, "twin_1", "user_1", chest_cm=chest_circ - 30.0)

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "twin_id": "twin_1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["fits_twin"] == "loose"


def test_unboxing_rejects_order_owned_by_other_user(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    _seed_order(mock_store, "order_1", chest_cm_for_size=garment_data["measurements"]["chest_circ_cm"])

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_2", "garment_id": garment_data["garment_id"]},
    )
    assert resp.status_code == 403


def test_unboxing_missing_size_chart_entry_returns_422(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    _seed_order(mock_store, "order_1", chest_cm_for_size=None)

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"]},
    )
    assert resp.status_code == 422


def test_unboxing_multi_item_order_requires_sku(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(
        mock_store, "order_1", chest_cm_for_size=chest_circ,
        extra_items=[{"sku": "sku_jeans", "title": "Jeans", "qty": 1, "size": "32"}],
    )

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"]},
    )
    assert resp.status_code == 422
    assert "sku" in resp.json()["detail"]


def test_unboxing_multi_item_order_selects_item_by_sku(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(
        mock_store, "order_1", chest_cm_for_size=chest_circ,
        extra_items=[{"sku": "sku_jeans", "title": "Jeans", "qty": 1, "size": "32"}],
    )

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "sku": "sku_shirt"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["product_id"] == "sku_shirt"
    assert data["label_check"] == "matches"


def test_unboxing_unknown_sku_returns_422(mock_cloud, mock_store):
    garment_data = _measure_garment(user_id="user_1")
    chest_circ = garment_data["measurements"]["chest_circ_cm"]
    _seed_order(mock_store, "order_1", chest_cm_for_size=chest_circ)

    resp = client.post(
        "/orders/order_1/unboxing",
        json={"user_id": "user_1", "garment_id": garment_data["garment_id"], "sku": "does_not_exist"},
    )
    assert resp.status_code == 422
