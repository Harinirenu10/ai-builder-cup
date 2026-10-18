"""Tests for wardrobe ingestion, cropping, pHash computation, and deduplication."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

from PIL import Image, ImageDraw
import pytest

from fitcore.schemas import Category, DetectedGarment
from apilib.wardrobe import (
    compute_dhash,
    crop_garment_box,
    hamming_distance,
    is_duplicate,
)
from tests.conftest import user


def _create_test_image(color: str = "blue", size: tuple[int, int] = (200, 200)) -> Image.Image:
    """Helper to create a simple solid/patterned PIL Image."""
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill="white")
    return img


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_compute_dhash_and_hamming_distance():
    img1 = _create_test_image("blue")
    img2 = _create_test_image("blue")
    
    hash1 = compute_dhash(img1)
    hash2 = compute_dhash(img2)
    assert len(hash1) == 16
    assert hash1 == hash2
    assert hamming_distance(hash1, hash2) == 0

    # Slightly modified image
    img_mod = _create_test_image("blue")
    draw = ImageDraw.Draw(img_mod)
    draw.rectangle([52, 52, 148, 148], fill="black")
    hash_mod = compute_dhash(img_mod)
    dist = hamming_distance(hash1, hash_mod)
    assert dist >= 0


def test_is_duplicate_logic():
    h1 = "0123456789abcdef"
    h2 = "0123456789abcdef"
    h_diff = "fedcba9876543210"

    # Exact match
    assert is_duplicate(h1, Category.shirt, "blue", h2, Category.shirt, "blue")

    # Different category
    assert not is_duplicate(h1, Category.shirt, "blue", h2, Category.jacket, "blue")

    # Different color
    assert not is_duplicate(h1, Category.shirt, "blue", h2, Category.shirt, "red")

    # Large hash difference
    assert not is_duplicate(h1, Category.shirt, "blue", h_diff, Category.shirt, "blue")


def test_crop_garment_box():
    img = Image.new("RGB", (1000, 1000), "red")
    box = [100, 200, 500, 600]
    cropped = crop_garment_box(img, box)
    assert cropped.size == (400, 400)

    # Test clamping with out-of-range coordinates
    clamped_box = [-50, -20, 1200, 1500]
    cropped_clamped = crop_garment_box(img, clamped_box)
    assert cropped_clamped.size[0] <= 1000
    assert cropped_clamped.size[1] <= 1000


def test_wardrobe_ingest_and_deduplication(client):
    test_img = _create_test_image("navy")
    b64_str = _img_to_b64(test_img)

    fake_detection = DetectedGarment(
        box=[100, 100, 900, 900],
        category=Category.shirt,
        color="navy",
        pattern="solid",
        formality=4,
        occasions=["office", "meeting"],
        season="all",
        ethnic_or_western="western",
    )

    with patch("apilib.wardrobe.detect_garments_in_image", return_value=[fake_detection]):
        # 1. First ingestion: new item should be added
        res1 = client.post(
            "/wardrobe/ingest",
            json={"images": [b64_str]},
            headers=user("shopper_1"),
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["added_count"] == 1
        assert data1["duplicates_count"] == 0
        added_id = data1["items"][0]["item"]["id"]

        # 2. Second ingestion of identical image: should be detected as duplicate
        res2 = client.post(
            "/wardrobe/ingest",
            json={"images": [b64_str]},
            headers=user("shopper_1"),
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["added_count"] == 0
        assert data2["duplicates_count"] == 1
        assert data2["items"][0]["is_duplicate"] is True
        assert data2["items"][0]["duplicate_of_id"] == added_id


def test_wardrobe_list_and_filters(client):
    img = _create_test_image("white")
    b64_str = _img_to_b64(img)

    detections = [
        DetectedGarment(
            box=[0, 0, 500, 500],
            category=Category.shirt,
            color="white",
            pattern="solid",
            formality=4,
            occasions=["office"],
            ethnic_or_western="western",
        ),
        DetectedGarment(
            box=[500, 0, 1000, 500],
            category=Category.tshirt,
            color="black",
            pattern="solid",
            formality=1,
            occasions=["casual"],
            ethnic_or_western="western",
        ),
    ]

    with patch("apilib.wardrobe.detect_garments_in_image", return_value=detections):
        client.post(
            "/wardrobe/ingest",
            json={"images": [b64_str]},
            headers=user("shopper_2"),
        )

    # Query all
    r_all = client.get("/wardrobe", headers=user("shopper_2"))
    assert r_all.status_code == 200
    assert len(r_all.json()) == 2

    # Query filter by category
    r_shirt = client.get("/wardrobe?category=shirt", headers=user("shopper_2"))
    assert r_shirt.status_code == 200
    assert len(r_shirt.json()) == 1
    assert r_shirt.json()[0]["category"] == "shirt"

    # Query filter by formality_min
    r_formal = client.get("/wardrobe?formality_min=3", headers=user("shopper_2"))
    assert r_formal.status_code == 200
    assert len(r_formal.json()) == 1
    assert r_formal.json()[0]["formality"] == 4


def test_crop_upload_to_gcs_success(monkeypatch):
    from unittest.mock import MagicMock
    from apilib.wardrobe import upload_wardrobe_crop

    monkeypatch.setenv("STORAGE_BUCKET", "prod-wardrobe-bucket")

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket

    crop_bytes = b"fake-jpeg-data"
    user_id = "user-123"
    item_id = "item-abc"

    uri = upload_wardrobe_crop(crop_bytes, user_id, item_id, storage_client=mock_storage_client)

    mock_storage_client.bucket.assert_called_once_with("prod-wardrobe-bucket")
    mock_bucket.blob.assert_called_once_with("wardrobe/user-123/crops/item-abc.jpg")
    mock_blob.upload_from_string.assert_called_once_with(crop_bytes, content_type="image/jpeg")
    assert uri == "gs://prod-wardrobe-bucket/wardrobe/user-123/crops/item-abc.jpg"


def test_crop_upload_gcs_failure_falls_back_to_base64(monkeypatch):
    from unittest.mock import MagicMock
    from apilib.wardrobe import upload_wardrobe_crop

    monkeypatch.setenv("STORAGE_BUCKET", "prod-wardrobe-bucket")

    mock_bucket = MagicMock()
    mock_bucket.blob.side_effect = RuntimeError("GCS connection timeout")
    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket

    crop_bytes = b"fake-jpeg-data"
    uri = upload_wardrobe_crop(crop_bytes, "u1", "i1", storage_client=mock_storage_client)

    # Must cleanly fall back to base64 data URI without raising
    assert uri.startswith("data:image/jpeg;base64,")


def test_crop_offline_fallback_when_no_bucket(monkeypatch):
    from apilib.wardrobe import upload_wardrobe_crop

    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    crop_bytes = b"fake-offline-jpeg-bytes"
    uri = upload_wardrobe_crop(crop_bytes, "u2", "i2")
    assert uri.startswith("data:image/jpeg;base64,")
