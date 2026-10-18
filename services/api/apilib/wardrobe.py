"""Wardrobe ingestion, bounding-box cropping, and perceptual-hash deduplication."""

from __future__ import annotations

import base64
import datetime
import io
import logging
import os
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, Field

from fitcore import llm
from fitcore.schemas import Category, DetectedGarment, IngestDetections, WardrobeItem
from apilib.auth import current_uid
from apilib.stores import Store, get_store

logger = logging.getLogger("api.wardrobe")

router = APIRouter(tags=["wardrobe"])


# ----------------------------------------------------------------------------- Perceptual Hash (dHash)


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash (dHash) for an image using Pillow.

    Grayscale conversion, resize to (hash_size + 1, hash_size), and compare
    adjacent pixel intensities across each row.
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            p1 = resized.getpixel((col, row))
            p2 = resized.getpixel((col + 1, row))
            diff.append("1" if p1 > p2 else "0")
    bits = "".join(diff)
    return f"{int(bits, 2):016x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the Hamming distance (number of bit differences) between two 16-hex hashes."""
    if not hash1 or not hash2:
        return 64
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    return bin(val1 ^ val2).count("1")


def is_duplicate(
    hash_a: str | None,
    cat_a: Category | str,
    color_a: str,
    hash_b: str | None,
    cat_b: Category | str,
    color_b: str,
    threshold: int = 6,
) -> bool:
    """Check if two items are duplicates based on perceptual hash and matching attributes."""
    if not hash_a or not hash_b:
        return False
    cat_a_val = cat_a.value if hasattr(cat_a, "value") else str(cat_a)
    cat_b_val = cat_b.value if hasattr(cat_b, "value") else str(cat_b)
    if cat_a_val.lower() != cat_b_val.lower():
        return False
    if color_a.strip().lower() != color_b.strip().lower():
        return False
    return hamming_distance(hash_a, hash_b) <= threshold


def crop_garment_box(image: Image.Image, box: list[int]) -> Image.Image:
    """Crop image using normalized 0-1000 coordinates [ymin, xmin, ymax, xmax]."""
    if len(box) != 4:
        return image
    ymin, xmin, ymax, xmax = box
    w, h = image.size
    left = int((xmin / 1000.0) * w)
    top = int((ymin / 1000.0) * h)
    right = int((xmax / 1000.0) * w)
    bottom = int((ymax / 1000.0) * h)

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    return image.crop((left, top, right, bottom))


def upload_wardrobe_crop(
    crop_bytes: bytes,
    user_id: str,
    item_id: str,
    storage_client: Any | None = None,
) -> str:
    """Upload cropped garment image to GCS if configured, or fall back to base64 data URI.
    
    Path format: wardrobe/{user_id}/crops/{item_id}.jpg
    Persisted object reference: gs://{bucket_name}/wardrobe/{user_id}/crops/{item_id}.jpg
    """
    bucket_name = os.environ.get("STORAGE_BUCKET") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if bucket_name or storage_client is not None:
        try:
            if storage_client is None:
                from google.cloud import storage
                client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
            else:
                client = storage_client

            b_name = bucket_name or "wardrobe-crops"
            bucket = client.bucket(b_name)
            object_name = f"wardrobe/{user_id}/crops/{item_id}.jpg"
            blob = bucket.blob(object_name)
            blob.upload_from_string(crop_bytes, content_type="image/jpeg")
            return f"gs://{b_name}/{object_name}"
        except Exception as exc:
            logger.warning("GCS crop upload failed (%s); falling back to base64 data URI.", exc)

    thumb_b64 = base64.b64encode(crop_bytes).decode()
    return f"data:image/jpeg;base64,{thumb_b64}"


# ----------------------------------------------------------------------------- Detection & Ingestion


SYSTEM_INGEST = """You detect clothing items in wardrobe flat-lays or outfit photos.
For each garment detected, return:
- box: [ymin, xmin, ymax, xmax] scaled 0 to 1000.
- category: one of tshirt, shirt, kurta, dress, blouse, trousers, jeans, jacket.
- color: primary color name, e.g. blue, white, black, beige, navy.
- pattern: solid, striped, plaid, floral, printed, etc.
- formality: integer 1 (very casual / loungewear) to 5 (black tie / formal evening).
- occasions: list of applicable occasion types, e.g. ["work", "casual", "party", "wedding"].
- season: spring, summer, autumn, winter, or all.
- ethnic_or_western: ethnic, western, or fusion.
Describe garments only. Never make comments on body shape or size."""


def detect_garments_in_image(
    image_bytes: bytes,
    client: Any | None = None,
) -> list[DetectedGarment]:
    """Call Gemini Flash (or offline client) to detect garments in a photo."""
    parts = ["Detect all clothing items in this photo.", image_bytes]
    detections: IngestDetections = llm.generate_json(
        "flash",
        parts,
        IngestDetections,
        system=SYSTEM_INGEST,
        client=client,
    )
    return detections.items


class IngestRequest(BaseModel):
    images: list[str] = Field(
        ...,
        description="List of base64-encoded image strings (with or without data:image/jpeg;base64 prefix)",
    )


class IngestItemResult(BaseModel):
    item: WardrobeItem
    is_duplicate: bool = False
    duplicate_of_id: str | None = None


class IngestResponse(BaseModel):
    added_count: int
    duplicates_count: int
    items: list[IngestItemResult]


def _decode_image_bytes(raw: str) -> bytes:
    if "," in raw:
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


@router.post("/wardrobe/ingest", response_model=IngestResponse)
def ingest_wardrobe_photos(
    req: IngestRequest,
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> IngestResponse:
    """Ingest wardrobe photos: detect garments, crop boxes, compute pHash, and deduplicate."""
    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided for ingestion.")

    existing_docs = store.find("wardrobe", user_id=uid)
    existing_items = [WardrobeItem.model_validate(d) for d in existing_docs]

    results: list[IngestItemResult] = []
    added = 0
    dups = 0

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for idx, img_str in enumerate(req.images):
        try:
            img_bytes = _decode_image_bytes(img_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to decode image %d: %s", idx, exc)
            continue

        try:
            detections = detect_garments_in_image(img_bytes)
        except Exception as exc:
            logger.error("Detection error for image %d: %s", idx, exc)
            continue

        for det in detections:
            crop = crop_garment_box(pil_img, det.box)
            phash = compute_dhash(crop)

            # Check duplicate against existing and newly added items
            dup_id = None
            for ex in existing_items:
                if is_duplicate(phash, det.category, det.color, ex.phash, ex.category, ex.color):
                    dup_id = ex.id
                    break

            if dup_id:
                dups += 1
                fake_item = WardrobeItem(
                    id=uuid.uuid4().hex[:12],
                    user_id=uid,
                    category=det.category,
                    color=det.color,
                    pattern=det.pattern,
                    formality=det.formality,
                    occasions=det.occasions,
                    season=det.season,
                    ethnic_or_western=det.ethnic_or_western,
                    phash=phash,
                    created_at=now_iso,
                )
                results.append(IngestItemResult(item=fake_item, is_duplicate=True, duplicate_of_id=dup_id))
            else:
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=75)
                crop_bytes = buf.getvalue()
                new_id = uuid.uuid4().hex[:12]
                image_url = upload_wardrobe_crop(crop_bytes, uid, new_id)

                new_item = WardrobeItem(
                    id=new_id,
                    user_id=uid,
                    category=det.category,
                    color=det.color,
                    pattern=det.pattern,
                    formality=det.formality,
                    occasions=det.occasions,
                    season=det.season,
                    ethnic_or_western=det.ethnic_or_western,
                    image_url=image_url,
                    phash=phash,
                    created_at=now_iso,
                )
                store.set("wardrobe", new_item.id, new_item.model_dump(mode="json"))
                existing_items.append(new_item)
                results.append(IngestItemResult(item=new_item, is_duplicate=False))
                added += 1

    return IngestResponse(added_count=added, duplicates_count=dups, items=results)


@router.get("/wardrobe", response_model=list[WardrobeItem])
def list_wardrobe_items(
    category: str | None = Query(default=None),
    formality_min: int | None = Query(default=None),
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> list[WardrobeItem]:
    """List wardrobe items for the authenticated user with optional filtering."""
    docs = store.find("wardrobe", user_id=uid)
    items = [WardrobeItem.model_validate(d) for d in docs]
    if category:
        items = [i for i in items if i.category.value == category or i.category == category]
    if formality_min:
        items = [i for i in items if i.formality >= formality_min]
    return items
