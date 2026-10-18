"""Shared pieces for the garment measurement tests: the synthetic cases and a stand-in for the landmark model."""

import json
from unittest.mock import MagicMock

import numpy as np

from fitcore.garment import geometry as g
from fitcore.garment.landmarks import FileLandmarkFinder
from fitcore.garment.synthetic import SyntheticPhoto
from fitcore.schemas import Category, MeasurementZone as Z

CASES = {
    "shirt": (Category.shirt, {Z.chest: 114, Z.waist: 110, Z.shoulder_width: 46, Z.sleeve_length: 61.5, Z.garment_length: 74}),
    "tshirt": (Category.tshirt, {Z.chest: 118, Z.shoulder_width: 50, Z.garment_length: 72}),
    "kurta": (Category.kurta, {Z.chest: 112, Z.waist: 116, Z.hip: 120, Z.shoulder_width: 44, Z.sleeve_length: 58, Z.garment_length: 100}),
    "dress": (Category.dress, {Z.chest: 96, Z.waist: 80, Z.hip: 104, Z.shoulder_width: 38, Z.garment_length: 95}),
    "trousers": (Category.trousers, {Z.waist: 88, Z.hip: 108, Z.thigh: 64, Z.inseam: 80}),
    "jacket": (Category.jacket, {Z.chest: 112, Z.shoulder_width: 50, Z.sleeve_length: 65, Z.garment_length: 74}),
}


def oracle(photo: SyntheticPhoto) -> FileLandmarkFinder:
    """Landmarks exactly where the generator put them (photo pixels): checks the geometry without a model."""
    return FileLandmarkFinder({k: list(v) for k, v in photo.landmarks.items()})


def model_reply(photo: SyntheticPhoto, jitter_px: float = 0.0, drop: tuple[str, ...] = (), seed: int = 0) -> MagicMock:
    """A fake genai client whose answer is the true landmarks (optionally jittered or missing), as the model would give them:
    0-1000 across the rectified image it was shown."""
    bgr = g.decode(photo.jpeg)
    view = g.rectify(bgr, g.detect_a4(bgr))
    h, w = view.image.shape[:2]
    rng = np.random.default_rng(seed)
    lms = []
    for name, pt in photo.landmarks.items():
        if name in drop:
            continue
        x, y = view.to_rect(pt)
        x, y = x + rng.normal(0, jitter_px), y + rng.normal(0, jitter_px)
        lms.append({"name": name, "x": round(x / w * 1000, 2), "y": round(y / h * 1000, 2)})
    client = MagicMock()
    client.models.generate_content.return_value.text = json.dumps({"landmarks": lms})
    return client
