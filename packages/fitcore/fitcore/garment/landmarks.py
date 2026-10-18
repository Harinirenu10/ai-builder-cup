"""Where on the garment to measure. A model (or a person, or a file) places landmarks; the maths is in measure.py."""

import io
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
from pydantic import BaseModel, Field

from fitcore import llm
from fitcore.garment.geometry import MeasureError, View
from fitcore.schemas import Category, MeasurementZone as Z

# Zone -> (landmark a, landmark b, multiplier). The distance a-b on the flat garment times the multiplier (2 turns a flat
# width into a circumference).
PAIRS: dict[Z, tuple[str, str, int]] = {
    Z.chest: ("armpit_left", "armpit_right", 2),
    Z.waist: ("waist_left", "waist_right", 2),
    Z.hip: ("hip_left", "hip_right", 2),
    Z.shoulder_width: ("shoulder_tip_left", "shoulder_tip_right", 1),
    Z.sleeve_length: ("shoulder_tip_left", "sleeve_cuff_left", 1),
    Z.thigh: ("thigh_outer_left", "crotch", 2),
    Z.inseam: ("crotch", "leg_hem_inner_left", 1),
}
# Points that sit inside the outline in a flat-lay (the slit between the legs is often too narrow to see), so they are not pulled onto it.
INTERIOR_OK = {"crotch"}
GARMENT_LENGTH_LANDMARKS = ("neck_left", "neck_right", "hem_left", "hem_right")  # high point of the shoulder to the hem
MEASURABLE = set(PAIRS) | {Z.garment_length}

DEFINITIONS = {
    "neck_left": "where the left shoulder seam meets the collar or neckline (the highest point of the shoulder next to the neck)",
    "neck_right": "where the right shoulder seam meets the collar or neckline",
    "shoulder_tip_left": "the outer end of the left shoulder seam, where the sleeve is sewn on",
    "shoulder_tip_right": "the outer end of the right shoulder seam",
    "armpit_left": "the underarm point where the left sleeve seam meets the side seam of the body",
    "armpit_right": "the underarm point where the right sleeve seam meets the side seam of the body",
    "waist_left": "the left edge of the garment at its narrowest point (for trousers: the left end of the waistband)",
    "waist_right": "the right edge of the garment at its narrowest point (for trousers: the right end of the waistband)",
    "hip_left": "the left edge of the garment at hip level (tops: about 20 cm below the waist; trousers: about 18 cm below the waistband)",
    "hip_right": "the right edge of the garment at the same hip level",
    "hem_left": "the lower left corner of the body hem",
    "hem_right": "the lower right corner of the body hem",
    "sleeve_cuff_left": "the middle of the end edge (cuff) of the same sleeve as shoulder_tip_left",
    "thigh_outer_left": "the outer edge of the left leg at the level of the crotch",
    "crotch": "the point where the two legs and the inner seams meet",
    "leg_hem_inner_left": "the inner bottom corner of the same leg's hem (the corner nearest the other leg)",
}


def required_landmarks(category: Category, zones: list[Z]) -> list[str]:
    names: list[str] = []
    for z in zones:
        if z in PAIRS:
            names += list(PAIRS[z][:2])
        elif z == Z.garment_length:
            names += list(GARMENT_LENGTH_LANDMARKS)
    return list(dict.fromkeys(names))


class Landmark(BaseModel):
    name: str
    x: float = Field(description="0 to 1000 across the image shown, 0 at the left edge")
    y: float = Field(description="0 to 1000 down the image shown, 0 at the top edge")


class LandmarkReply(BaseModel):
    landmarks: list[Landmark]


class LandmarkFinder(Protocol):
    def __call__(self, view: View, category: Category, names: list[str]) -> dict[str, tuple[float, float]]:
        """Landmark name -> point in the rectified view's pixels. Names it cannot place are left out."""


SYSTEM = (
    "You place landmarks on a photo of a garment laid flat, seen from directly above. The white rectangle is an A4 sheet: "
    "ignore it. The garment may be turned at any angle. Return one point per requested landmark, as x and y between 0 and 1000 "
    "across and down the image (0,0 is the top-left corner). Put each point exactly on the garment's outline or seam as defined. "
    "'left' and 'right' just name the two sides: a '..._left' landmark and its partner belong to the same side. "
    "Leave out any landmark you cannot see; never guess."
)


class LlmLandmarkFinder:
    """Gemini (pro) looks at the rectified view and places the landmarks. Through fitcore.llm like every model call."""

    def __init__(self, client: Any | None = None, ollama_transport: Any | None = None, max_side: int = 1280):
        self.client, self.ollama_transport, self.max_side = client, ollama_transport, max_side

    def __call__(self, view: View, category: Category, names: list[str]) -> dict[str, tuple[float, float]]:
        img = view.image
        k = min(1.0, self.max_side / max(img.shape[:2]))
        shown = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise MeasureError("bad_image", "Could not prepare the photo.")
        listing = "\n".join(f"- {n}: {DEFINITIONS[n]}" for n in names)
        prompt = f"Garment category: {category.value}.\nPlace these landmarks:\n{listing}"
        reply = llm.generate_json("pro", [prompt, jpg.tobytes()], LandmarkReply, system=SYSTEM, client=self.client, ollama_transport=self.ollama_transport)
        h, w = img.shape[:2]
        return {lm.name: (lm.x / 1000.0 * w, lm.y / 1000.0 * h) for lm in reply.landmarks if lm.name in names and 0 <= lm.x <= 1000 and 0 <= lm.y <= 1000}


class FileLandmarkFinder:
    """Landmarks written down in advance, as photo pixels: {"armpit_left": [x, y], ...}. For samples, tests and hand-annotated photos."""

    def __init__(self, points: dict[str, list[float]] | Path | str):
        self.points = json.loads(Path(points).read_text(encoding="utf-8")) if isinstance(points, (str, Path)) else points

    def __call__(self, view: View, category: Category, names: list[str]) -> dict[str, tuple[float, float]]:
        return {n: view.to_rect((self.points[n][0], self.points[n][1])) for n in names if n in self.points}


FinderFactory = Callable[[], LandmarkFinder]
