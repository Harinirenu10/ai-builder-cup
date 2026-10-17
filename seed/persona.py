"""The demo persona's closet and calendar. Everything here is SIMULATED.

Closet pieces that the fit engine can measure get a one-size garment chart built from the persona's twin: the ideal ease
for the cut plus a per-zone nudge, so a few pieces are deliberately close-cut and show up in the occasion test.
Calendar days are stored as offsets ("tomorrow", "next Saturday") and resolved when a sandbox is created.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from fitcore.fit.config import get_config
from fitcore.occasion import CalendarEvent, ClosetItem, calendar_event
from fitcore.schemas import Category, FitType, MeasurementZone as Z, SizeChart

ROOT = Path(__file__).resolve().parents[1]
PERSONA_UID = "demo-persona"
HEIGHT_CM = 172.0


def twin_zones() -> dict[str, dict[str, Any]]:
    return json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))["zones"]


def seated_delta() -> dict[str, float]:
    return json.loads((ROOT / "samples" / "poses_seated.json").read_text(encoding="utf-8"))["poses"]["seated"]["delta_cm"]


def chart_for(category: Category, fit_type: FitType, size: str, nudge: dict[str, float] | None = None) -> SizeChart:
    """Garment chart (cm) for a piece that fits the persona with ideal ease, plus a nudge in cm per zone."""
    body = {k: v["value_cm"] for k, v in twin_zones().items()}
    table = get_config().target_ease_cm[category][fit_type]
    row: dict[Z, float] = {}
    for zone, rng in table.items():
        ref = "torso_length" if zone == Z.garment_length else zone.value
        if ref in body:
            row[zone] = round(body[ref] + rng.ideal + (nudge or {}).get(zone.value, 0.0), 1)
    return SizeChart(sizes={size: row}, chart_type="garment", size_system="ALPHA", unit="cm")


# slot, category, fit, style, formality, circles, colour, size, nudge
_CLOSET: list[tuple] = [
    ("White oxford shirt", "top", "shirt", "regular", "western", 3, ["work", "friends"], "white", "M", {}),
    ("Navy slim-fit formal shirt", "top", "shirt", "slim", "western", 4, ["work"], "navy", "M", {}),
    ("Light blue linen shirt", "top", "shirt", "regular", "western", 2, ["friends", "family"], "light blue", "M", {}),
    ("Black cotton crew tee", "top", "tshirt", "regular", "western", 1, ["friends"], "black", "M", {}),
    ("Grey heavyweight tee", "top", "tshirt", "relaxed", "western", 1, ["friends"], "grey", "M", {}),
    ("Ivory silk-blend kurta", "top", "kurta", "regular", "ethnic", 4, ["family", "community"], "ivory", "M", {}),
    ("Mustard cotton kurta", "top", "kurta", "regular", "ethnic", 3, ["family", "friends", "community"], "mustard", "M", {}),
    ("Maroon embroidered festive kurta", "top", "kurta", "regular", "ethnic", 5, ["family"], "maroon", "M", {"chest": -5.0}),
    ("Charcoal tailored trousers", "bottom", "trousers", "regular", "western", 4, ["work", "family"], "charcoal", "32", {"waist": 2.5}),
    ("Beige cotton chinos", "bottom", "trousers", "regular", "western", 3, ["work", "friends"], "beige", "32", {"waist": 2.5}),
    ("Slim dark-wash jeans", "bottom", "jeans", "slim", "western", 2, ["friends"], "dark indigo", "32", {"thigh": -3.0, "waist": 2.0}),
    ("Cream churidar trousers", "bottom", "trousers", "slim", "ethnic", 4, ["family", "community"], "cream", "32", {"thigh": -4.5}),
    ("Navy wool-blend blazer", "layer", "jacket", "regular", "western", 4, ["work", "family"], "navy", "40", {}),
    ("Olive canvas utility jacket", "layer", "jacket", "regular", "western", 2, ["friends"], "olive", "40", {}),
    ("Black bandhgala jacket", "layer", "jacket", "regular", "ethnic", 5, ["family", "community"], "black", "40", {}),
    ("Black leather oxford shoes", "shoes", None, "regular", "western", 4, ["work", "family"], "black", "", {}),
    ("Tan leather loafers", "shoes", None, "regular", "either", 3, ["work", "friends", "family"], "tan", "", {}),
    ("Embroidered kolhapuri sandals", "shoes", None, "regular", "ethnic", 3, ["family", "friends", "community"], "tan", "", {}),
    ("White leather sneakers", "shoes", None, "regular", "western", 1, ["friends"], "white", "", {}),
    ("Silk-blend embroidered stole", "accessory", None, "regular", "ethnic", 4, ["family", "community"], "gold", "", {}),
]


def closet() -> list[ClosetItem]:
    out = []
    for n, (name, slot, cat, fit, style, formality, circles, color, size, nudge) in enumerate(_CLOSET, start=1):
        category = Category(cat) if cat else None
        chart = chart_for(category, FitType(fit), size, nudge) if category and size else None
        out.append(ClosetItem(id=f"closet-{n:02d}", name=name, slot=slot, category=category, fit_type=FitType(fit), style=style, formality=formality,  # type: ignore[arg-type]
                              circles=circles, color=color, size_label=size, chart=chart))
    return out


# id, title, day ("saturday" or days from today), clock time, duration min, circle, formality, style, dress code, seated minutes
_EVENTS: list[tuple] = [
    ("event-1", "Client meeting", 1, "10:00", 90, "work", 4, "western", "business formal", 90),
    ("event-2", "Wedding reception", 1, "19:00", 240, "family", 5, "ethnic", "festive ethnic, formal", 150),
    ("event-3", "Sangeet", "saturday", "18:30", 300, "family", 4, "ethnic", "festive ethnic", 60),
    ("event-4", "Lunch with college friends", 3, "13:00", 120, "friends", 2, "either", "smart casual", 90),
    ("event-5", "Community festival volunteering", 5, "09:00", 240, "community", 3, "ethnic", "simple ethnic", 30),
]


def event_docs() -> list[dict[str, Any]]:
    """Calendar rows as stored: the day is relative ("tomorrow", "saturday") and becomes a date when the event is read."""
    return [{"event_id": eid, "title": title, "when": {"day": day, "time": clock}, "duration_min": dur, "circle": circle, "formality": formality,
             "style": style, "dress_code": code, "seated_minutes": seated} for eid, title, day, clock, dur, circle, formality, style, code, seated in _EVENTS]


def events(today: date | None = None) -> list[CalendarEvent]:
    return [calendar_event(d, today) for d in event_docs()]
