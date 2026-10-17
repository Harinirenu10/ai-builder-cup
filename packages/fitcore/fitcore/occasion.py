"""Occasion test: for a calendar event, pick the closet items that suit the dress code, the circle and the time spent seated.

Deterministic. The fit part is the real fit engine run on each item's own garment chart with the event's seated minutes
applied as a pose; no model call. The words describe the garment and the event, never the body.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from fitcore.fit.engine import evaluate
from fitcore.schemas import (
    Category,
    FabricPart,
    FitType,
    Measurements,
    PoseDelta,
    PoseDeltas,
    Product,
    SizeChart,
    StretchClass,
    Verdict,
)

Slot = Literal["top", "bottom", "layer", "shoes", "accessory"]
Style = Literal["western", "ethnic", "either"]
Circle = Literal["work", "family", "friends", "community"]

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def resolve_start(day: int | str, clock: str, today: date | None = None) -> str:
    """ISO start time for a relative day: an offset in days (1 = tomorrow) or "saturday" (the first Saturday at least 2 days away), India time."""
    today = today or datetime.now(IST).date()
    if day == "saturday":
        d = today + timedelta(days=2)
        d += timedelta(days=(5 - d.weekday()) % 7)
    else:
        d = today + timedelta(days=int(day))
    hh, mm = clock.split(":")
    return datetime.combine(d, time(int(hh), int(mm)), tzinfo=IST).isoformat()


REQUIRED_SLOTS: tuple[Slot, ...] = ("top", "bottom", "shoes")


class ClosetItem(BaseModel):
    id: str
    name: str
    slot: Slot
    category: Category | None = Field(default=None, description="Fit-engine category; None for shoes and accessories")
    fit_type: FitType = FitType.regular
    style: Style
    formality: int = Field(ge=1, le=5)
    circles: list[Circle]
    color: str
    size_label: str = ""
    chart: SizeChart | None = Field(default=None, description="Single-size garment chart in cm")
    fabric: list[FabricPart] = Field(default_factory=list)
    stretch_class: StretchClass = StretchClass.none
    simulated: bool = True


class CalendarEvent(BaseModel):
    id: str
    title: str
    starts_at: str = Field(description="ISO 8601 with offset")
    duration_min: int
    circle: Circle
    formality: int = Field(ge=1, le=5, description="Dress code: 1 casual to 5 formal")
    style: Style
    dress_code: str
    seated_minutes: int = Field(default=0, ge=0)


def calendar_event(doc: dict, today: date | None = None) -> CalendarEvent:
    """A stored calendar row (see seed/persona.py) as a dated event."""
    return CalendarEvent(id=doc["event_id"], title=doc["title"], starts_at=resolve_start(doc["when"]["day"], doc["when"]["time"], today),
                         duration_min=doc["duration_min"], circle=doc["circle"], formality=doc["formality"], style=doc["style"],
                         dress_code=doc["dress_code"], seated_minutes=doc.get("seated_minutes", 0))


class Pick(BaseModel):
    item_id: str
    name: str
    slot: Slot
    score: float
    reasons: list[str]
    fit_notes: list[str] = Field(default_factory=list)
    fit_state: Literal["measured_ok", "measured_watch", "not_measured"] = "not_measured"


class OccasionResult(BaseModel):
    event_id: str
    event_title: str
    outfit: list[Pick]
    alternatives: dict[str, list[Pick]] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    seated_minutes: int = 0
    notes: list[str] = Field(default_factory=list)


def _fit_check(item: ClosetItem, event: CalendarEvent, twin: Measurements, seated_delta: dict | None) -> tuple[float, list[str], str]:
    """(score change, notes, state) from the fit engine on this item's own chart, with the event's seated minutes as a pose."""
    if item.chart is None or item.category is None or not item.size_label or item.size_label not in item.chart.sizes:
        return 0.0, [], "not_measured"
    poses = None
    if seated_delta and event.seated_minutes > 0:
        poses = PoseDeltas(poses={"seated": PoseDelta(minutes=float(event.seated_minutes), delta_cm=seated_delta)})
    product = Product(id=item.id, brand="closet", title=item.name, category=item.category, fit_type=item.fit_type, fabric=item.fabric,
                      stretch_class=item.stretch_class, pre_shrunk=True, size_chart=item.chart)  # closet pieces have been washed already
    try:
        result = evaluate(twin, product, None, None, poses)
    except ValueError:
        return 0.0, [], "not_measured"
    sf = result.size_fit(item.size_label)
    if sf is None:
        return 0.0, [], "not_measured"
    delta, notes, watch = 5.0, [], False
    for z in sf.zones.values():
        label = z.zone.value.replace("_", " ")
        strain = [(name, p) for name, p in z.poses.items() if p.strain]
        if strain:
            name, p = strain[0]
            notes.append(f"Seated for {event.seated_minutes} min, the {label} of this garment keeps {p.ease_cm:.1f} cm of ease, below the {z.target_min_cm:.1f} cm minimum for its cut.")
            delta -= 25.0
            watch = True
        elif z.hard and Verdict.tight in (z.verdict, z.verdict_after_wash):
            notes.append(f"The {label} of this garment is cut closer than its intended fit.")
            delta -= 20.0
            watch = True
        elif Verdict.loose in (z.verdict, z.verdict_after_wash):
            delta -= 2.0
    if not watch:
        notes.append(f"Measured against your twin at size {item.size_label}: the ease is within the range for this cut" + (f", also when seated {event.seated_minutes} min." if poses else "."))
    return delta, notes, "measured_watch" if watch else "measured_ok"


def score_item(item: ClosetItem, event: CalendarEvent, twin: Measurements, seated_delta: dict | None) -> Pick:
    score, reasons = 100.0, []
    gap = abs(item.formality - event.formality)
    score -= 18.0 * gap
    reasons.append(f"Formality {item.formality} of 5 against a dress code of {event.formality} ({event.dress_code}).")
    if item.style == event.style:
        score += 10.0
        reasons.append(f"Its {item.style} style matches the event.")
    elif "either" in (item.style, event.style):
        score += 4.0
    else:
        score -= 20.0
        reasons.append(f"Its {item.style} style differs from the {event.style} style the event calls for.")
    if event.circle in item.circles:
        score += 8.0
        reasons.append(f"Tagged for your {event.circle} circle.")
    delta, notes, state = _fit_check(item, event, twin, seated_delta)
    return Pick(item_id=item.id, name=item.name, slot=item.slot, score=round(score + delta, 1), reasons=reasons, fit_notes=notes, fit_state=state)  # type: ignore[arg-type]


def occasion_test(event: CalendarEvent, closet: list[ClosetItem], twin: Measurements, seated_delta: dict | None = None,
                  alternatives: int = 2) -> OccasionResult:
    """Best item per slot (top, bottom, shoes always; layer for dress codes of 4 and up; accessory for formal ethnic events)."""
    slots: list[Slot] = list(REQUIRED_SLOTS)
    if event.formality >= 4:
        slots.insert(2, "layer")
    if event.formality >= 4 and event.style == "ethnic":
        slots.append("accessory")
    picks_by_slot: dict[str, list[Pick]] = {}
    for item in closet:
        if item.slot in slots:
            picks_by_slot.setdefault(item.slot, []).append(score_item(item, event, twin, seated_delta))
    outfit, alts, missing = [], {}, []
    for slot in slots:
        ranked = sorted(picks_by_slot.get(slot, []), key=lambda p: (-p.score, p.name))
        if not ranked:
            missing.append(slot)
            continue
        outfit.append(ranked[0])
        alts[slot] = ranked[1:1 + alternatives]
    notes = []
    if any(p.fit_state == "measured_watch" for p in outfit):
        notes.append("One chosen piece is the best match for the dress code but needs a look at the fit notes.")
    if event.seated_minutes:
        notes.append(f"Seated time of {event.seated_minutes} min is included in the fit check for pieces that have a measured chart.")
    return OccasionResult(event_id=event.id, event_title=event.title, outfit=outfit, alternatives=alts, missing_slots=missing,
                          seated_minutes=event.seated_minutes, notes=notes)
