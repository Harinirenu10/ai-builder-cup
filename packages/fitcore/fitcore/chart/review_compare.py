"""Put what reviewers say beside what the seller just measured."""

from statistics import median
from typing import Literal

from pydantic import BaseModel

from fitcore.schemas import MeasurementZone as Z, ReviewSignals, SizeChart

SAME_WITHIN_CM = 0.5
PHRASES = {"runs_small": "runs small at the {z}", "tight": "feels tight at the {z}", "short": "is short at the {z}",
           "runs_large": "runs large at the {z}", "loose": "feels loose at the {z}", "long": "is long at the {z}"}
SMALLER = {"runs_small", "tight", "short"}


class ReviewNote(BaseModel):
    zone: Z
    direction: str
    mentions: int
    old_cm: float | None
    new_cm: float | None
    delta_cm: float | None
    agreement: Literal["agrees", "differs", "same", "no_comparison"]
    text: str


def compare_reviews(new: dict[str, dict[Z, float]], old: SizeChart | None, reviews: ReviewSignals | None) -> list[ReviewNote]:
    """One note per review signal. `delta` is the median over sizes of (new measured - old chart value), when the old chart is a garment chart."""
    if not reviews or not reviews.zone_signals:
        return []
    notes: list[ReviewNote] = []
    for s in sorted(reviews.zone_signals, key=lambda s: (-s.mention_count, s.zone.value)):
        zone = s.zone.value.replace("_", " ")
        who = "A reviewer says" if s.mention_count == 1 else f"{s.mention_count} reviewers say"
        said = f"{who} it {PHRASES[s.direction].format(z=zone)}"
        deltas, olds, news = [], [], []
        if old is not None and old.chart_type == "garment":
            for size, row in new.items():
                if size in old.sizes and s.zone in row and s.zone in old.sizes[size]:
                    deltas.append(row[s.zone] - old.sizes[size][s.zone])
                    olds.append(old.sizes[size][s.zone])
                    news.append(row[s.zone])
        if not deltas:
            why = "your old chart lists body sizes, so it cannot be compared" if old is not None and old.chart_type == "body" else "there is no earlier garment chart to compare with"
            notes.append(ReviewNote(zone=s.zone, direction=s.direction, mentions=s.mention_count, old_cm=None, new_cm=None, delta_cm=None,
                                    agreement="no_comparison", text=f"{said}; {why}."))
            continue
        d = round(median(deltas), 1)
        if abs(d) < SAME_WITHIN_CM:
            agreement, tail = "same", f"your measured {zone} matches your old chart within {SAME_WITHIN_CM:g} cm."
        else:
            agreement = "agrees" if (s.direction in SMALLER) == (d < 0) else "differs"
            tail = f"your measured {zone} is {abs(d):.1f} cm {'below' if d < 0 else 'above'} your old chart."
        notes.append(ReviewNote(zone=s.zone, direction=s.direction, mentions=s.mention_count, old_cm=round(median(olds), 1), new_cm=round(median(news), 1),
                                delta_cm=d, agreement=agreement, text=f"{said}; {tail}"))  # type: ignore[arg-type]
    return notes
