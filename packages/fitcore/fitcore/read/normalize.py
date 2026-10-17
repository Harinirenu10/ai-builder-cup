"""Post-processing for page extraction, in Python rather than in the prompt.

Units to cm, zone names to MeasurementZone, flat-to-full circumference, chart-type check, monotonic flag,
fabric percent clean-up, and clamping of review counts to what was actually read.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from fitcore.read.models import ExtractedProduct, ExtractedReviews, PageExtraction, RawSize
from fitcore.schemas import (
    FabricPart,
    MeasurementZone as Z,
    Product,
    ReviewerContext,
    ReviewSignals,
    SizeChart,
    ZoneSignal,
    Category,
)

INCH_TO_CM = 2.54
CIRCUMFERENCE = {Z.chest, Z.waist, Z.hip, Z.thigh}
BOTTOMS = {Category.trousers, Category.jeans}


def to_cm(value: float, unit: str) -> float:
    return round(value * INCH_TO_CM, 2) if unit == "in" else round(value, 2)


# ----------------------------------------------------------------------------- zones

# First match wins, so specific patterns come first.
_ZONE_PATTERNS: list[tuple[str, Z]] = [
    (r"under\s*bust|front rise|waist to|^length to|leg opening|bottom opening|hem width", None),  # type: ignore[list-item]
    (r"back rise|^rise\b|crotch", Z.back_rise),
    (r"inseam|inside leg|inner leg", Z.inseam),
    (r"across back|back width|upper back", Z.upper_back),
    (r"shoulder", Z.shoulder_width),
    (r"sleeve|arm length", Z.sleeve_length),
    (r"thigh", Z.thigh),
    (r"neck|collar", Z.neck),
    (r"\bhips?\b|seat", Z.hip),
    (r"waist", Z.waist),
    (r"bust|chest|pit to pit", Z.chest),
    (r"torso", Z.torso_length),
    (r"length|kurta length|top length|body length|dress length", Z.garment_length),
]


def normalize_zone(label: str, category: Category | None = None) -> Z | None:
    """Map a printed measurement name to a MeasurementZone, or None when it has no zone."""
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", label.lower())
    text = re.sub(r"\b(to fit|fits?|body|garment|flat|approx\.?|measurement|size|in|cm|inches|inch)\b", " ", text)
    text = re.sub(r"[^a-z ]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    # "body length" / "garment length" were stripped above, so restore the intent for plain "length"
    for pattern, zone in _ZONE_PATTERNS:
        if re.search(pattern, text):
            if zone is None:
                return None
            if zone == Z.garment_length and category in BOTTOMS:
                return None  # a bare 'length' on trousers is outseam; the engine uses inseam
            return zone
    return None


# ----------------------------------------------------------------------------- sizes

_LETTER_RANK = {"XXS": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "2XL": 6, "XXXL": 7, "3XL": 7,
                "4XL": 8, "XXXXL": 8, "5XL": 9, "6XL": 10}


def size_rank(label: str) -> float | None:
    key = re.sub(r"[^A-Z0-9]", "", label.upper())
    if key in _LETTER_RANK:
        return float(_LETTER_RANK[key])
    m = re.match(r"^\d+(\.\d+)?", label.strip())
    return float(m.group(0)) if m else None


def order_sizes(sizes: list[RawSize]) -> list[RawSize]:
    """Smallest first when every label can be ranked; otherwise the page order."""
    ranks = [size_rank(s.label) for s in sizes]
    if sizes and all(r is not None for r in ranks):
        return [s for _, s in sorted(zip(ranks, sizes), key=lambda t: t[0])]  # type: ignore[arg-type]
    return sizes


def non_monotonic_zones(rows: dict[str, dict[Z, float]]) -> list[Z]:
    """Zones whose value goes down from one size to the next larger one (equal is fine)."""
    zones = sorted({z for r in rows.values() for z in r}, key=lambda z: z.value)
    bad = []
    for zone in zones:
        vals = [r[zone] for r in rows.values() if zone in r]
        if any(b < a - 1e-9 for a, b in zip(vals, vals[1:])):
            bad.append(zone)
    return bad


def infer_unit(values_by_zone: dict[Z, list[float]]) -> Literal["cm", "in"]:
    """Chest/waist/hip values under 70 are inches (garments run 28-60 in, 70-150 cm)."""
    circ = [v for z, vs in values_by_zone.items() if z in CIRCUMFERENCE for v in vs]
    if not circ:
        return "cm"
    circ.sort()
    return "in" if circ[len(circ) // 2] < 70 else "cm"


# ----------------------------------------------------------------------------- chart type

_BODY = re.compile(r"to fit|body|fits? (chest|bust|waist|hip)|your (chest|bust|waist|hip)|(chest|bust|waist|hip) size", re.I)
_GARMENT = re.compile(r"garment|flat|product measurement|item measurement|actual|laid", re.I)


def resolve_chart_type(model_type: str, evidence: str | None) -> tuple[Literal["body", "garment"], bool]:
    """Chart type and whether it was a guess. The label text wins over the model when it is clear."""
    ev = evidence or ""
    body, garment = bool(_BODY.search(ev)), bool(_GARMENT.search(ev))
    if body and not garment:
        return "body", False
    if garment and not body:
        return "garment", False
    if model_type in ("body", "garment"):
        return model_type, False  # type: ignore[return-value]
    return "body", True


# ----------------------------------------------------------------------------- assembling


@dataclass
class ReadOutcome:
    status: Literal["ok", "unsupported"]
    product: Product | None
    reviews: ReviewSignals | None
    flags: list[str] = field(default_factory=list)


def _fabric(ex: ExtractedProduct, flags: list[str]) -> list[FabricPart]:
    parts = [(p.fiber.strip().lower(), p.percent) for p in ex.fabric if p.fiber.strip() and p.percent > 0]
    if not parts:
        return []
    total = sum(p for _, p in parts)
    if abs(total - 100) > 1.0:
        flags.append(f"fabric_percent_sum:{total:g}")
        if total > 0:
            parts = [(f, p * 100.0 / total) for f, p in parts]
    return [FabricPart(fiber=f, percent=round(min(p, 100.0), 1)) for f, p in parts]


def build_chart(ex: ExtractedProduct, flags: list[str]) -> SizeChart | None:
    if not ex.sizes:
        return None
    sizes = order_sizes(ex.sizes)
    raw: dict[str, dict[Z, float]] = {}
    for s in sizes:
        row: dict[Z, float] = {}
        for m in s.measurements:
            zone = normalize_zone(m.label, ex.category)
            if zone is None:
                flags.append(f"unmapped_zone:{m.label.strip()}")
                continue
            hi = m.value_max if m.value_max is not None else m.value_min
            row[zone] = (m.value_min + hi) / 2.0
        if row:
            raw[s.label.strip()] = row
    if not raw:
        return None

    unit = ex.unit
    if unit == "unknown":
        by_zone: dict[Z, list[float]] = {}
        for row in raw.values():
            for z, v in row.items():
                by_zone.setdefault(z, []).append(v)
        unit = infer_unit(by_zone)
        flags.append("unit_inferred")

    chart_type, guessed = resolve_chart_type(ex.chart_type, ex.chart_type_evidence)
    if guessed:
        flags.append("chart_type_guessed")
    flat = ex.circumference_basis == "flat" and chart_type == "garment"
    if ex.circumference_basis == "flat" and chart_type == "body":
        flags.append("flat_basis_ignored_for_body_chart")

    sizes_cm: dict[str, dict[Z, float]] = {}
    for label, row in raw.items():
        sizes_cm[label] = {z: to_cm(v * (2 if flat and z in CIRCUMFERENCE else 1), unit) for z, v in row.items()}

    chart_flags = [f"non_monotonic:{z.value}" for z in non_monotonic_zones(sizes_cm)]
    system = ex.size_system or "letter"
    return SizeChart(
        sizes=sizes_cm,
        chart_type=chart_type,
        size_system=system,
        unit="cm",
        original_unit=unit,  # type: ignore[arg-type]
        type_evidence=ex.chart_type_evidence,
        flags=chart_flags + [f for f in flags if f in ("unit_inferred", "chart_type_guessed")],
    )


def build_reviews(raw: ExtractedReviews) -> ReviewSignals:
    """Counts must come from reviews actually read: nothing read means nothing returned; counts are capped."""
    n = max(0, raw.reviews_read)
    if n == 0:
        return ReviewSignals()
    merged: dict[tuple[Z, str], ZoneSignal] = {}
    for sig in raw.zone_signals:
        count = min(max(sig.mention_count, 0), n)
        if count == 0:
            continue
        ctx = [ReviewerContext(**c.model_dump()) for c in sig.reviewer_context][:5]
        key = (sig.zone, sig.direction)
        strength = min(1.0, max(0.0, sig.strength))
        if key in merged:
            prev = merged[key]
            total = min(prev.mention_count + count, n)
            merged[key] = ZoneSignal(zone=sig.zone, direction=sig.direction, mention_count=total,
                                     strength=round((prev.strength * prev.mention_count + strength * count) / (prev.mention_count + count), 3),
                                     reviewer_context=(prev.reviewer_context + ctx)[:5])
        else:
            merged[key] = ZoneSignal(zone=sig.zone, direction=sig.direction, strength=round(strength, 3), mention_count=count, reviewer_context=ctx)
    cap = lambda v: min(max(v, 0), n)  # noqa: E731
    return ReviewSignals(
        reviews_read=n,
        zone_signals=sorted(merged.values(), key=lambda s: (-s.mention_count, s.zone.value)),
        shrinkage_mentions=cap(raw.shrinkage_mentions),
        color_mismatch_mentions=cap(raw.color_mismatch_mentions),
        sheerness_mentions=cap(raw.sheerness_mentions),
    )


def assemble(ext: PageExtraction, *, url: str | None, product_id: str, fallback_brand: str = "unknown",
             image_urls: list[str] | None = None, with_reviews: bool = True) -> ReadOutcome:
    ex = ext.product
    flags: list[str] = []
    if ex.category is None:
        return ReadOutcome(status="unsupported", product=None, reviews=None, flags=["category_unsupported"])
    chart = build_chart(ex, flags)
    if chart is None:
        flags.append("no_size_chart")
    pre = {"true": True, "false": False}.get(ex.pre_shrunk)
    images = list(dict.fromkeys([*(ex.image_urls or []), *(image_urls or [])]))[:10]
    product = Product(
        id=product_id,
        brand=(ex.brand or fallback_brand).strip() or fallback_brand,
        title=ex.title.strip(),
        category=ex.category,
        fit_type=ex.fit_type or "regular",  # type: ignore[arg-type]
        fabric=_fabric(ex, flags),
        stretch_class=ex.stretch_class or "none",  # type: ignore[arg-type]
        pre_shrunk=pre,
        pre_shrunk_evidence=(ex.pre_shrunk_evidence or None) if pre is not None else None,
        size_chart=chart,
        source_url=url,
        image_urls=images,
    )
    if ex.fit_type is None:
        flags.append("fit_type_defaulted")
    if ex.stretch_class is None:
        flags.append("stretch_class_defaulted")
    reviews = build_reviews(ext.reviews) if with_reviews else None
    return ReadOutcome(status="ok", product=product, reviews=reviews, flags=list(dict.fromkeys(flags)))
