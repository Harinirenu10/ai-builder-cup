"""Deterministic fit engine: ease per size and zone, wash, pose strain, scoring, alterations."""

from fitcore.fit.alterations import suggest_alterations
from fitcore.fit.config import FitConfig, EaseRange, get_config
from fitcore.schemas import (
    Category,
    FitPreferences,
    FitResult,
    FitWarning,
    Measurements,
    MeasurementZone,
    PoseDeltas,
    PoseFit,
    Product,
    ReviewAdjustments,
    SizeFit,
    Verdict,
    ZoneFit,
)


def zone_label(zone: MeasurementZone) -> str:
    return zone.value.replace("_", " ")


def _fiber_key(name: str, cfg: FitConfig) -> str:
    key = name.strip().lower()
    key = cfg.fiber_aliases.get(key, key)
    return key if key in cfg.shrinkage_pct else "default"


def shrinkage_pct(product: Product, zone: MeasurementZone, cfg: FitConfig) -> float:
    """Percent a garment dimension shrinks in the wash: fiber-weighted, scaled down when pre-shrunk."""
    kind = "length" if zone in cfg.length_zones else "width"
    if product.fabric:
        total = sum(p.percent for p in product.fabric)
        pct = sum(getattr(cfg.shrinkage_pct[_fiber_key(p.fiber, cfg)], kind) * p.percent for p in product.fabric) / total
    else:
        pct = getattr(cfg.shrinkage_pct["default"], kind)
    return pct * (cfg.pre_shrunk_factor if product.pre_shrunk is True else 1.0)


def low_threshold(rng: EaseRange, allowance: float) -> float:
    return rng.min - allowance


def classify(ease: float, rng: EaseRange, zone: MeasurementZone, allowance: float, cfg: FitConfig) -> tuple[Verdict, float]:
    """Verdict and margin (cm outside the acceptable range; negative below, positive above, 0 inside)."""
    eps = cfg.scoring.verdict_epsilon_cm
    is_length = zone in cfg.length_zones
    low = low_threshold(rng, 0.0 if is_length else allowance)
    if ease < low - eps:
        return (Verdict.short if is_length else Verdict.tight), ease - low
    if ease > rng.max + eps:
        return (Verdict.long if is_length else Verdict.loose), ease - rng.max
    return Verdict.ideal, 0.0


def _penalty(margin: float, cfg: FitConfig) -> float:
    s = cfg.scoring
    if abs(margin) <= s.verdict_epsilon_cm:
        return 0.0
    return min(s.base_penalty + s.per_cm_penalty * abs(margin), s.penalty_cap)


def _weight(zone: MeasurementZone, hard: bool, cfg: FitConfig) -> float:
    if hard:
        return cfg.scoring.hard_weight
    return cfg.scoring.easy_weight if zone in cfg.easy_zones else cfg.scoring.other_weight


def evaluate(
    body: Measurements,
    product: Product,
    preferences: FitPreferences | None = None,
    review_adjustments: ReviewAdjustments | None = None,
    pose_deltas: PoseDeltas | None = None,
) -> FitResult:
    """Evaluate every size of the product's chart for this body."""
    cfg = get_config()
    if product.size_chart is None or not product.size_chart.sizes:
        raise ValueError("Product has no size chart")
    chart = product.size_chart
    cat: Category = product.category
    ease_table = cfg.target_ease_cm[cat][product.fit_type]
    hard_set = set(cfg.hard_zones[cat])
    allowance = cfg.stretch_allowance_cm[product.stretch_class]
    offsets = (preferences.offsets_cm.get(cat, {}) if preferences else {})
    reviews = review_adjustments.cm if review_adjustments else {}
    path = chart.chart_type

    warnings: list[FitWarning] = []
    missing_body: set[MeasurementZone] = set()
    sizes: list[SizeFit] = []
    body_conf: dict[MeasurementZone, float] = {}

    for size, row in chart.sizes.items():
        zones: dict[MeasurementZone, ZoneFit] = {}
        for zone in cfg.zones_by_category[cat]:
            if zone not in row or zone not in ease_table:
                continue
            body_zone = cfg.body_ref.get(zone, zone)
            body_cm = body.get_cm(body_zone)
            if body_cm is None:
                if zone in hard_set:
                    missing_body.add(body_zone)
                continue
            body_conf[zone] = body.zones[body_zone].confidence
            rng = ease_table[zone]
            tweak = offsets.get(zone, 0.0) - reviews.get(zone, 0.0)

            if path == "garment":
                garment_cm = row[zone]
                ease = garment_cm - body_cm + tweak
            else:
                # Chart is body dimensions: a body exactly on the chart value gets the ideal ease.
                garment_cm = row[zone] + rng.ideal
                ease = rng.ideal + (row[zone] - body_cm) + tweak

            shrink = garment_cm * shrinkage_pct(product, zone, cfg) / 100.0
            ease_wash = ease - shrink
            verdict, margin = classify(ease, rng, zone, allowance, cfg)
            verdict_w, margin_w = classify(ease_wash, rng, zone, allowance, cfg)

            poses: dict[str, PoseFit] = {}
            pose_penalty = 0.0
            if pose_deltas:
                low = low_threshold(rng, 0.0 if zone in cfg.length_zones else allowance)
                for pose, pd in pose_deltas.poses.items():
                    delta = pd.delta_cm.get(zone)
                    if delta is None:
                        continue
                    ease_pose = ease_wash - delta
                    strain = ease_pose < low - cfg.scoring.verdict_epsilon_cm
                    poses[pose] = PoseFit(ease_cm=round(ease_pose, 2), strain=strain, minutes=pd.minutes)
                    if strain:
                        factor = 1.0 if pd.minutes > cfg.scoring.long_pose_minutes else cfg.scoring.short_pose_factor
                        pose_penalty = max(pose_penalty, factor * _penalty(ease_pose - low, cfg))

            hard = zone in hard_set
            raw = max(_penalty(margin, cfg), _penalty(margin_w, cfg), pose_penalty)
            zones[zone] = ZoneFit(
                zone=zone,
                hard=hard,
                path=path,
                ease_now_cm=round(ease, 2),
                ease_after_wash_cm=round(ease_wash, 2),
                shrink_cm=round(shrink, 2),
                target_min_cm=rng.min,
                target_ideal_cm=rng.ideal,
                target_max_cm=rng.max,
                verdict=verdict,
                margin_cm=round(margin, 2),
                verdict_after_wash=verdict_w,
                margin_after_wash_cm=round(margin_w, 2),
                poses=poses,
                penalty=round(raw * _weight(zone, hard, cfg), 3),
            )
        total = sum(z.penalty for z in zones.values())
        sizes.append(SizeFit(
            size=size,
            zones=zones,
            penalty=round(total, 3),
            score=round(100.0 / (1.0 + total / cfg.scoring.score_halflife), 2),
        ))

    if not any(s.zones for s in sizes):
        raise ValueError("No zones could be compared: the size chart and the body share no measured zones")
    for zone in sorted(missing_body, key=lambda z: z.value):
        warnings.append(FitWarning(kind="data", zone=zone, message=f"No body measurement for {zone_label(zone)}, so that area was not checked."))

    result = FitResult(category=cat, fit_type=product.fit_type, sizes=sizes, warnings=warnings)
    warnings = result.warnings  # pydantic copies the list; keep appending to the model's own

    # A hard zone that fails in every size cannot be fixed by a tailor: no recommendation.
    hard_all_fail: list[tuple[MeasurementZone, str]] = []
    for zone in cfg.zones_by_category[cat]:
        if zone not in hard_set:
            continue
        fits = [s.zones[zone] for s in sizes if zone in s.zones]
        if len(fits) == len(sizes) and all(not _is_ok(f) for f in fits):
            hard_all_fail.append((zone, _direction(fits)))
    if hard_all_fail:
        for zone, direction in hard_all_fail:
            warnings.append(FitWarning(
                kind="size", zone=zone,
                message=f"No size in this chart fits the {zone_label(zone)} ({direction}). This area is hard to alter, so no size is recommended.",
            ))
        result.confidence = 0.0
        result.explanation_facts = [w.message for w in warnings if w.kind == "size"]
        return result

    # Best score wins; ties go to the earlier (smaller) size.
    best = max(range(len(sizes)), key=lambda i: (sizes[i].score, -i))
    rec = sizes[best]
    result.recommended_size = rec.size

    others = [s.score for i, s in enumerate(sizes) if i != best]
    gap_conf = 0.5 if not others else min(1.0, (rec.score - max(others)) / cfg.scoring.confidence_gap_full)
    meas_conf = (sum(body_conf[z] for z in rec.zones) / len(rec.zones)) if rec.zones else 0.0
    result.confidence = round(max(0.0, min(1.0, (0.5 + 0.5 * gap_conf) * meas_conf)), 3)

    alterations, alt_warnings = suggest_alterations(rec, cat, cfg)
    result.alterations = alterations
    warnings.extend(alt_warnings)
    warnings.extend(_wash_pose_warnings(rec, product, cfg))
    result.explanation_facts = build_facts(result, rec, product, cfg)
    return result


def _is_ok(z: ZoneFit) -> bool:
    return z.verdict == Verdict.ideal and z.verdict_after_wash == Verdict.ideal


def _direction(fits: list[ZoneFit]) -> str:
    worst = [f.verdict if f.verdict != Verdict.ideal else f.verdict_after_wash for f in fits]
    if all(v in (Verdict.tight, Verdict.short) for v in worst):
        return "too snug in every size"
    if all(v in (Verdict.loose, Verdict.long) for v in worst):
        return "too roomy in every size"
    return "between sizes"


def _wash_pose_warnings(rec: SizeFit, product: Product, cfg: FitConfig) -> list[FitWarning]:
    out: list[FitWarning] = []
    for zone, z in rec.zones.items():
        if z.verdict_after_wash != z.verdict:
            note = " Pre-shrunk status is unknown and was assumed not pre-shrunk." if product.pre_shrunk is None else ""
            out.append(FitWarning(
                kind="wash", zone=zone,
                message=f"After washing, the {zone_label(zone)} is expected to shrink about {z.shrink_cm:.1f} cm and read {z.verdict_after_wash.value}.{note}",
            ))
        for pose, p in z.poses.items():
            if p.strain:
                held = f" for {p.minutes:g} min" if p.minutes else ""
                out.append(FitWarning(
                    kind="pose", zone=zone,
                    message=f"When {pose}{held}, the {zone_label(zone)} has {p.ease_cm:.1f} cm of ease, below the {z.target_min_cm:.1f} cm minimum for this garment.",
                ))
    return out


def _fmt(v: float) -> str:
    return f"{v:+.1f}"


def build_facts(result: FitResult, rec: SizeFit, product: Product, cfg: FitConfig) -> list[str]:
    """Short factual statements about the garment; the only source for explain()."""
    facts = [f"Size {rec.size} has the best match across {len(rec.zones)} checked areas of this {product.fit_type.value}-fit {product.category.value}."]
    for zone, z in rec.zones.items():
        label = zone_label(zone).capitalize()
        rng = f"{z.target_min_cm:.1f} to {z.target_max_cm:.1f} cm"
        if z.verdict == Verdict.ideal:
            if z.hard:
                facts.append(f"{label} ease is {_fmt(z.ease_now_cm)} cm, inside the {product.fit_type.value}-fit range of {rng}.")
        else:
            facts.append(f"{label} ease is {_fmt(z.ease_now_cm)} cm, which is {abs(z.margin_cm):.1f} cm {'under' if z.margin_cm < 0 else 'over'} the {product.fit_type.value}-fit range of {rng}.")
    for w in result.warnings:
        if w.kind in ("wash", "pose"):
            facts.append(w.message)
    for a in result.alterations:
        facts.append(f"Suggested alteration: {a.text.lower()}.")
    for w in result.warnings:
        if w.kind == "data":
            facts.append(w.message)
    return facts
