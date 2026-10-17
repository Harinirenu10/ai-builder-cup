"""Charts and populations for the simulation tests."""

import numpy as np

from fitcore.fit.config import get_config
from fitcore.population import BODY_ZONES, Population, generate_standin
from fitcore.schemas import Category, FitType, MeasurementZone as Z, Product, SizeChart

SIZES = ["S", "M", "L", "XL", "XXL"]
MEANS = {Z.chest: 96, Z.waist: 82, Z.hip: 99, Z.shoulder_width: 41, Z.sleeve_length: 59, Z.torso_length: 42, Z.thigh: 56, Z.inseam: 78,
         Z.neck: 36, Z.back_rise: 27, Z.upper_back: 38}
STEPS = {Z.chest: 5.5, Z.waist: 5.5, Z.hip: 5.0, Z.shoulder_width: 1.4, Z.sleeve_length: 1.2, Z.garment_length: 2.0, Z.thigh: 2.5, Z.inseam: 0.5,
         Z.back_rise: 0.8, Z.neck: 1.0, Z.upper_back: 1.2}


def subset(pop: Population, n: int, seed: int = 0) -> Population:
    idx = np.random.default_rng(seed).choice(len(pop), n, replace=False)
    return Population(pop.source, pop.seed, pop.gender[idx], pop.height_cm[idx], {z: pop.zones[z][idx] for z in BODY_ZONES})


def make_product(category: Category, fit: FitType = FitType.regular, chart_type: str = "garment", fabric=None, stretch: str = "none",
                 pre_shrunk=None, shift: float = 0.0, sizes: list[str] | None = None) -> Product:
    cfg = get_config()
    ease = cfg.target_ease_cm[category][fit]
    sizes = sizes or SIZES
    rows: dict[str, dict[str, float]] = {}
    for i, label in enumerate(sizes):
        row = {}
        for z in cfg.zones_by_category[category]:
            if z not in ease:
                continue
            body = MEANS[cfg.body_ref.get(z, z)]
            step = STEPS[z]
            base = body + (ease[z].ideal if chart_type == "garment" else 0.0)
            row[z.value] = round(base + (i - 1) * step + shift, 1)
        rows[label] = row
    return Product(id="t", brand="b", title="t", category=category, fit_type=fit, fabric=fabric if fabric is not None else [{"fiber": "cotton", "percent": 100}],
                   stretch_class=stretch, pre_shrunk=pre_shrunk,
                   size_chart=SizeChart(sizes={s: {Z(k): v for k, v in r.items()} for s, r in rows.items()}, chart_type=chart_type))


_POP = None


def population() -> Population:
    global _POP
    if _POP is None:
        _POP = generate_standin()
    return _POP
