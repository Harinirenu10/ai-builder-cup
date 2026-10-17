"""The vectorised simulation must give exactly the engine's answers, body by body."""

import numpy as np
import pytest

from fitcore.fit import evaluate
from fitcore.fit.config import get_config
from fitcore.schemas import Category, FitType, Verdict
from fitcore.sim import _Context, _evaluate
from tests.sim.helpers import make_product, population, subset

CODE = {Verdict.tight: -1, Verdict.short: -1, Verdict.ideal: 0, Verdict.loose: 1, Verdict.long: 1}

SCENARIOS = [
    (Category.shirt, FitType.regular, "garment", [{"fiber": "cotton", "percent": 100}], "none", None, 0.0),
    (Category.kurta, FitType.regular, "garment", [{"fiber": "rayon", "percent": 100}], "none", False, 0.0),
    (Category.kurta, FitType.slim, "body", [{"fiber": "rayon", "percent": 100}], "none", None, 0.0),
    (Category.tshirt, FitType.relaxed, "garment", [{"fiber": "cotton", "percent": 100}], "none", True, 3.0),
    (Category.tshirt, FitType.oversized, "garment", [{"fiber": "cotton", "percent": 100}], "none", None, -4.0),
    (Category.dress, FitType.slim, "body", [{"fiber": "cotton", "percent": 95}, {"fiber": "elastane", "percent": 5}], "high", None, 0.0),
    (Category.blouse, FitType.regular, "body", [{"fiber": "silk", "percent": 40}, {"fiber": "rayon", "percent": 60}], "low", None, 2.0),
    (Category.trousers, FitType.slim, "garment", [{"fiber": "cotton", "percent": 98}, {"fiber": "elastane", "percent": 2}], "low", None, 0.0),
    (Category.jeans, FitType.regular, "garment", [{"fiber": "cotton", "percent": 100}], "none", False, -3.0),
    (Category.jacket, FitType.regular, "garment", [{"fiber": "wool", "percent": 100}], "none", None, 0.0),
    (Category.jacket, FitType.oversized, "body", [{"fiber": "polyester", "percent": 100}], "none", True, 5.0),
]


@pytest.mark.parametrize("sc", SCENARIOS, ids=[f"{s[0].value}-{s[1].value}-{s[2]}" for s in SCENARIOS])
def test_vectorised_engine_matches_evaluate_for_every_body(sc):
    cat, fit, chart_type, fabric, stretch, pre, shift = sc
    product = make_product(cat, fit, chart_type, fabric, stretch, pre, shift)
    pop = subset(population(), 140, seed=len(cat.value))
    ev = _evaluate({s: dict(r) for s, r in product.size_chart.sizes.items()}, chart_type, _Context(product, pop))
    cfg = get_config()
    hard = set(cfg.hard_zones[cat])
    for i in range(len(pop)):
        res = evaluate(pop.body(i), product)
        expected_rec = -1 if res.recommended_size is None else ev.sizes.index(res.recommended_size)
        assert ev.rec[i] == expected_rec, (i, ev.rec[i], res.recommended_size)
        for si, size in enumerate(ev.sizes):
            zones = res.size_fit(size).zones
            for k, z in enumerate(ev.zones):
                if z not in zones:
                    assert not ev.present[si, k]
                    continue
                assert ev.present[si, k]
                assert ev.code_now[i, si, k] == CODE[zones[z].verdict], (i, size, z, zones[z].ease_now_cm)
                assert ev.code_wash[i, si, k] == CODE[zones[z].verdict_after_wash], (i, size, z, zones[z].ease_after_wash_cm)
            good = all(zf.verdict == Verdict.ideal and zf.verdict_after_wash == Verdict.ideal for z, zf in zones.items() if z in hard)
            assert bool(ev.good[i, si]) == good, (i, size)


def test_scenarios_actually_exercise_none_ties_and_every_size():
    """The comparison above would be vacuous if every body got the same answer."""
    seen_none = seen_sizes = 0
    for sc in SCENARIOS:
        product = make_product(*sc[:2], sc[2], sc[3], sc[4], sc[5], sc[6])
        ev = _evaluate({s: dict(r) for s, r in product.size_chart.sizes.items()}, sc[2], _Context(product, subset(population(), 400)))
        seen_none += int((ev.rec < 0).any())
        seen_sizes += len(set(ev.rec.tolist()) - {-1})
    assert seen_none >= 3 and seen_sizes >= 20


def test_a_zone_missing_from_a_size_matches_the_engine():
    product = make_product(Category.shirt)
    del product.size_chart.sizes["L"][list(product.size_chart.sizes["L"])[1]]       # drop a zone from one size
    pop = subset(population(), 100)
    ev = _evaluate({s: dict(r) for s, r in product.size_chart.sizes.items()}, "garment", _Context(product, pop))
    for i in range(len(pop)):
        res = evaluate(pop.body(i), product)
        assert ev.rec[i] == (-1 if res.recommended_size is None else ev.sizes.index(res.recommended_size))
