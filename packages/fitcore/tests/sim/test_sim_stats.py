import time

import numpy as np
import pytest

from fitcore.chart.grading import check_grading
from fitcore.fit import evaluate
from fitcore.fit.config import get_config
from fitcore.population import BODY_ZONES, Population
from fitcore.schemas import Category, MeasurementZone as Z, Product, Verdict
from fitcore.sim import simulate
from tests.sim.helpers import MEANS, make_product, population, subset


def toy(chests, weights=None) -> Population:
    """Bodies that differ only in chest; every other zone is at the mean."""
    n = len(chests)
    zones = {z: np.full(n, float(MEANS[z])) for z in BODY_ZONES}
    zones[Z.chest] = np.array(chests, float)
    return Population("stand-in", 0, np.array(["female"] * n), np.full(n, 165.0), zones,
                      np.array(weights, float) if weights is not None else None)


def oracle(product, pop):
    """Aggregate the real engine's answers body by body, with the population's weights."""
    cfg = get_config()
    hard = set(cfg.hard_zones[product.category])
    mix = {s: 0.0 for s in product.size_chart.sizes}
    none = good = 0.0
    zone_counts: dict[str, dict[Z, list[float]]] = {}
    for i in range(len(pop)):
        res, w = evaluate(pop.body(i), product), pop.weights[i]
        if any(all(not zf.hard or (zf.verdict == Verdict.ideal and zf.verdict_after_wash == Verdict.ideal) for zf in sf.zones.values()) for sf in res.sizes):
            good += w
        if res.recommended_size is None:
            none += w
            continue
        mix[res.recommended_size] += w
        for z, zf in res.size_fit(res.recommended_size).zones.items():
            c = zone_counts.setdefault(res.recommended_size, {}).setdefault(z, [0.0, 0.0, 0.0])
            ok = zf.verdict == Verdict.ideal and zf.verdict_after_wash == Verdict.ideal
            below = zf.verdict in (Verdict.tight, Verdict.short) or zf.verdict_after_wash in (Verdict.tight, Verdict.short)
            c[0 if below else 1 if ok else 2] += w
    return mix, none, good, zone_counts


def test_aggregation_matches_the_engine_with_weights():
    product = make_product(Category.shirt, shift=0.0)
    pop = toy([84, 90, 96, 102, 108, 116, 124], weights=[0.05, 0.1, 0.3, 0.25, 0.15, 0.1, 0.05])
    r = simulate(product, pop, regrade=False)
    mix, none, good, zc = oracle(product, pop)
    for s, share in mix.items():
        assert r.size_mix[s] == pytest.approx(share, abs=1e-3)
    assert r.no_recommendation_share == pytest.approx(none, abs=1e-3)
    assert r.coverage == pytest.approx(good, abs=1e-3) and r.no_good_size_share == pytest.approx(1 - good, abs=1e-3)
    for size, zones in zc.items():
        for z, (below, ideal, above) in zones.items():
            mass = below + ideal + above
            sh = r.zone_fit[size][z]
            assert sh.low == pytest.approx(below / mass, abs=1e-3) and sh.ideal == pytest.approx(ideal / mass, abs=1e-3) and sh.high == pytest.approx(above / mass, abs=1e-3)


def test_shares_are_valid_and_sum_to_one():
    r = simulate(make_product(Category.kurta, chart_type="body"), subset(population(), 800), regrade=False)
    assert sum(r.size_mix.values()) + r.no_recommendation_share == pytest.approx(1.0, abs=2e-3)
    assert 0 <= r.coverage <= 1 and r.coverage + r.no_good_size_share == pytest.approx(1.0, abs=1e-3)
    assert r.no_recommendation_share <= r.no_good_size_share + 1e-9           # no recommendation implies no good size
    for zs in r.zone_fit.values():
        for sh in zs.values():
            assert 0 <= sh.low <= 1 and 0 <= sh.high <= 1 and sh.low + sh.ideal + sh.high == pytest.approx(1.0, abs=2e-3)


def test_words_follow_the_zone_kind():
    r = simulate(make_product(Category.shirt), subset(population(), 300), regrade=False)
    assert r.zone_fit["M"][Z.chest].low_word == "tight" and r.zone_fit["M"][Z.chest].high_word == "loose"
    assert r.zone_fit["M"][Z.sleeve_length].low_word == "short" and r.zone_fit["M"][Z.garment_length].high_word == "long"


def test_a_chart_that_runs_small_covers_fewer_bodies():
    small = simulate(make_product(Category.shirt, shift=-12.0), subset(population(), 1000), regrade=False)
    right = simulate(make_product(Category.shirt, shift=0.0), subset(population(), 1000), regrade=False)
    assert small.coverage < right.coverage


def test_reweighting_to_the_target_market_changes_the_mix_and_says_so():
    product = make_product(Category.shirt)
    base = simulate(product, population(), regrade=False)
    tall = simulate(product, population().reweighted(height_cm={"female": [172.0, 6.0], "male": [186.0, 6.0]}, gender_mix={"female": 0.1, "male": 0.9}), regrade=False)
    assert not base.reweighted and tall.reweighted
    assert tall.size_mix != base.size_mix
    assert simulate(product, population(), regrade=False, target=True).reweighted


def test_the_result_names_the_population_and_carries_its_banner():
    r = simulate(make_product(Category.shirt), population(), regrade=False)
    assert r.population_source == "stand-in" and r.n_bodies == 4000 and "not the SMPL shape space" in r.banner and r.banner.endswith("before relying on it.")


# ----------------------------------------------------------------------------- regrade


def test_regrade_raises_coverage_within_two_cm_and_never_breaks_the_grading():
    product = make_product(Category.shirt, shift=-6.0)                        # the whole chart runs small
    r = simulate(product, subset(population(), 1500), regrade=True)
    rg = r.regrade
    assert rg is not None and rg.coverage_after > rg.coverage_before and rg.changes
    for c in rg.changes:
        assert abs(c.delta_cm) <= 2.0 + 1e-9 and c.to_cm == pytest.approx(r.rows[c.size][c.zone] + c.delta_cm, abs=0.011)
    assert not [f for f in check_grading(Category.shirt, rg.rows_after, tolerance_cm=0) if f.kind == "decrease"]
    assert rg.coverage_before == pytest.approx(r.coverage, abs=1e-3)
    assert set(rg.size_mix_after) == set(r.sizes) and sum(rg.size_mix_after.values()) <= 1.0 + 1e-6


def test_regrade_is_deterministic_and_never_lowers_coverage():
    product = make_product(Category.kurta)
    pop = subset(population(), 1000)
    a, b = simulate(product, pop), simulate(product, pop)
    assert [c.model_dump() for c in a.regrade.changes] == [c.model_dump() for c in b.regrade.changes]
    assert a.regrade.coverage_after >= a.regrade.coverage_before


def test_regrade_changes_nothing_when_every_body_is_already_covered():
    pop = toy([96] * 40)
    product = make_product(Category.tshirt, sizes=["XS", "M"], fabric=[{"fiber": "polyester", "percent": 100}])   # M sits at the ideal ease for the mean body
    r = simulate(product, pop, regrade=True)
    assert r.coverage == 1.0 and r.size_mix["M"] == 1.0, "the toy chart should cover bodies at the mean"
    assert r.regrade.changes == [] and r.regrade.coverage_after == r.regrade.coverage_before == 1.0


def test_regrade_off_gives_none():
    assert simulate(make_product(Category.shirt), subset(population(), 200), regrade=False).regrade is None


# ----------------------------------------------------------------------------- speed and errors


def test_a_full_simulation_with_regrade_on_the_whole_population_is_well_under_20_seconds():
    product = make_product(Category.kurta)
    started = time.perf_counter()
    r = simulate(product, population(), regrade=True)
    assert time.perf_counter() - started < 20 and r.n_bodies == 4000 and r.seconds < 20


def test_a_chart_with_no_zone_the_category_uses_is_an_error_not_a_made_up_answer():
    product = make_product(Category.tshirt)
    product.size_chart.sizes = {"M": {Z.neck: 40.0}}
    with pytest.raises(ValueError, match="No zones could be compared"):
        simulate(product, subset(population(), 50), regrade=False)


def test_a_product_without_a_chart_is_an_error():
    with pytest.raises(ValueError):
        simulate(Product(id="x", brand="b", title="t", category=Category.shirt))
