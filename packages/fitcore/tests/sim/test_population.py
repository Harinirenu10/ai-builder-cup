import numpy as np
import pytest

from fitcore.fit.config import get_config
from fitcore.population import BANNER_SMPL, BANNER_STANDIN, BODY_ZONES, ORDER, Population, banner, generate_standin, load_or_create
from fitcore.schemas import MeasurementZone as Z
from tests.sim.helpers import population


def column(pop: Population, name: str) -> np.ndarray:
    return pop.height_cm if name == "height" else pop.zones[Z(name)]


def test_two_thousand_bodies_per_gender_and_a_fixed_seed():
    a, b = generate_standin(), generate_standin()
    assert len(a) == 4000 and (a.gender == "female").sum() == 2000 and (a.gender == "male").sum() == 2000
    assert a.seed == get_config().population["seed"] and a.source == "stand-in"
    assert np.array_equal(a.height_cm, b.height_cm) and all(np.array_equal(a.zones[z], b.zones[z]) for z in BODY_ZONES)
    c = generate_standin(seed=1)
    assert not np.array_equal(a.height_cm, c.height_cm)


@pytest.mark.parametrize("gender", ["female", "male"])
def test_means_and_spreads_follow_the_config(gender):
    pop, cfg = population(), get_config().population["standin"][gender]
    sel = pop.gender == gender
    for name in ORDER:
        mean, sd = cfg[name]
        x = column(pop, name)[sel]
        assert abs(x.mean() - mean) < 0.12 * sd, (name, x.mean(), mean)
        assert 0.9 * sd < x.std() < 1.05 * sd, (name, x.std(), sd)           # clipping at 3 sd trims the tails a little


def test_bodies_stay_within_the_clip():
    cfg = get_config().population
    for g in ("female", "male"):
        sel = population().gender == g
        for name in ORDER:
            mean, sd = cfg["standin"][g][name]
            x = column(population(), name)[sel]
            assert x.min() >= mean - cfg["clip_sd"] * sd - 1e-9 and x.max() <= mean + cfg["clip_sd"] * sd + 1e-9


def test_measurements_move_together_like_bodies_do():
    sel = population().gender == "female"
    r = np.corrcoef(np.vstack([column(population(), n)[sel] for n in ORDER]))
    ix = {n: i for i, n in enumerate(ORDER)}
    assert r[ix["chest"], ix["waist"]] > 0.55 and r[ix["height"], ix["inseam"]] > 0.55 and r[ix["shoulder_width"], ix["upper_back"]] > 0.55
    assert abs(r[ix["height"], ix["waist"]]) < 0.45                           # height and girth are only loosely related
    assert np.linalg.eigvalsh(r).min() > 0


def test_every_body_is_a_valid_engine_input():
    m = population().body(7)
    assert set(m.zones) == set(BODY_ZONES) and all(v.value_cm > 0 for v in m.zones.values())


def test_the_banner_never_claims_smpl_for_a_stand_in():
    assert banner("stand-in") == BANNER_STANDIN and "not the SMPL shape space" in BANNER_STANDIN and "stand-in" in BANNER_STANDIN
    assert banner("smpl") == BANNER_SMPL and BANNER_SMPL.startswith("Synthetic population from the SMPL shape space. Calibrate with consented, anonymized twins")
    assert population().banner() == BANNER_STANDIN


# ----------------------------------------------------------------------------- parquet


def test_parquet_round_trip_keeps_everything(tmp_path):
    pop = population()
    path = tmp_path / "pop.parquet"
    pop.save(path)
    back = Population.load(path)
    assert back.source == "stand-in" and back.seed == pop.seed and len(back) == len(pop)
    assert np.array_equal(back.gender, pop.gender) and np.allclose(back.height_cm, pop.height_cm)
    assert all(np.allclose(back.zones[z], pop.zones[z]) for z in BODY_ZONES)


def test_load_or_create_caches_and_survives_a_bad_path(tmp_path):
    path = tmp_path / "cache" / "pop.parquet"
    first = load_or_create(path)
    assert path.is_file() and len(first) == 4000
    assert np.allclose(load_or_create(path).height_cm, first.height_cm)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so nothing can be created below it")
    assert len(load_or_create(blocker / "pop.parquet")) == 4000               # an unwritable cache never stops the simulation
    assert len(load_or_create(None)) == 4000


# ----------------------------------------------------------------------------- reweighting


def weighted_stats(pop: Population, gender: str) -> tuple[float, float]:
    sel = pop.gender == gender
    w = pop.weights[sel] / pop.weights[sel].sum()
    m = float(np.sum(w * pop.height_cm[sel]))
    return m, float(np.sqrt(np.sum(w * (pop.height_cm[sel] - m) ** 2)))


def test_reweighting_hits_the_target_height_distribution():
    target = get_config().target
    r = population().reweighted()
    assert abs(r.weights.sum() - 1) < 1e-9 and r.reweighted_to is not None
    for g in ("female", "male"):
        mean, sd = target["height_cm"][g]
        m, s = weighted_stats(r, g)
        assert abs(m - mean) < 0.4 and abs(s - sd) < 0.5, (g, m, s)


def test_reweighting_sets_the_gender_mix():
    r = population().reweighted(gender_mix={"female": 0.7, "male": 0.3})
    assert abs(r.weights[r.gender == "female"].sum() - 0.7) < 1e-9 and abs(r.weights[r.gender == "male"].sum() - 0.3) < 1e-9


def test_a_custom_target_and_the_weight_cap():
    target = {"female": [170.0, 5.0], "male": [180.0, 5.0]}
    means = {cap: weighted_stats(population().reweighted(height_cm=target, max_weight=cap), "female")[0] for cap in (1.0, 3.0, 1000.0)}
    assert means[1.0] < means[3.0] < means[1000.0]                            # a tighter cap limits how far the weights can pull toward the target
    assert means[1.0] >= weighted_stats(population(), "female")[0] - 0.5 and means[1000.0] > 166


def test_reweighting_leaves_the_bodies_alone_and_works_on_any_population():
    pop = population()
    r = pop.reweighted()
    assert r.zones is pop.zones and np.array_equal(r.height_cm, pop.height_cm)
    shifted = Population("smpl", 1, pop.gender, pop.height_cm + 8.0, pop.zones)       # a different source with taller bodies
    s = shifted.reweighted()
    assert weighted_stats(s, "female")[0] < weighted_stats(shifted, "female")[0] - 5   # pulled back toward 160
