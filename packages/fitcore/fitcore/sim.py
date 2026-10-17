"""Pre-production return simulation: run a size chart against a population of bodies.

    python -m fitcore.sim samples/product_kurta.json [--no-regrade] [--target]

For every body the fit engine's verdicts decide the best size. The engine is run for each body in a vectorised form (NumPy, the
same formulas, rounding and tie-breaking as `fitcore.fit.engine.evaluate`; tests/sim/test_sim_equivalence.py checks that they agree
on random charts and bodies), because the regrade search evaluates the whole population hundreds of times.

Output: the size mix, the share of bodies below / inside / above each zone's range per size, the share with no good size, and
regrade suggestions from a search of +-2 cm per zone per size that maximises coverage (bodies with a good size).
A "good size" is one where every hard-to-alter zone is ideal, both as bought and after the first wash.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from fitcore.chart.grading import GradingFlag, check_grading
from fitcore.fit.config import get_config
from fitcore.fit.engine import shrinkage_pct
from fitcore.population import Population, generate_standin
from fitcore.schemas import Category, FitType, MeasurementZone as Z, Product

UNITS = {"low": {"tight", "short"}, "high": {"loose", "long"}}


# ----------------------------------------------------------------------------- result models


class ZoneShare(BaseModel):
    low: float                  # share below the zone's range (tight, or short for lengths), as bought or after a wash
    ideal: float
    high: float                 # share above the range (loose, or long)
    low_word: str
    high_word: str


class RegradeChange(BaseModel):
    size: str
    zone: Z
    from_cm: float
    to_cm: float
    delta_cm: float
    coverage_gain_alone: float  # coverage gained if only this change were made


class Regrade(BaseModel):
    changes: list[RegradeChange]
    coverage_before: float
    coverage_after: float
    size_mix_after: dict[str, float]
    rows_after: dict[str, dict[Z, float]]
    grading_flags_after: list[GradingFlag] = Field(default_factory=list)


class SimResult(BaseModel):
    n_bodies: int
    population_source: str
    population_seed: int
    reweighted: bool
    banner: str
    category: Category
    fit_type: FitType
    chart_type: str
    sizes: list[str]
    rows: dict[str, dict[Z, float]]
    size_mix: dict[str, float]                      # share of bodies whose best size is this size
    no_recommendation_share: float                  # the engine gave no size: a hard zone fails in every size
    no_good_size_share: float                       # no size has every hard zone ideal (as bought and after washing)
    coverage: float                                 # 1 - no_good_size_share
    recommended_counts: dict[str, float]            # share of bodies per size (same as size_mix), kept for denominators in zone_fit
    zone_fit: dict[str, dict[Z, ZoneShare]]         # among the bodies whose best size is this one
    regrade: Regrade | None = None
    seconds: float = 0.0


# ----------------------------------------------------------------------------- the vectorised engine


@dataclass
class _Evaluated:
    sizes: list[str]
    zones: list[Z]
    present: np.ndarray            # (S, Z) bool
    code_now: np.ndarray           # (N, S, Z) -1 below, 0 ideal, +1 above
    code_wash: np.ndarray
    rec: np.ndarray                # (N,) index of the best size, -1 for none
    good: np.ndarray               # (N, S) every hard zone ideal
    hard: np.ndarray               # (Z,) bool


class _Context:
    """Everything about one product and population that does not change while the chart does."""

    def __init__(self, product: Product, pop: Population):
        self.cfg = get_config()
        self.product = product
        self.cat = product.category
        self.ease_table = self.cfg.target_ease_cm[self.cat][product.fit_type]
        self.hard_set = set(self.cfg.hard_zones[self.cat])
        self.allowance = self.cfg.stretch_allowance_cm[product.stretch_class]
        self.zones = [z for z in self.cfg.zones_by_category[self.cat] if z in self.ease_table]
        self.shrink = {z: shrinkage_pct(product, z, self.cfg) / 100.0 for z in self.zones}
        self.body = {z: pop.zones[self.cfg.body_ref.get(z, z)] for z in self.zones if self.cfg.body_ref.get(z, z) in pop.zones}
        self.weights = pop.weights
        self.n = len(pop)


def _penalty(margin: np.ndarray, s: Any) -> np.ndarray:
    a = np.abs(margin)
    return np.where(a <= s.verdict_epsilon_cm, 0.0, np.minimum(s.base_penalty + s.per_cm_penalty * a, s.penalty_cap))


def _evaluate(rows: dict[str, dict[Z, float]], chart_type: str, ctx: _Context) -> _Evaluated:
    cfg, s = ctx.cfg, ctx.cfg.scoring
    sizes = list(rows)
    zones = [z for z in ctx.zones if z in ctx.body]
    n, S, K = ctx.n, len(sizes), len(zones)
    present = np.zeros((S, K), bool)
    code_now = np.zeros((n, S, K), np.int8)
    code_wash = np.zeros((n, S, K), np.int8)
    zone_pen = np.zeros((n, S, K))
    hard = np.array([z in ctx.hard_set for z in zones])
    for si, size in enumerate(sizes):
        row = rows[size]
        for k, z in enumerate(zones):
            if z not in row:
                continue
            present[si, k] = True
            rng = ctx.ease_table[z]
            body = ctx.body[z]
            if chart_type == "garment":
                garment = row[z]
                ease = garment - body
            else:
                garment = row[z] + rng.ideal
                ease = rng.ideal + (row[z] - body)
            wash = ease - garment * ctx.shrink[z]
            low = rng.min - (0.0 if z in cfg.length_zones else ctx.allowance)
            eps = s.verdict_epsilon_cm
            cn = np.where(ease < low - eps, -1, np.where(ease > rng.max + eps, 1, 0))
            cw = np.where(wash < low - eps, -1, np.where(wash > rng.max + eps, 1, 0))
            mn = np.where(cn == -1, ease - low, np.where(cn == 1, ease - rng.max, 0.0))
            mw = np.where(cw == -1, wash - low, np.where(cw == 1, wash - rng.max, 0.0))
            raw = np.maximum(_penalty(mn, s), _penalty(mw, s))
            weight = s.hard_weight if hard[k] else (s.easy_weight if z in cfg.easy_zones else s.other_weight)
            zone_pen[:, si, k] = np.round(raw * weight, 3)
            code_now[:, si, k], code_wash[:, si, k] = cn, cw
    total = zone_pen.sum(axis=2)
    score = np.round(100.0 / (1.0 + total / s.score_halflife), 2)
    rec = np.argmax(score, axis=1)                      # first maximum: ties go to the smaller size, as in the engine
    ok = (code_now == 0) & (code_wash == 0)
    none = np.zeros(n, bool)
    for k in np.nonzero(hard)[0]:
        if present[:, k].all():
            none |= ~ok[:, :, k].any(axis=1)             # a hard zone that fails in every size: no recommendation
    rec = np.where(none, -1, rec)
    good = np.ones((n, S), bool)
    for k in np.nonzero(hard)[0]:
        good &= ok[:, :, k] | ~present[:, k][None, :]
    good &= present.any(axis=1)[None, :]
    return _Evaluated(sizes, zones, present, code_now, code_wash, rec, good, hard)


def _coverage(ev: _Evaluated, w: np.ndarray) -> float:
    return float(np.sum(w * ev.good.any(axis=1)))


def _objective(ev: _Evaluated, w: np.ndarray) -> float:
    """Coverage, with a small credit for hard zones that are ideal at each body's best size, to give the search a slope on plateaus."""
    n = len(w)
    hard_idx = np.nonzero(ev.hard)[0]
    if len(hard_idx) == 0:
        return _coverage(ev, w)
    rec = np.where(ev.rec < 0, 0, ev.rec)
    ok = (ev.code_now == 0) & (ev.code_wash == 0)
    at_rec = ok[np.arange(n), rec][:, hard_idx]
    pres = ev.present[rec][:, hard_idx]
    frac = np.where(pres.sum(axis=1) > 0, (at_rec & pres).sum(axis=1) / np.maximum(pres.sum(axis=1), 1), 0.0)
    return _coverage(ev, w) + 0.01 * float(np.sum(w * frac))


def _mix(ev: _Evaluated, w: np.ndarray) -> dict[str, float]:
    return {size: round(float(np.sum(w * (ev.rec == i))), 4) for i, size in enumerate(ev.sizes)}


# ----------------------------------------------------------------------------- regrade search


def _grading_decreases(cat: Category, rows: dict[str, dict[Z, float]]) -> int:
    return sum(f.kind == "decrease" for f in check_grading(cat, rows))


def regrade_search(rows: dict[str, dict[Z, float]], chart_type: str, ctx: _Context, grid_cm: float, step_cm: float, passes: int, min_gain: float) -> Regrade:
    """Coordinate search: for each size and zone try moving it up to grid_cm either way (in step_cm steps) and keep what raises the
    objective. A change that would make a size smaller than the next smaller size is never allowed."""
    w = ctx.weights
    original = {s: dict(r) for s, r in rows.items()}
    current = {s: dict(r) for s, r in rows.items()}
    base_ev = _evaluate(original, chart_type, ctx)
    best = _objective(base_ev, w)
    baseline_decreases = _grading_decreases(ctx.cat, original)
    k_max = int(round(grid_cm / step_cm))
    for _ in range(passes):
        improved = False
        for size in list(current):
            for z in sorted(current[size], key=lambda z: z.value):
                if z not in ctx.body:
                    continue
                keep = current[size][z]
                top, top_val = best, keep
                for k in range(-k_max, k_max + 1):
                    cand = round(original[size][z] + k * step_cm, 2)
                    if cand == keep or cand <= 0:
                        continue
                    current[size][z] = cand
                    if _grading_decreases(ctx.cat, current) > baseline_decreases:
                        continue
                    val = _objective(_evaluate(current, chart_type, ctx), w)
                    if val > top + min_gain:
                        top, top_val = val, cand
                current[size][z] = top_val
                if top_val != keep:
                    best, improved = top, True
        if not improved:
            break
    final_ev = _evaluate(current, chart_type, ctx)
    base_cov = _coverage(base_ev, w)
    changes = []
    for size in current:
        for z, v in current[size].items():
            if abs(v - original[size][z]) > 1e-9:
                alone = {s: dict(r) for s, r in original.items()}
                alone[size][z] = v
                gain = _coverage(_evaluate(alone, chart_type, ctx), w) - base_cov
                changes.append(RegradeChange(size=size, zone=z, from_cm=original[size][z], to_cm=v, delta_cm=round(v - original[size][z], 2),
                                             coverage_gain_alone=round(gain, 4)))
    changes.sort(key=lambda c: (-c.coverage_gain_alone, c.size, c.zone.value))
    return Regrade(changes=changes, coverage_before=round(base_cov, 4), coverage_after=round(_coverage(final_ev, w), 4), size_mix_after=_mix(final_ev, w),
                   rows_after=current, grading_flags_after=check_grading(ctx.cat, current))


# ----------------------------------------------------------------------------- the simulation


def simulate(product: Product, pop: Population | None = None, regrade: bool = True, target: bool = False) -> SimResult:
    """Run the product's size chart against the population. `target` reweights the population to the configured market first."""
    started = time.perf_counter()
    if product.size_chart is None or not product.size_chart.sizes:
        raise ValueError("The product has no size chart")
    pop = pop or generate_standin()
    if target:
        pop = pop.reweighted()
    cfg = get_config()
    ctx = _Context(product, pop)
    chart = product.size_chart
    rows = {size: {z: v for z, v in row.items()} for size, row in chart.sizes.items()}
    ev = _evaluate(rows, chart.chart_type, ctx)
    if not ev.present.any():
        raise ValueError("No zones could be compared: the size chart and this category share no measured zones")   # as the engine does
    w = pop.weights
    coverage = _coverage(ev, w)
    mix = _mix(ev, w)
    zone_fit: dict[str, dict[Z, ZoneShare]] = {}
    for si, size in enumerate(ev.sizes):
        chosen = (ev.rec == si)
        mass = float(np.sum(w * chosen))
        zone_fit[size] = {}
        for k, z in enumerate(ev.zones):
            if not ev.present[si, k]:
                continue
            below = (ev.code_now[:, si, k] < 0) | (ev.code_wash[:, si, k] < 0)
            above = ((ev.code_now[:, si, k] > 0) | (ev.code_wash[:, si, k] > 0)) & ~below
            lo, hi = (("short", "long") if z in cfg.length_zones else ("tight", "loose"))
            if mass <= 0:
                zone_fit[size][z] = ZoneShare(low=0.0, ideal=0.0, high=0.0, low_word=lo, high_word=hi)
                continue
            a, b = float(np.sum(w * chosen * below)) / mass, float(np.sum(w * chosen * above)) / mass
            zone_fit[size][z] = ZoneShare(low=round(a, 4), ideal=round(1 - a - b, 4), high=round(b, 4), low_word=lo, high_word=hi)
    rg = None
    if regrade:
        r = cfg.regrade
        rg = regrade_search(rows, chart.chart_type, ctx, r["grid_cm"], r["step_cm"], r["passes"], r["min_gain"])
    return SimResult(
        n_bodies=len(pop), population_source=pop.source, population_seed=pop.seed, reweighted=pop.reweighted_to is not None, banner=pop.banner(),
        category=product.category, fit_type=product.fit_type, chart_type=chart.chart_type, sizes=ev.sizes, rows=rows, size_mix=mix,
        no_recommendation_share=round(float(np.sum(w * (ev.rec < 0))), 4), no_good_size_share=round(1 - coverage, 4), coverage=round(coverage, 4),
        recommended_counts=mix, zone_fit=zone_fit, regrade=rg, seconds=round(time.perf_counter() - started, 3))


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product", help="Product JSON with a size_chart (see samples/product_kurta.json)")
    ap.add_argument("--no-regrade", action="store_true")
    ap.add_argument("--target", action="store_true", help="reweight to the target market in config/sim.yaml")
    ap.add_argument("--memo", action="store_true", help="also write the merchandiser memo (needs a model backend)")
    a = ap.parse_args()
    product = Product.model_validate(json.loads(Path(a.product).read_text(encoding="utf-8")))
    r = simulate(product, regrade=not a.no_regrade, target=a.target)
    print(f"{r.banner}\n")
    print(f"{product.title}: {r.category.value}, {r.fit_type.value} fit, {r.chart_type} chart, {r.n_bodies} bodies, {r.seconds:.2f} s\n")
    print("size mix (best size): " + "  ".join(f"{s} {v:.1%}" for s, v in r.size_mix.items()) + f"  | none {r.no_recommendation_share:.1%}")
    print(f"no good size: {r.no_good_size_share:.1%}   coverage: {r.coverage:.1%}\n")
    print(f"{'size':<6}{'zone':<16}{'below':>8}{'ideal':>8}{'above':>8}")
    for size, zs in r.zone_fit.items():
        for z, sh in zs.items():
            print(f"{size:<6}{z.value:<16}{sh.low:>8.1%}{sh.ideal:>8.1%}{sh.high:>8.1%}")
    if r.regrade:
        print(f"\nregrade: coverage {r.regrade.coverage_before:.1%} -> {r.regrade.coverage_after:.1%}")
        for c in r.regrade.changes:
            print(f"  {c.size:<5}{c.zone.value:<16}{c.from_cm:>7.1f} -> {c.to_cm:>6.1f} cm ({c.delta_cm:+.1f})   alone: {c.coverage_gain_alone:+.1%}")
    if a.memo:
        from fitcore.sim_memo import write_memo
        print("\n" + write_memo(r).render())


if __name__ == "__main__":
    main()
