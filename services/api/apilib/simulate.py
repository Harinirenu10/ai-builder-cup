"""Pre-production return simulation for the merchandiser: run a size chart against a synthetic population."""

import os
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from apilib.auth import RateLimiter, err, get_limiter
from apilib.fitquery import enforce_rate_limit
from apilib.returns import desk_merchant
from apilib.stores import Store, get_store
from fitcore.population import Population, load_or_create
from fitcore.schemas import Category, FabricPart, FitType, MeasurementZone as Z, Product, SizeChart, StretchClass
from fitcore.sim import SimResult, simulate
from fitcore.sim_memo import Memo, memo_facts, write_memo

router = APIRouter()
_population: Population | None = None


class SimChart(BaseModel):
    sizes: dict[str, dict[Z, float]] = Field(min_length=1, max_length=14)
    chart_type: Literal["body", "garment"] = "garment"
    unit: Literal["cm", "in"] = "cm"


class SimulateIn(BaseModel):
    """A published studio chart (`chart_id`) or a chart sent inline (an upload)."""
    chart_id: str | None = None
    chart: SimChart | None = None
    category: Category | None = None
    fit_type: FitType = FitType.regular
    fabric: list[FabricPart] = Field(default_factory=list)
    stretch_class: StretchClass = StretchClass.none
    pre_shrunk: bool | None = None
    target: bool = Field(default=False, description="Reweight the population to the target market in config/sim.yaml")
    regrade: bool = True


class MemoOutResponse(BaseModel):
    memo: Memo
    seconds: float


def get_population() -> Population:
    """The cached population: read from parquet (POPULATION_PATH, a local path or gs://), generated and cached there if missing.
    Loaded once per process."""
    global _population
    if _population is None:
        path = os.environ.get("POPULATION_PATH") or str(Path(os.environ.get("API_DATA_DIR", Path(__file__).resolve().parent.parent / ".data")) / "population_standin_v1.parquet")
        _population = load_or_create(path)
    return _population


def get_memo_writer() -> Callable[[SimResult], Memo]:
    return write_memo


def build_product(body: SimulateIn, store: Store, merchant: dict[str, Any]) -> Product:
    if body.chart_id:
        c = store.get("charts", body.chart_id)
        if not c or c["merchant_id"] != merchant["id"]:
            raise err(404, "not_found", "No such chart.")
        chart = SizeChart(sizes={s: {Z(z): v for z, v in row.items()} for s, row in c["sizes"].items()}, chart_type="garment")
        category = body.category or Category(c["category"])
        fit = body.fit_type if "fit_type" in body.model_fields_set or not c.get("fit_type") else FitType(c["fit_type"])
        title = c["name"]
    elif body.chart:
        if body.category is None:
            raise err(422, "category_required", "Say which category the chart is for.")
        chart = SizeChart(sizes=body.chart.sizes, chart_type=body.chart.chart_type, unit=body.chart.unit)
        category, fit, title = body.category, body.fit_type, "uploaded chart"
    else:
        raise err(422, "chart_required", "Send a chart_id or a chart.")
    return Product(id="simulation", brand=merchant["name"], title=title, category=category, fit_type=fit, fabric=body.fabric,
                   stretch_class=body.stretch_class, pre_shrunk=body.pre_shrunk, size_chart=chart)


def run(body: SimulateIn, store: Store, merchant: dict[str, Any], pop: Population) -> SimResult:
    try:
        return simulate(build_product(body, store, merchant), pop, regrade=body.regrade, target=body.target)
    except ValueError as exc:
        raise err(422, "cannot_simulate", str(exc)) from exc


@router.post("/biz/simulate", response_model=SimResult)
def simulate_chart(body: SimulateIn, m: dict = Depends(desk_merchant), store: Store = Depends(get_store), pop: Population = Depends(get_population)) -> SimResult:
    """The statistics: size mix, tight or loose shares per zone per size, the share with no good size, regrade suggestions.
    Runs on the cached population in about a second. No per-body data is returned."""
    return run(body, store, m, pop)


@router.post("/biz/simulate/memo", response_model=MemoOutResponse)
def simulate_memo(body: SimulateIn, m: dict = Depends(desk_merchant), store: Store = Depends(get_store), pop: Population = Depends(get_population),
                  writer: Callable = Depends(get_memo_writer), limiter: RateLimiter = Depends(get_limiter)) -> MemoOutResponse:
    """The merchandiser memo for the same request. A separate call because the model takes tens of seconds; the page shows the
    statistics first. The simulation is deterministic, so the memo is written from exactly the numbers the page shows."""
    import time

    enforce_rate_limit(m, limiter)
    started = time.monotonic()
    result = run(body, store, m, pop)
    memo = writer(result)
    return MemoOutResponse(memo=memo, seconds=round(time.monotonic() - started, 1))


@router.post("/biz/simulate/facts")
def simulate_facts(body: SimulateIn, m: dict = Depends(desk_merchant), store: Store = Depends(get_store), pop: Population = Depends(get_population)) -> dict[str, Any]:
    """The exact statistics the memo is written from, so a reader can check the memo's numbers."""
    return memo_facts(run(body, store, m, pop))
