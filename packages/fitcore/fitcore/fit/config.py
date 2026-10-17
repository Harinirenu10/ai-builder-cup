"""Loads and validates the fit engine's YAML tables from fitcore/config/."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from fitcore.schemas import Category, FitType, MeasurementZone, StretchClass

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class EaseRange(BaseModel):
    min: float
    ideal: float
    max: float


class Shrinkage(BaseModel):
    length: float
    width: float


class Scoring(BaseModel):
    hard_weight: float
    easy_weight: float
    other_weight: float
    base_penalty: float
    per_cm_penalty: float
    penalty_cap: float
    score_halflife: float
    long_pose_minutes: float
    short_pose_factor: float
    confidence_gap_full: float
    verdict_epsilon_cm: float


class Guarantee(BaseModel):
    min_confidence: float


class AlterationAction(BaseModel):
    action: str
    text: str
    max_cm: float


class ZoneActions(BaseModel):
    decrease: AlterationAction
    increase: AlterationAction


class ReviewContext(BaseModel):
    body_zone: MeasurementZone
    op: str
    threshold_cm: float


class ReviewAdapter(BaseModel):
    min_mentions: int
    moderate_from: float
    strong_from: float
    context_share: float


class FitConfig(BaseModel):
    zones_by_category: dict[Category, list[MeasurementZone]]
    hard_zones: dict[Category, list[MeasurementZone]]
    easy_zones: list[MeasurementZone]
    body_ref: dict[MeasurementZone, MeasurementZone]
    length_zones: list[MeasurementZone]
    scoring: Scoring
    guarantee: Guarantee
    target_ease_cm: dict[Category, dict[FitType, dict[MeasurementZone, EaseRange]]]
    stretch_allowance_cm: dict[StretchClass, float]
    shrinkage_pct: dict[str, Shrinkage]
    pre_shrunk_factor: float
    fiber_aliases: dict[str, str]
    rounding_step_cm: float
    min_alteration_cm: float
    actions: dict[MeasurementZone, ZoneActions]
    strength_cm: dict[str, float]
    max_adjustment_cm: float
    contexts: dict[str, ReviewContext]
    adapter: ReviewAdapter


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_config() -> FitConfig:
    raw: dict[str, Any] = {}
    for name in ("zones.yaml", "ease.yaml", "fabric.yaml", "alterations.yaml", "reviews.yaml"):
        raw.update(_load(name))
    for fits in raw["target_ease_cm"].values():
        for zones in fits.values():
            for zone, triple in list(zones.items()):
                lo, ideal, hi = triple
                zones[zone] = {"min": lo, "ideal": ideal, "max": hi}
    return FitConfig.model_validate(raw)
