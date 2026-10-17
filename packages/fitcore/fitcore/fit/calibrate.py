"""Calibrate target ease from size charts that list body and garment measurements side by side.

    python -m fitcore.fit.calibrate samples/calibration_example.json [more.json ...] [--out proposed.yaml]
    python -m fitcore.fit.calibrate --extract saved_chart.png      # draft a chart JSON with the LLM, then review it

Ease per size and zone is garment minus body. Samples are pooled per category, fit type and zone, and a
min/ideal/max is proposed next to the current config. Nothing is applied: copy accepted values into
fitcore/config/ease.yaml yourself.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from fitcore.fit.config import get_config
from fitcore.schemas import Category, FitType, MeasurementZone

CIRCUMFERENCE = {MeasurementZone.chest, MeasurementZone.waist, MeasurementZone.hip, MeasurementZone.thigh}


class CalibrationSize(BaseModel):
    body: dict[MeasurementZone, float] = Field(default_factory=dict)
    garment: dict[MeasurementZone, float] = Field(default_factory=dict)


class CalibrationChart(BaseModel):
    """One product's chart. Put where it came from in `source`; mark made-up data as simulated."""
    category: Category
    fit_type: FitType
    unit: Literal["cm", "in"] = "cm"
    garment_basis: Literal["full", "flat"] = Field(
        default="full", description="'flat': garment circumferences are measured flat (half), so they are doubled"
    )
    source: str = ""
    sizes: dict[str, CalibrationSize]


def _to_cm(v: float, unit: str) -> float:
    return v * 2.54 if unit == "in" else v


def chart_eases(chart: CalibrationChart) -> dict[MeasurementZone, list[float]]:
    """Ease samples per zone (cm) from one chart; zones without both a body and a garment value are skipped."""
    cfg = get_config()
    out: dict[MeasurementZone, list[float]] = defaultdict(list)
    for size in chart.sizes.values():
        for zone, g in size.garment.items():
            body = size.body.get(cfg.body_ref.get(zone, zone))
            if body is None:
                continue
            g_cm, b_cm = _to_cm(g, chart.unit), _to_cm(body, chart.unit)
            if chart.garment_basis == "flat" and zone in CIRCUMFERENCE:
                g_cm *= 2
            out[zone].append(round(g_cm - b_cm, 2))
    return out


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def propose(samples: list[float]) -> tuple[float, float, float]:
    """(min, ideal, max): median as ideal; the observed spread (p10-p90, or min-max under 5 samples) as the range."""
    ideal = statistics.median(samples)
    lo, hi = (min(samples), max(samples)) if len(samples) < 5 else (_quantile(samples, 0.1), _quantile(samples, 0.9))
    return round(min(lo, ideal), 1), round(ideal, 1), round(max(hi, ideal), 1)


def calibrate(charts: list[CalibrationChart], min_samples: int = 3) -> dict:
    """Pool ease samples and compare with the current config. Returns rows plus the proposed YAML structure."""
    cfg = get_config()
    pooled: dict[tuple[Category, FitType, MeasurementZone], list[float]] = defaultdict(list)
    sources: dict[tuple[Category, FitType, MeasurementZone], set[str]] = defaultdict(set)
    for chart in charts:
        for zone, eases in chart_eases(chart).items():
            key = (chart.category, chart.fit_type, zone)
            pooled[key].extend(eases)
            sources[key].add(chart.source or "unnamed")
    rows, proposed = [], {}
    for key in sorted(pooled, key=lambda k: (k[0].value, k[1].value, k[2].value)):
        cat, fit, zone = key
        samples = pooled[key]
        cur = cfg.target_ease_cm.get(cat, {}).get(fit, {}).get(zone)
        new = propose(samples)
        enough = len(samples) >= min_samples and len(sources[key]) >= 2
        rows.append({
            "category": cat.value, "fit_type": fit.value, "zone": zone.value, "n": len(samples),
            "products": len(sources[key]), "current": [cur.min, cur.ideal, cur.max] if cur else None,
            "proposed": list(new), "enough_data": enough,
        })
        if enough:
            proposed.setdefault(cat.value, {}).setdefault(fit.value, {})[zone.value] = list(new)
    return {"rows": rows, "proposed_yaml": {"target_ease_cm": proposed}}


def print_report(result: dict, min_samples: int) -> None:
    print(f"{'category':<9}{'fit':<10}{'zone':<15}{'n':>3}{'prod':>5}  {'current (min/ideal/max)':<26}{'proposed':<22}")
    for r in result["rows"]:
        cur = "/".join(f"{v:g}" for v in r["current"]) if r["current"] else "-"
        new = "/".join(f"{v:g}" for v in r["proposed"])
        flag = "" if r["enough_data"] else f"  (need >= {min_samples} sizes from >= 2 products)"
        print(f"{r['category']:<9}{r['fit_type']:<10}{r['zone']:<15}{r['n']:>3}{r['products']:>5}  {cur:<26}{new:<22}{flag}")


_SYSTEM = (
    "You read a retailer's size chart and copy its numbers into the schema. Copy only numbers that are printed; "
    "never estimate. 'body' holds body measurements per size, 'garment' holds garment measurements per size. "
    "Use zone names: chest (bust), waist, hip, shoulder_width, sleeve_length, garment_length, inseam, thigh, back_rise. "
    "Set unit to what the chart prints. If circumferences are labelled flat or 'across', set garment_basis to flat."
)


def extract(path: Path) -> CalibrationChart:
    """Draft a chart JSON from a saved image, HTML or text file. A human must review the result before use."""
    from fitcore import llm

    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        parts: list = ["Extract the size chart. Product category and fit type: infer from the page title if shown.", path.read_bytes()]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        parts = ["Extract the size chart from this page text:\n" + re.sub(r"\s+", " ", text)[:20000]]
    chart = llm.generate_json("pro", parts, CalibrationChart, system=_SYSTEM)
    if not chart.source:
        chart.source = f"extracted from {path.name}"
    return chart


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("charts", nargs="*", help="calibration chart JSON files")
    ap.add_argument("--out", help="write proposed ease YAML here (never applied automatically)")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--extract", metavar="FILE", help="draft a chart JSON from an image/HTML/text file and exit")
    args = ap.parse_args()

    if args.extract:
        src = Path(args.extract)
        chart = extract(src)
        dest = src.with_suffix(".calib.json")
        dest.write_text(chart.model_dump_json(indent=2), encoding="utf-8")
        print(f"Draft written to {dest}. Check every number against the page before using it.")
        return
    if not args.charts:
        ap.error("give at least one chart JSON, or --extract FILE")
    charts = [CalibrationChart.model_validate(json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.charts]
    result = calibrate(charts, args.min_samples)
    print_report(result, args.min_samples)
    if args.out:
        Path(args.out).write_text(
            "# PROPOSED by fitcore.fit.calibrate. Review, then copy into fitcore/config/ease.yaml.\n"
            + yaml.safe_dump(result["proposed_yaml"], sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nProposal written to {args.out} (not applied).")


if __name__ == "__main__":
    main()
