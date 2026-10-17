"""Extracted versus true size-chart values for the sample pages."""

import json
from pathlib import Path
from typing import Any

from fitcore.read.normalize import CIRCUMFERENCE, to_cm
from fitcore.schemas import MeasurementZone as Z
from fitcore.schemas import Product

SAMPLES = Path(__file__).resolve().parents[3] / "samples" / "products"


STORE = SAMPLES.parents[1] / "web" / "store"


def load_truth(directory: Path | None = None) -> dict[str, Any]:
    return json.loads(((directory or SAMPLES) / "truth.json").read_text(encoding="utf-8"))


def page_for(directory: Path, name: str) -> Path:
    """Saved page for a truth key: samples use the key as the file name, the store prefixes it with store_."""
    for candidate in (f"{name}.html", f"{name.removeprefix('store_')}.html"):
        if (directory / candidate).is_file():
            return directory / candidate
    raise FileNotFoundError(f"{name} in {directory}")


def true_cm(truth: dict[str, Any]) -> dict[str, dict[str, float]]:
    flat = truth.get("circumference_basis") == "flat" and truth["chart_type"] == "garment"
    return {
        size: {zone: to_cm(v * (2 if flat and Z(zone) in CIRCUMFERENCE else 1), truth["unit"]) for zone, v in row.items()}
        for size, row in truth["sizes"].items()
    }


def compare_chart(product: Product, truth: dict[str, Any], tol_cm: float = 0.6) -> list[dict[str, Any]]:
    """One row per size and zone: true cm, extracted cm, difference and whether it is within tol_cm."""
    rows = []
    chart = product.size_chart
    for size, zones in true_cm(truth).items():
        for zone, tv in zones.items():
            ev = chart.sizes.get(size, {}).get(Z(zone)) if chart else None
            rows.append({"size": size, "zone": zone, "true_cm": tv, "extracted_cm": ev,
                         "diff_cm": None if ev is None else round(ev - tv, 2),
                         "ok": ev is not None and abs(ev - tv) <= tol_cm})
    return rows


def format_table(name: str, product: Product, truth: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    chart = product.size_chart
    lines = [f"== {name} ==",
             f"category {product.category.value} (true {truth['category']}) | chart_type {chart.chart_type if chart else '-'} (true {truth['chart_type']}) | "
             f"unit {chart.original_unit if chart else '-'} (true {truth['unit']}) | pre_shrunk {product.pre_shrunk} (true {truth['pre_shrunk']})",
             f"{'size':<6}{'zone':<16}{'true cm':>9}{'extracted':>11}{'diff':>7}"]
    for r in rows:
        ext = "-" if r["extracted_cm"] is None else f"{r['extracted_cm']:.1f}"
        diff = "" if r["diff_cm"] is None else f"{r['diff_cm']:+.1f}"
        lines.append(f"{r['size']:<6}{r['zone']:<16}{r['true_cm']:>9.1f}{ext:>11}{diff:>7}{'' if r['ok'] else '   <-- MISMATCH'}")
    ok = sum(r["ok"] for r in rows)
    lines.append(f"{ok}/{len(rows)} values match within 0.6 cm" + (f" | chart flags: {chart.flags}" if chart and chart.flags else ""))
    return "\n".join(lines)
