"""Accuracy evaluation for garment flat-lay photo measurement.

Reads a CSV of garment photos (one row per garment) with manual tape ground
truth per measurement field, runs the full fitcore.garment_photo pipeline on
each photo, and reports mean absolute error (MAE) per measurement.

CSV columns:
    image_path    - path to the flat-lay photo (JPEG/PNG), relative paths
                     resolved against the CSV's own directory.
    category      - one of: tops, kurtas, bottoms
    keypoints_path - optional path to a JSON file of corrected keypoints.
                     When present, Gemini is skipped (useful for fully offline
                     regression runs); when absent, the configured LLM_PROVIDER
                     (see fitcore/llm.py) is called.
    <measurement>_cm - one column per applicable GarmentFlatMeasurements field
                     (chest_flat_cm, chest_circ_cm, shoulder_width_cm,
                     sleeve_length_cm, garment_length_cm, hem_width_cm,
                     waist_flat_cm, waist_circ_cm, hip_flat_cm, inseam_cm,
                     leg_opening_cm) holding the manual tape measurement in cm.
                     Leave blank for fields not applicable to that garment.

Usage:
    python eval/garment_photo_eval.py --input eval/garment_photo_fixture.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fitcore.garment_photo.pipeline import A4NotFoundError, measure_garment_photo
from fitcore.garment_photo.schemas import GarmentCategory, KeypointXY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.garment_photo_eval")

MEASUREMENT_FIELDS: tuple[str, ...] = (
    "chest_flat_cm",
    "chest_circ_cm",
    "shoulder_width_cm",
    "sleeve_length_cm",
    "garment_length_cm",
    "hem_width_cm",
    "waist_flat_cm",
    "waist_circ_cm",
    "hip_flat_cm",
    "inseam_cm",
    "leg_opening_cm",
)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_garment_photo_csv(csv_path: Path | str) -> dict[str, Any]:
    """Run the pipeline against every row of the CSV and compute per-field MAE.

    Returns a dict with per-field metrics and a list of per-row diagnostics
    (including rows that failed, e.g. A4 not detected).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")

    rows = _load_csv_rows(csv_path)
    base_dir = csv_path.parent

    field_errors: dict[str, list[float]] = {f: [] for f in MEASUREMENT_FIELDS}
    row_diagnostics: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        image_path = base_dir / row["image_path"]
        category_str = row["category"].strip().lower()
        diag: dict[str, Any] = {"row": idx, "image_path": str(image_path), "category": category_str}

        try:
            category = GarmentCategory(category_str)
        except ValueError:
            diag["error"] = f"Unknown category '{category_str}'"
            row_diagnostics.append(diag)
            continue

        if not image_path.exists():
            diag["error"] = f"Missing image file: {image_path}"
            row_diagnostics.append(diag)
            continue

        corrected_keypoints = None
        kp_path_str = (row.get("keypoints_path") or "").strip()
        if kp_path_str:
            kp_path = base_dir / kp_path_str
            if not kp_path.exists():
                diag["error"] = f"Missing keypoints file: {kp_path}"
                row_diagnostics.append(diag)
                continue
            raw_kp = json.loads(kp_path.read_text(encoding="utf-8"))
            corrected_keypoints = {name: KeypointXY(**v) for name, v in raw_kp.items()}

        try:
            result = measure_garment_photo(
                image_bytes=image_path.read_bytes(),
                category=category,
                corrected_keypoints=corrected_keypoints,
            )
        except A4NotFoundError as e:
            diag["error"] = str(e)
            row_diagnostics.append(diag)
            continue
        except Exception as e:
            diag["error"] = f"Pipeline error: {e}"
            row_diagnostics.append(diag)
            continue

        diag["a4_confidence"] = result.a4.a4_confidence
        diag["measurement_confidence"] = result.measurements.confidence
        diag["per_field"] = {}

        measurements_dict = result.measurements.model_dump()
        for field in MEASUREMENT_FIELDS:
            tape_raw = (row.get(field) or "").strip()
            if not tape_raw:
                continue
            try:
                tape_cm = float(tape_raw)
            except ValueError:
                continue
            estimated_cm = measurements_dict.get(field)
            if estimated_cm is None:
                diag["per_field"][field] = {"tape_cm": tape_cm, "estimated_cm": None, "error": "not_detected"}
                continue
            abs_error = abs(estimated_cm - tape_cm)
            field_errors[field].append(abs_error)
            diag["per_field"][field] = {
                "tape_cm": tape_cm,
                "estimated_cm": estimated_cm,
                "abs_error_cm": round(abs_error, 2),
            }

        row_diagnostics.append(diag)

    field_metrics: dict[str, dict[str, float]] = {}
    for field, errors in field_errors.items():
        if errors:
            field_metrics[field] = {
                "sample_count": len(errors),
                "mae_cm": round(float(np.mean(errors)), 2),
                "max_error_cm": round(float(np.max(errors)), 2),
            }

    return {
        "csv_path": str(csv_path),
        "total_rows": len(rows),
        "evaluated_rows": sum(1 for d in row_diagnostics if "error" not in d),
        "failed_rows": sum(1 for d in row_diagnostics if "error" in d),
        "field_metrics": field_metrics,
        "row_diagnostics": row_diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate garment flat-lay photo measurement accuracy.")
    parser.add_argument("--input", required=True, help="Path to garment photo eval CSV")
    parser.add_argument("--output", default="eval/garment_photo_results.json", help="Path to output JSON")
    args = parser.parse_args(argv)

    report = evaluate_garment_photo_csv(args.input)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"GARMENT PHOTO MEASUREMENT EVAL: {report['csv_path']}")
    print("=" * 60)
    print(f"Rows: {report['total_rows']}  Evaluated: {report['evaluated_rows']}  Failed: {report['failed_rows']}")
    for field, m in report["field_metrics"].items():
        print(f"{field:<20} | N={m['sample_count']:<3} | MAE={m['mae_cm']:.2f}cm | max={m['max_error_cm']:.2f}cm")
    for d in report["row_diagnostics"]:
        if "error" in d:
            print(f"  [FAILED] row {d['row']} ({d['image_path']}): {d['error']}")
    print(f"\nSaved results to: {out_p.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
