"""Calibration tool for photo-based body measurements.

Fits per-zone calibration factors (k_zone) by comparing ground-truth tape measurements
with photo-based estimates across evaluation subjects:
    tape_cm = k_zone * estimated_raw_cm

Minimizes sum of squared errors:
    k_zone = sum(tape_i * est_i) / sum(est_i^2)

Outputs:
- Fitted k factors per zone.
- Mean Absolute Error (MAE) before and after calibration.
- JSON mapping saved to output path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

from fitcore.schemas import MeasurementZone

logger = logging.getLogger("eval.calibrate_k")


def fit_calibration_factors(
    data: list[dict[str, Any]] | Path | str,
) -> tuple[dict[MeasurementZone, float], dict[str, dict[str, float]]]:
    """Fit per-zone calibration factors k from tabular data.

    Args:
        data: Either a list of row dicts or a path to a CSV file.
              Expected columns: 'person', 'zone', and either:
              - 'estimated_cm' and 'tape_cm'
              - OR 'estimated_cm' and 'cm' (where 'cm' is ground truth tape measurement).

    Returns:
        (k_factors_by_zone, evaluation_metrics)
    """
    rows: list[dict[str, Any]] = []
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Calibration CSV file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif isinstance(data, list):
        rows = data
    else:
        raise TypeError(f"Unsupported data type for calibration: {type(data)}")

    if not rows:
        raise ValueError("No data rows found for calibration fitting.")

    # Group by zone
    zone_samples: dict[MeasurementZone, list[tuple[float, float]]] = {z: [] for z in MeasurementZone}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        # Extract zone
        raw_zone = str(row.get("zone", "")).strip().lower()
        try:
            zone = MeasurementZone(raw_zone)
        except ValueError:
            logger.warning("Row %d has unknown zone '%s'; skipping.", idx, raw_zone)
            continue

        # Extract values
        raw_est = row.get("estimated_cm") or row.get("estimated") or row.get("estimate")
        raw_tape = row.get("tape_cm") or row.get("cm") or row.get("ground_truth") or row.get("tape")

        if raw_est is None or raw_tape is None:
            logger.warning("Row %d missing estimated or tape value; skipping.", idx)
            continue

        try:
            cleaned_est = str(raw_est).strip().replace(",", ".")
            cleaned_tape = str(raw_tape).strip().replace(",", ".")
            val_est = float(cleaned_est)
            val_tape = float(cleaned_tape)
        except (ValueError, TypeError):
            continue

        if val_est > 0.0 and val_tape > 0.0 and np.isfinite(val_est) and np.isfinite(val_tape):
            zone_samples[zone].append((val_est, val_tape))


    k_factors: dict[MeasurementZone, float] = {}
    metrics: dict[str, dict[str, float]] = {}

    for zone in MeasurementZone:
        samples = zone_samples[zone]
        if not samples:
            # Default to 1.0 if no data available for this zone
            k_factors[zone] = 1.0
            metrics[zone.value] = {
                "sample_count": 0,
                "k_factor": 1.0,
                "mae_before": 0.0,
                "mae_after": 0.0,
            }
            continue

        estimates = np.array([s[0] for s in samples], dtype=np.float64)
        tapes = np.array([s[1] for s in samples], dtype=np.float64)

        # Closed-form least-squares: k = sum(tape * est) / sum(est^2)
        denom = float(np.sum(estimates ** 2))
        if denom > 1e-6:
            k_opt = float(np.sum(tapes * estimates) / denom)
        else:
            k_opt = 1.0

        # Clamp k to reasonable photogrammetry range [0.70, 1.30]
        k_clamped = max(0.70, min(1.30, k_opt))
        k_factors[zone] = round(k_clamped, 4)

        mae_before = float(np.mean(np.abs(estimates - tapes)))
        mae_after = float(np.mean(np.abs(estimates * k_clamped - tapes)))

        metrics[zone.value] = {
            "sample_count": len(samples),
            "k_factor": round(k_clamped, 4),
            "mae_before": round(mae_before, 2),
            "mae_after": round(mae_after, 2),
            "mae_improvement_pct": round(max(0.0, (mae_before - mae_after) / max(0.01, mae_before) * 100.0), 1),
        }

    return k_factors, metrics


def _fit_k_from_samples(samples: list[tuple[float, float]]) -> float:
    """Closed-form least-squares k for a list of (estimated_cm, tape_cm) pairs, same clamp as fit_calibration_factors."""
    estimates = np.array([s[0] for s in samples], dtype=np.float64)
    tapes = np.array([s[1] for s in samples], dtype=np.float64)
    denom = float(np.sum(estimates ** 2))
    k_opt = float(np.sum(tapes * estimates) / denom) if denom > 1e-6 else 1.0
    return max(0.70, min(1.30, k_opt))


def leave_one_subject_out_eval(
    data: list[dict[str, Any]] | Path | str,
) -> dict[str, dict[str, float]]:
    """Leave-one-subject-out evaluation: for each zone, fit k on all subjects but one
    and measure error on the held-out subject's sample. This avoids reporting
    calibration accuracy on the same subjects the calibration factors were fit from.

    Requires a 'person' column to group samples by subject.

    Returns per-zone dict with:
        - n_subjects: number of distinct subjects with data for this zone
        - held_out_mae: mean absolute error on held-out subjects (unseen during their fold's fit)
        - train_mae: mean absolute error on the same folds' training subjects (in-fold, not held out)
    """
    rows: list[dict[str, Any]] = []
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Calibration CSV file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif isinstance(data, list):
        rows = data
    else:
        raise TypeError(f"Unsupported data type for calibration: {type(data)}")

    # Group by zone, then by subject
    zone_person_samples: dict[MeasurementZone, dict[str, list[tuple[float, float]]]] = {
        z: {} for z in MeasurementZone
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_zone = str(row.get("zone", "")).strip().lower()
        try:
            zone = MeasurementZone(raw_zone)
        except ValueError:
            continue
        person = str(row.get("person", "")).strip()
        raw_est = row.get("estimated_cm") or row.get("estimated") or row.get("estimate")
        raw_tape = row.get("tape_cm") or row.get("cm") or row.get("ground_truth") or row.get("tape")
        if not person or raw_est is None or raw_tape is None:
            continue
        try:
            val_est = float(str(raw_est).strip().replace(",", "."))
            val_tape = float(str(raw_tape).strip().replace(",", "."))
        except (ValueError, TypeError):
            continue
        if val_est > 0.0 and val_tape > 0.0 and np.isfinite(val_est) and np.isfinite(val_tape):
            zone_person_samples[zone].setdefault(person, []).append((val_est, val_tape))

    results: dict[str, dict[str, float]] = {}
    for zone, person_samples in zone_person_samples.items():
        persons = sorted(person_samples)
        if len(persons) < 2:
            # Cannot hold out a subject and still have training data.
            continue

        held_out_errors: list[float] = []
        train_errors: list[float] = []
        for held_person in persons:
            train_samples = [s for p in persons if p != held_person for s in person_samples[p]]
            test_samples = person_samples[held_person]
            k_fold = _fit_k_from_samples(train_samples)

            for est, tape in test_samples:
                held_out_errors.append(abs(est * k_fold - tape))
            for est, tape in train_samples:
                train_errors.append(abs(est * k_fold - tape))

        results[zone.value] = {
            "n_subjects": len(persons),
            "held_out_sample_count": len(held_out_errors),
            "held_out_mae": round(float(np.mean(held_out_errors)), 2),
            "train_sample_count": len(train_errors),
            "train_mae": round(float(np.mean(train_errors)), 2),
        }

    return results


def main() -> int:
    """CLI runner for calibration fitting."""
    parser = argparse.ArgumentParser(description="Fit zone calibration factors k from CSV dataset.")
    parser.add_argument("--input", required=True, help="Path to input CSV (person, zone, tape_cm, estimated_cm)")
    parser.add_argument("--output", default="eval/calibration_factors.json", help="Path to output JSON")
    args = parser.parse_args()

    k_factors, metrics = fit_calibration_factors(args.input)
    loso_metrics = leave_one_subject_out_eval(args.input)

    # Format JSON
    out_dict = {
        "factors": {z.value: k for z, k in k_factors.items()},
        "metrics": metrics,
        "leave_one_subject_out": loso_metrics,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print("=" * 60)
    print("CALIBRATION FACTORS FITTED ON FULL DATASET (in-sample; same subjects used to fit k and report MAE)")
    print("=" * 60)
    for zone_str, m in metrics.items():
        if m["sample_count"] > 0:
            print(
                f"{zone_str:<16} | N={m['sample_count']:<3} | k={m['k_factor']:.4f} | "
                f"MAE: {m['mae_before']:.1f}cm -> {m['mae_after']:.1f}cm ({m['mae_improvement_pct']}% impr)"
            )

    print()
    print("=" * 60)
    print("LEAVE-ONE-SUBJECT-OUT EVAL (held-out subject never seen by that fold's k fit)")
    print("=" * 60)
    for zone_str, m in loso_metrics.items():
        print(
            f"{zone_str:<16} | subjects={m['n_subjects']:<2} | "
            f"train MAE: {m['train_mae']:.2f}cm (N={m['train_sample_count']}) | "
            f"held-out MAE: {m['held_out_mae']:.2f}cm (N={m['held_out_sample_count']})"
        )
    print(f"\nSaved factors to: {out_p.resolve()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
