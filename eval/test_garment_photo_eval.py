"""Smoke test for the garment photo eval harness against the synthetic fixture.

This only proves the CSV-driven evaluation pipeline runs correctly end-to-end
(image load -> pipeline -> MAE aggregation). It is not a real-garment accuracy
claim; see eval/results.md for the real-garment validation status.
"""

from __future__ import annotations

from pathlib import Path

from eval.garment_photo_eval import evaluate_garment_photo_csv

FIXTURE_CSV = Path(__file__).parent / "garment_photo_synthetic_fixture.csv"


def test_evaluate_synthetic_fixture_runs_and_reports_zero_mae():
    report = evaluate_garment_photo_csv(FIXTURE_CSV)

    assert report["total_rows"] == 1
    assert report["evaluated_rows"] == 1
    assert report["failed_rows"] == 0

    for field in ("chest_flat_cm", "chest_circ_cm", "shoulder_width_cm", "garment_length_cm"):
        assert field in report["field_metrics"]
        assert report["field_metrics"][field]["mae_cm"] == 0.0


def test_evaluate_missing_csv_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        evaluate_garment_photo_csv(Path(__file__).parent / "does_not_exist.csv")
