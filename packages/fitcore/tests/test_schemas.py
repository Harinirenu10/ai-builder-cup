import pytest
from pydantic import ValidationError
from fitcore.schemas import (
    MeasurementZone,
    Measurement,
    Measurements,
    SizeChart,
    Product,
    FitResult,
)


def test_measurement_zone_enum_contains_all_12_required_zones():
    expected_zones = {
        "chest",
        "waist",
        "hip",
        "shoulder_width",
        "sleeve_length",
        "torso_length",
        "garment_length",
        "inseam",
        "thigh",
        "neck",
        "back_rise",
        "upper_back",
    }
    actual_zones = {z.value for z in MeasurementZone}
    assert expected_zones == actual_zones


def test_measurement_validates_cm_and_confidence():
    m = Measurement(value_cm=102.5, confidence=0.95, source="3d_twin")
    assert m.value_cm == 102.5
    assert m.confidence == 0.95

    # Out-of-bounds confidence must raise
    with pytest.raises(ValidationError):
        Measurement(value_cm=102.5, confidence=1.5, source="3d_twin")

    # Negative value_cm must raise
    with pytest.raises(ValidationError):
        Measurement(value_cm=-5.0, confidence=0.8, source="manual")


def test_measurements_collection_helpers():
    m_collection = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=100.0, confidence=0.9, source="tape"),
            MeasurementZone.waist: Measurement(value_cm=82.0, confidence=0.85, source="tape"),
        }
    )
    assert m_collection.get_cm(MeasurementZone.chest) == 100.0
    assert m_collection.get_cm(MeasurementZone.waist) == 82.0
    assert m_collection.get_cm(MeasurementZone.hip) is None


def test_size_chart_and_product():
    chart = SizeChart(
        sizes={
            "S": {MeasurementZone.chest: 96.0, MeasurementZone.garment_length: 68.0},
            "M": {MeasurementZone.chest: 104.0, MeasurementZone.garment_length: 71.0},
            "L": {MeasurementZone.chest: 112.0, MeasurementZone.garment_length: 74.0},
        },
        chart_type="garment",
        size_system="ALPHA",
    )
    product = Product(
        id="prod_tee_001",
        brand="Demo Threads Co.",
        title="Heavyweight Boxy Tee",
        size_chart=chart,
        material_composition={"cotton": 1.0},
    )
    assert product.id == "prod_tee_001"
    assert product.size_chart.sizes["M"][MeasurementZone.chest] == 104.0


def test_fit_result_describes_garment_not_body():
    fit = FitResult(
        recommended_size="M",
        confidence=0.92,
        verdict="Standard regular ease with +5.0 cm room around chest.",
        zone_deltas={MeasurementZone.chest: 5.0},
    )
    assert fit.recommended_size == "M"
    assert "chest" in fit.zone_deltas
