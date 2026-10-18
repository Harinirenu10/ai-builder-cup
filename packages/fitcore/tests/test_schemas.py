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
        category="tshirt",
        size_chart=chart,
        material_composition={"cotton": 1.0},
    )
    assert product.id == "prod_tee_001"
    assert product.size_chart.sizes["M"][MeasurementZone.chest] == 104.0


def test_fit_result_describes_garment_not_body():
    from fitcore.schemas import Category, FitType
    fit = FitResult(category=Category.tshirt, fit_type=FitType.regular, sizes=[], recommended_size="M", confidence=0.92)
    assert fit.recommended_size == "M"
    assert fit.size_fit("M") is None


# --- Phase 1 Tests: 3D Twin & Body Measurement Pipeline Domain Models ---


def test_job_status_enum_contains_all_8_lifecycle_states():
    from fitcore.schemas import JobStatus
    expected_statuses = {
        "checking",
        "measuring",
        "fitting",
        "refining",
        "done",
        "needs_retake",
        "rejected",
        "failed",
    }
    actual_statuses = {s.value for s in JobStatus}
    assert expected_statuses == actual_statuses


def test_zone_calculation_methods_defines_all_12_zones():
    from fitcore.schemas import ZONE_CALCULATION_METHODS, MeasurementZone
    for zone in MeasurementZone:
        assert zone in ZONE_CALCULATION_METHODS
        info = ZONE_CALCULATION_METHODS[zone]
        assert info["type"] in ("circumference", "linear")
        assert len(info["method"]) > 10


def test_body_twin_job_request_validates_height_and_consent():
    from fitcore.schemas import BodyTwinJobRequest

    req = BodyTwinJobRequest(
        user_id="user_123",
        height_cm=178.5,
        gender="neutral",
        age_18_confirmed=True,
        opt_in_keep_front_photo=False,
        photo_front="data:image/jpeg;base64,abc",
        photo_back="data:image/jpeg;base64,def",
        photo_left="data:image/jpeg;base64,ghi",
        photo_right="data:image/jpeg;base64,jkl",
    )
    assert req.user_id == "user_123"
    assert req.height_cm == 178.5
    assert req.age_18_confirmed is True

    # Height too low
    with pytest.raises(ValidationError):
        BodyTwinJobRequest(
            user_id="user_123",
            height_cm=40.0,
            gender="neutral",
            photo_front="a",
            photo_back="b",
            photo_left="c",
            photo_right="d",
        )

    # Height too high
    with pytest.raises(ValidationError):
        BodyTwinJobRequest(
            user_id="user_123",
            height_cm=270.0,
            gender="neutral",
            photo_front="a",
            photo_back="b",
            photo_left="c",
            photo_right="d",
        )


def test_critic_round_result_bounds_and_adjustments():
    from fitcore.schemas import CriticRoundResult, MeasurementZone

    res = CriticRoundResult(
        round_number=2,
        suggested_adjustments={MeasurementZone.waist: -1.2, MeasurementZone.chest: 0.5},
        suggested_beta_delta=[0.05, -0.02],
        confidence=0.88,
        notes="Garment ease is consistent; small reduction on waist contour.",
    )
    assert res.round_number == 2
    assert res.suggested_adjustments[MeasurementZone.waist] == -1.2
    assert res.confidence == 0.88

    # Invalid round number (must be 1-5)
    with pytest.raises(ValidationError):
        CriticRoundResult(
            round_number=6,
            confidence=0.9,
            notes="Invalid round",
        )


def test_round_log_document_iou_and_rollback_flag():
    from fitcore.schemas import RoundLogDocument, MeasurementZone

    log = RoundLogDocument(
        round=1,
        mean_iou_before=0.880,
        mean_iou_after=0.885,
        iou_delta=0.005,
        accepted=True,
        step_size=1.0,
        zone_adjustments={MeasurementZone.waist: -0.5},
        critic_notes="Accepted round",
        critic_latency_ms=950,
        timestamp="2026-10-16T12:00:00Z",
    )
    assert log.accepted is True
    assert log.iou_delta == 0.005
    assert log.step_size == 1.0


def test_round_log_document_allows_rejected_rounds_with_low_or_negative_iou_delta():
    from fitcore.schemas import RoundLogDocument, MeasurementZone

    # iou_delta < 0.002, rejected round
    log_low = RoundLogDocument(
        round=2,
        mean_iou_before=0.885,
        mean_iou_after=0.886,
        iou_delta=0.001,  # below 0.002 threshold
        accepted=False,
        step_size=0.5,
        zone_adjustments={MeasurementZone.chest: 1.0},
        critic_notes="Insufficient improvement; rolled back and step halved.",
        critic_latency_ms=820,
        timestamp="2026-10-16T12:01:00Z",
    )
    assert log_low.accepted is False
    assert log_low.iou_delta == 0.001
    assert log_low.step_size == 0.5

    # Negative iou_delta (regression), rejected round
    log_neg = RoundLogDocument(
        round=3,
        mean_iou_before=0.885,
        mean_iou_after=0.880,
        iou_delta=-0.005,
        accepted=False,
        step_size=0.25,
        critic_notes="IoU degraded; rollback applied.",
        critic_latency_ms=750,
        timestamp="2026-10-16T12:02:00Z",
    )
    assert log_neg.accepted is False
    assert log_neg.iou_delta == -0.005
    assert log_neg.step_size == 0.25



def test_job_document_and_twin_document():
    from fitcore.schemas import (
        JobDocument,
        JobStatus,
        TwinDocument,
        Measurements,
        Measurement,
        MeasurementZone,
    )

    job = JobDocument(
        id="job_001",
        user_id="user_456",
        status=JobStatus.checking,
        height_cm=172.0,
        gender="female",
        created_at="2026-10-16T12:00:00Z",
        updated_at="2026-10-16T12:00:00Z",
    )
    assert job.status == JobStatus.checking
    assert job.gender == "female"

    m_zones = {
        MeasurementZone.chest: Measurement(value_cm=92.0, confidence=0.95, source="3d_twin"),
        MeasurementZone.waist: Measurement(value_cm=70.0, confidence=0.93, source="3d_twin"),
    }
    twin = TwinDocument(
        id="twin_001",
        user_id="user_456",
        gender="female",
        height_cm=172.0,
        betas=[0.1, -0.2, 0.3],
        measurements=Measurements(zones=m_zones),
        glb_storage_url="gs://bucket/twins/twin_001.glb",
        final_mean_iou=0.912,
        created_at="2026-10-16T12:00:05Z",
    )
    assert twin.id == "twin_001"
    assert twin.measurements.get_cm(MeasurementZone.chest) == 92.0
    assert twin.final_mean_iou == 0.912


def test_critic_report_and_region_feedback_valid():
    from fitcore.schemas import (
        CriticReport,
        CriticRegionFeedback,
        CriticView,
        CriticRegion,
        CriticIssue,
        CriticCause,
    )

    feedback = [
        CriticRegionFeedback(
            view=CriticView.front,
            region=CriticRegion.chest,
            issue=CriticIssue.too_wide,
            severity=0.6,
            cause=CriticCause.model_error,
            trust=0.9,
        ),
        CriticRegionFeedback(
            view=CriticView.left,
            region=CriticRegion.waist,
            issue=CriticIssue.ok,
            severity=0.0,
            cause=CriticCause.model_error,
            trust=0.95,
        ),
    ]
    report = CriticReport(
        regions=feedback,
        stop=False,
        summary="Front silhouette slightly wider than subject at chest plane.",
    )
    assert len(report.regions) == 2
    assert report.regions[0].region == CriticRegion.chest
    assert report.regions[0].issue == CriticIssue.too_wide
    assert report.regions[0].severity == 0.6
    assert report.stop is False


def test_critic_region_feedback_invalid_enums_and_ranges():
    from fitcore.schemas import CriticRegionFeedback

    # Invalid view
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="isometric",  # type: ignore
            region="chest",
            issue="too_wide",
            severity=0.5,
            cause="model_error",
            trust=0.8,
        )

    # Invalid region
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="front",
            region="head",  # type: ignore
            issue="too_wide",
            severity=0.5,
            cause="model_error",
            trust=0.8,
        )

    # Invalid issue
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="front",
            region="chest",
            issue="blurry",  # type: ignore
            severity=0.5,
            cause="model_error",
            trust=0.8,
        )

    # Invalid cause
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="front",
            region="chest",
            issue="too_wide",
            severity=0.5,
            cause="unknown_glitch",  # type: ignore
            trust=0.8,
        )

    # Severity out of bounds (> 1.0)
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="front",
            region="chest",
            issue="too_wide",
            severity=1.2,
            cause="model_error",
            trust=0.8,
        )

    # Trust out of bounds (< 0.0)
    with pytest.raises(ValidationError):
        CriticRegionFeedback(
            view="front",
            region="chest",
            issue="too_wide",
            severity=0.5,
            cause="model_error",
            trust=-0.1,
        )


def test_job_and_twin_document_height_boundaries():
    from fitcore.schemas import JobDocument, JobStatus, TwinDocument, Measurements

    # JobDocument height lower bound (50.0 is invalid, gt=50.0 required)
    with pytest.raises(ValidationError):
        JobDocument(
            id="job_low",
            user_id="u1",
            status=JobStatus.checking,
            height_cm=50.0,
            created_at="2026-10-16T12:00:00Z",
            updated_at="2026-10-16T12:00:00Z",
        )

    # JobDocument height upper bound (260.0 is invalid, lt=260.0 required)
    with pytest.raises(ValidationError):
        JobDocument(
            id="job_high",
            user_id="u1",
            status=JobStatus.checking,
            height_cm=260.0,
            created_at="2026-10-16T12:00:00Z",
            updated_at="2026-10-16T12:00:00Z",
        )

    # TwinDocument height boundaries
    with pytest.raises(ValidationError):
        TwinDocument(
            id="twin_low",
            user_id="u1",
            gender="neutral",
            height_cm=49.9,
            measurements=Measurements(),
            created_at="2026-10-16T12:00:00Z",
        )

    # TwinDocument final_mean_iou bounds
    with pytest.raises(ValidationError):
        TwinDocument(
            id="twin_bad_iou",
            user_id="u1",
            gender="neutral",
            height_cm=170.0,
            measurements=Measurements(),
            final_mean_iou=1.05,
            created_at="2026-10-16T12:00:00Z",
        )


def test_all_12_zones_in_measurements_collection():
    from fitcore.schemas import Measurements, Measurement, MeasurementZone

    all_zones = {
        zone: Measurement(value_cm=round(50.0 + idx * 5.0, 1), confidence=0.9, source="3d_twin")
        for idx, zone in enumerate(MeasurementZone)
    }
    assert len(all_zones) == 12

    collection = Measurements(zones=all_zones)
    for zone in MeasurementZone:
        val = collection.get_cm(zone)
        assert val is not None
        assert val > 0


def test_size_chart_inches_are_converted_to_cm():
    chart = SizeChart(sizes={"M": {MeasurementZone.chest: 40.0}}, unit="in")
    assert chart.unit == "cm"
    assert chart.sizes["M"][MeasurementZone.chest] == pytest.approx(101.6)


def test_product_legacy_material_composition_becomes_fabric():
    p = Product(id="x", brand="b", title="t", category="shirt", material_composition={"cotton": 0.98, "elastane": 0.02})
    assert {f.fiber: round(f.percent) for f in p.fabric} == {"cotton": 98, "elastane": 2}


def test_fit_result_defaults():
    fit = FitResult(category="kurta", fit_type="regular", sizes=[])
    assert fit.recommended_size is None
    assert fit.confidence == 0.0
