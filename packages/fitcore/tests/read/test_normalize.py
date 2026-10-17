import pytest

from fitcore.read.models import (
    ExtractedProduct, ExtractedReviews, PageExtraction, RawMeasure, RawReviewerContext, RawSize, RawZoneSignal,
)
from fitcore.read.normalize import (
    assemble, build_chart, build_reviews, infer_unit, non_monotonic_zones, normalize_zone, order_sizes,
    resolve_chart_type, size_rank, to_cm,
)
from fitcore.schemas import Category, MeasurementZone as Z


def raw_size(label, **vals):
    return RawSize(label=label, measurements=[RawMeasure(label=k.replace("_", " "), value_min=v) for k, v in vals.items()])


def product(sizes, **kw):
    return ExtractedProduct(title="Test tee", category="tshirt", sizes=sizes, **kw)


# ----------------------------------------------------------------------------- unit conversion


@pytest.mark.parametrize("value,unit,expected", [(40, "in", 101.6), (40, "cm", 40.0), (16.5, "in", 41.91), (0, "in", 0.0), (36.5, "in", 92.71)])
def test_to_cm(value, unit, expected):
    assert to_cm(value, unit) == pytest.approx(expected, abs=0.01)


def test_chart_converted_to_cm_and_original_unit_kept():
    chart = build_chart(product([raw_size("M", chest=40)], unit="in", chart_type="garment"), [])
    assert chart.sizes["M"][Z.chest] == pytest.approx(101.6) and chart.unit == "cm" and chart.original_unit == "in"


def test_range_cells_use_the_midpoint():
    ex = product([RawSize(label="M", measurements=[RawMeasure(label="To fit waist", value_min=32, value_max=33)])], unit="in", chart_type="body")
    assert build_chart(ex, []).sizes["M"][Z.waist] == pytest.approx(82.55)


def test_flat_garment_circumference_is_doubled_but_lengths_are_not():
    ex = product([raw_size("M", chest=56, length=74)], unit="cm", chart_type="garment", circumference_basis="flat")
    row = build_chart(ex, []).sizes["M"]
    assert row[Z.chest] == 112.0 and row[Z.garment_length] == 74.0


def test_flat_basis_ignored_for_body_chart_with_flag():
    flags: list[str] = []
    chart = build_chart(product([raw_size("M", chest=96)], unit="cm", chart_type="body", circumference_basis="flat"), flags)
    assert chart.sizes["M"][Z.chest] == 96.0 and "flat_basis_ignored_for_body_chart" in flags


@pytest.mark.parametrize("vals,expected", [({Z.chest: [34, 36, 38]}, "in"), ({Z.chest: [96, 100]}, "cm"), ({Z.sleeve_length: [60]}, "cm"), ({Z.waist: [69.9]}, "in"), ({Z.waist: [70]}, "cm")])
def test_unit_inference(vals, expected):
    assert infer_unit(vals) == expected


def test_unknown_unit_is_inferred_and_flagged():
    flags: list[str] = []
    chart = build_chart(product([raw_size("M", chest=38)], unit="unknown", chart_type="garment"), flags)
    assert chart.original_unit == "in" and "unit_inferred" in flags and chart.sizes["M"][Z.chest] == pytest.approx(96.52)


# ----------------------------------------------------------------------------- zone names


@pytest.mark.parametrize("label,zone", [
    ("To fit chest", Z.chest), ("Chest (in)", Z.chest), ("Bust", Z.chest), ("Garment Chest", Z.chest), ("Pit to pit", Z.chest),
    ("Waist", Z.waist), ("Waist (Relaxed)", Z.waist), ("Hips", Z.hip), ("Seat", Z.hip),
    ("Across Shoulder", Z.shoulder_width), ("Shoulder", Z.shoulder_width), ("Sleeve*", Z.sleeve_length), ("Sleeve Length", Z.sleeve_length),
    ("Front Length", Z.garment_length), ("Body Length", Z.garment_length), ("Kurta Length", Z.garment_length), ("Length**", Z.garment_length),
    ("Inseam", Z.inseam), ("Inside leg", Z.inseam), ("Thigh", Z.thigh), ("Neck", Z.neck), ("Back Rise", Z.back_rise),
    ("Across back", Z.upper_back),
])
def test_zone_normalization(label, zone):
    assert normalize_zone(label) == zone


@pytest.mark.parametrize("label", ["Under Bust", "Front Rise", "Leg Opening", "Height", "", "Weight"])
def test_unmapped_labels_return_none(label):
    assert normalize_zone(label) is None


def test_bare_length_on_trousers_is_not_mapped():
    assert normalize_zone("Length", Category.jeans) is None
    assert normalize_zone("Length", Category.kurta) == Z.garment_length
    assert normalize_zone("Inseam", Category.jeans) == Z.inseam


def test_unmapped_zone_is_flagged_not_dropped_silently():
    flags: list[str] = []
    ex = product([RawSize(label="M", measurements=[RawMeasure(label="Chest", value_min=100), RawMeasure(label="Leg opening", value_min=19)])], unit="cm")
    chart = build_chart(ex, flags)
    assert set(chart.sizes["M"]) == {Z.chest} and "unmapped_zone:Leg opening" in flags


# ----------------------------------------------------------------------------- monotonic flag


def test_non_monotonic_zone_flagged():
    ex = product([raw_size("S", chest=96, length=70), raw_size("M", chest=104, length=72), raw_size("L", chest=100, length=74)], unit="cm", chart_type="garment")
    chart = build_chart(ex, [])
    assert "non_monotonic:chest" in chart.flags and "non_monotonic:garment_length" not in chart.flags


def test_equal_values_are_fine():
    assert non_monotonic_zones({"S": {Z.garment_length: 70.0}, "M": {Z.garment_length: 70.0}, "L": {Z.garment_length: 72.0}}) == []


def test_sizes_listed_largest_first_are_sorted_before_checking():
    ex = product([raw_size("XL", chest=116), raw_size("M", chest=104), raw_size("S", chest=98), raw_size("L", chest=110)], unit="cm", chart_type="garment")
    chart = build_chart(ex, [])
    assert list(chart.sizes) == ["S", "M", "L", "XL"] and not chart.flags


@pytest.mark.parametrize("label,rank", [("S", 2.0), ("xl", 5.0), ("2XL", 6.0), ("XXL", 6.0), ("32", 32.0), ("36 – XS", 36.0), ("Free", None)])
def test_size_rank(label, rank):
    assert size_rank(label) == rank


def test_unrankable_labels_keep_page_order():
    sizes = [raw_size("Free", chest=100), raw_size("Tall", chest=110)]
    assert [s.label for s in order_sizes(sizes)] == ["Free", "Tall"]


# ----------------------------------------------------------------------------- chart type


@pytest.mark.parametrize("model,evidence,expected,guessed", [
    ("garment", "To fit chest", "body", False),          # the label beats the model
    ("body", "Garment measurements (flat)", "garment", False),
    ("garment", None, "garment", False),
    ("unknown", "Body measurements", "body", False),
    ("unknown", None, "body", True),
    ("garment", "Garment size chart: to fit body size", "garment", False),  # both words: trust the model
])
def test_resolve_chart_type(model, evidence, expected, guessed):
    assert resolve_chart_type(model, evidence) == (expected, guessed)


# ----------------------------------------------------------------------------- reviews


def test_no_reviews_read_means_nothing_is_returned():
    raw = ExtractedReviews(reviews_read=0, zone_signals=[RawZoneSignal(zone=Z.chest, direction="tight", mention_count=3)], shrinkage_mentions=4)
    out = build_reviews(raw)
    assert out.reviews_read == 0 and out.zone_signals == [] and out.shrinkage_mentions == 0


def test_counts_cannot_exceed_reviews_read():
    raw = ExtractedReviews(reviews_read=4, zone_signals=[RawZoneSignal(zone=Z.chest, direction="tight", mention_count=9, strength=1.7)],
                           shrinkage_mentions=10, sheerness_mentions=2)
    out = build_reviews(raw)
    assert out.zone_signals[0].mention_count == 4 and out.zone_signals[0].strength == 1.0
    assert out.shrinkage_mentions == 4 and out.sheerness_mentions == 2


def test_duplicate_signals_merge_and_zero_mentions_drop():
    raw = ExtractedReviews(reviews_read=10, zone_signals=[
        RawZoneSignal(zone=Z.chest, direction="tight", mention_count=2, strength=0.4),
        RawZoneSignal(zone=Z.chest, direction="tight", mention_count=2, strength=0.8),
        RawZoneSignal(zone=Z.waist, direction="loose", mention_count=0)])
    out = build_reviews(raw)
    assert len(out.zone_signals) == 1 and out.zone_signals[0].mention_count == 4 and out.zone_signals[0].strength == pytest.approx(0.6)


# ----------------------------------------------------------------------------- assemble


def test_assemble_product_fields():
    ex = ExtractedProduct(title="  Kurta ", brand=None, category="kurta", pre_shrunk="true", pre_shrunk_evidence="pre-washed",
                          fabric=[{"fiber": "Rayon", "percent": 60}, {"fiber": "cotton", "percent": 60}], sizes=[raw_size("M", chest=40)], unit="in", chart_type="garment")
    out = assemble(PageExtraction(product=ex), url="https://shop.example/p/1", product_id="abc", fallback_brand="shop.example")
    p = out.product
    assert out.status == "ok" and p.title == "Kurta" and p.brand == "shop.example" and p.pre_shrunk is True and p.pre_shrunk_evidence == "pre-washed"
    assert sum(f.percent for f in p.fabric) == pytest.approx(100.0, abs=0.2) and any(f.startswith("fabric_percent_sum") for f in out.flags)
    assert "fit_type_defaulted" in out.flags and p.source_url == "https://shop.example/p/1"


def test_unknown_category_is_unsupported():
    out = assemble(PageExtraction(product=ExtractedProduct(title="Running shoe", category=None)), url=None, product_id="x")
    assert out.status == "unsupported" and out.product is None


def test_pre_shrunk_unknown_drops_evidence():
    ex = ExtractedProduct(title="t", category="shirt", pre_shrunk="unknown", pre_shrunk_evidence="maybe")
    assert assemble(PageExtraction(product=ex), url=None, product_id="x").product.pre_shrunk_evidence is None


def test_no_chart_is_flagged():
    out = assemble(PageExtraction(product=ExtractedProduct(title="t", category="shirt")), url=None, product_id="x")
    assert out.product.size_chart is None and "no_size_chart" in out.flags


def test_reviewer_context_is_kept():
    raw = ExtractedReviews(reviews_read=3, zone_signals=[RawZoneSignal(zone=Z.garment_length, direction="short", mention_count=2,
                           reviewer_context=[RawReviewerContext(height_cm=185, torso="long", note="6'1")])])
    assert build_reviews(raw).zone_signals[0].reviewer_context[0].torso == "long"
