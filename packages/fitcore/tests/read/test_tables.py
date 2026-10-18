"""Tests for fitcore.read.tables: reading a size chart out of a page's own HTML table.

The reader falls back to this when the extraction model returns no chart, which on a small local
model is a large share of attempts even for a chart printed plainly in the page text. What comes
out must be the same lenient shape the model fills in, so fitcore.read.normalize.build_chart
keeps doing all the unit/zone/chart-type work.
"""

from __future__ import annotations

from fitcore.read.models import ExtractedProduct, PageExtraction, RawFabric
from fitcore.read.normalize import assemble, build_chart
from fitcore.read.tables import parse_size_chart_html
from fitcore.schemas import Category, MeasurementZone as Z

SHIRT_MODAL = """
<html><body>
<h1>Oxford Button-Down Shirt</h1>
<button id="size-guide-btn">Size guide</button>
<div id="chart-modal" hidden>
  <h2>Size guide: garment measurements</h2>
  <p>Measurements of the garment laid flat, full circumference, in cm.</p>
  <table>
    <tr><th>Size</th><th>Chest (cm)</th><th>Shoulder (cm)</th><th>Sleeve length (cm)</th><th>Length (cm)</th></tr>
    <tr><td>S</td><td>108</td><td>44</td><td>60</td><td>72</td></tr>
    <tr><td>M</td><td>114</td><td>46</td><td>61.5</td><td>74</td></tr>
    <tr><td>L</td><td>120</td><td>48</td><td>63</td><td>76</td></tr>
    <tr><td>XL</td><td>126</td><td>50</td><td>64.5</td><td>78</td></tr>
  </table>
</div></body></html>
"""


def test_reads_the_demo_shirt_chart_with_unit_and_chart_type_evidence():
    parsed = parse_size_chart_html(SHIRT_MODAL, Category.shirt)
    assert parsed is not None
    assert [s.label for s in parsed.sizes] == ["S", "M", "L", "XL"]
    assert parsed.unit == "cm"
    # The page states this outright, so it is evidence rather than normalize.py's inference.
    assert parsed.chart_type == "garment"
    assert parsed.chart_type_evidence and "garment measurement" in parsed.chart_type_evidence.lower()

    chest = {s.label: next(m.value_min for m in s.measurements if m.label.startswith("Chest")) for s in parsed.sizes}
    assert chest == {"S": 108.0, "M": 114.0, "L": 120.0, "XL": 126.0}


def test_parsed_chart_normalizes_through_build_chart_to_real_zones_in_cm():
    """The whole point of matching the model's shape: build_chart handles it unchanged."""
    parsed = parse_size_chart_html(SHIRT_MODAL, Category.shirt)
    assert parsed is not None
    ex = ExtractedProduct(
        title="Oxford Button-Down Shirt", category=Category.shirt, sizes=parsed.sizes,
        unit=parsed.unit, chart_type=parsed.chart_type, chart_type_evidence=parsed.chart_type_evidence,
    )
    flags: list[str] = []
    chart = build_chart(ex, flags)
    assert chart is not None
    assert chart.unit == "cm" and chart.original_unit == "cm"
    assert chart.chart_type == "garment"
    assert chart.sizes["M"][Z.chest] == 114.0
    assert chart.sizes["M"][Z.shoulder_width] == 46.0
    assert chart.sizes["XL"][Z.garment_length] == 78.0
    # Evidence came from the page, so neither unit nor chart type had to be guessed.
    assert "unit_inferred" not in flags and "chart_type_guessed" not in flags


def test_assemble_produces_a_usable_product_from_a_parsed_chart():
    parsed = parse_size_chart_html(SHIRT_MODAL, Category.shirt)
    assert parsed is not None
    ext = PageExtraction(product=ExtractedProduct(
        title="Oxford Button-Down Shirt", category=Category.shirt, sizes=parsed.sizes, unit=parsed.unit,
        chart_type=parsed.chart_type, chart_type_evidence=parsed.chart_type_evidence,
        fabric=[RawFabric(fiber="cotton", percent=100)],
    ))
    outcome = assemble(ext, url="http://localhost/store/shirt.html", product_id="p1", fallback_brand="demo")
    assert outcome.status == "ok" and outcome.product is not None
    assert outcome.product.size_chart is not None
    assert "no_size_chart" not in outcome.flags
    assert sorted(outcome.product.size_chart.sizes) == ["L", "M", "S", "XL"]


def test_reads_a_transposed_chart_sizes_across_the_header():
    html = """<h2>Size chart</h2><table>
      <tr><th>Measurement</th><th>S</th><th>M</th><th>L</th></tr>
      <tr><td>Chest (in)</td><td>38</td><td>40</td><td>42</td></tr>
      <tr><td>Waist (in)</td><td>32</td><td>34</td><td>36</td></tr>
    </table>"""
    parsed = parse_size_chart_html(html, Category.shirt)
    assert parsed is not None
    assert [s.label for s in parsed.sizes] == ["S", "M", "L"]
    assert parsed.unit == "in"
    got = {s.label: {m.label: m.value_min for m in s.measurements} for s in parsed.sizes}
    assert got["M"]["Chest (in)"] == 40.0 and got["L"]["Waist (in)"] == 36.0


def test_range_cells_keep_both_ends():
    html = """<h2>Size guide</h2><table>
      <tr><th>Size</th><th>To fit chest</th></tr>
      <tr><td>M</td><td>38-40</td></tr>
      <tr><td>L</td><td>42 – 44</td></tr>
    </table>"""
    parsed = parse_size_chart_html(html, Category.shirt)
    assert parsed is not None
    by_size = {s.label: s.measurements[0] for s in parsed.sizes}
    assert (by_size["M"].value_min, by_size["M"].value_max) == (38.0, 40.0)
    assert (by_size["L"].value_min, by_size["L"].value_max) == (42.0, 44.0)
    # "To fit chest" is a body chart in the page's own words.
    assert parsed.chart_type == "body"


def test_a_table_that_is_not_a_size_chart_is_ignored():
    html = """<h2>Shipping</h2><table>
      <tr><th>Region</th><th>Days</th><th>Cost</th></tr>
      <tr><td>India</td><td>3</td><td>99</td></tr>
      <tr><td>EU</td><td>7</td><td>499</td></tr>
    </table>"""
    assert parse_size_chart_html(html, Category.shirt) is None


def test_the_size_chart_wins_over_an_unrelated_table_on_the_same_page():
    html = """
    <h2>Care</h2><table><tr><th>Size</th><th>Temp</th></tr><tr><td>M</td><td>30</td></tr></table>
    <h2>Size guide</h2><table>
      <tr><th>Size</th><th>Chest (cm)</th><th>Shoulder (cm)</th></tr>
      <tr><td>S</td><td>100</td><td>42</td></tr>
      <tr><td>M</td><td>106</td><td>44</td></tr>
    </table>"""
    parsed = parse_size_chart_html(html, Category.shirt)
    assert parsed is not None
    labels = {m.label for s in parsed.sizes for m in s.measurements}
    assert any("Chest" in lab for lab in labels)
    assert not any("Temp" in lab for lab in labels)


def test_pages_without_tables_are_handled_quietly():
    assert parse_size_chart_html("", Category.shirt) is None
    assert parse_size_chart_html("<html><body><p>no tables here</p></body></html>", Category.shirt) is None
    assert parse_size_chart_html("<table><tr><td>broken", Category.shirt) is None


def test_ambiguous_body_and_garment_wording_is_left_for_normalize_to_resolve():
    html = """<h2>Size guide</h2>
    <p>To fit chest measurements below; garment measurements laid flat differ.</p>
    <table><tr><th>Size</th><th>Chest (cm)</th></tr><tr><td>M</td><td>100</td></tr>
    <tr><td>L</td><td>106</td></tr></table>"""
    parsed = parse_size_chart_html(html, Category.shirt)
    assert parsed is not None
    assert parsed.chart_type == "unknown" and parsed.chart_type_evidence is None
