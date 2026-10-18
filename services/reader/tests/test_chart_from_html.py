"""The reader recovers a size chart from the page's HTML when the model returns none.

This is what stopped the demo store's shirt from intermittently coming back as
"no_size_chart" (HTTP 422 from the API's /v1/me/fit-from-url) even though the capture had the
chart's numbers verbatim in its text: the chart is published as a real HTML table, so it does
not need a model to be read.
"""

from __future__ import annotations

import asyncio

import pytest

from readerlib import pipeline as pipeline_mod
from readerlib.cache import ReadCache
from readerlib.capture import Capture
from readerlib.pipeline import _fill_chart_from_html, read_url

from fitcore.read.models import ExtractedProduct, PageExtraction, RawMeasure, RawSize
from fitcore.schemas import Category

CHART_HTML = """
<html><body><h1>Oxford Button-Down Shirt</h1>
<div id="chart-modal"><h2>Size guide: garment measurements</h2>
<table>
  <tr><th>Size</th><th>Chest (cm)</th><th>Shoulder (cm)</th></tr>
  <tr><td>S</td><td>108</td><td>44</td></tr>
  <tr><td>M</td><td>114</td><td>46</td></tr>
  <tr><td>L</td><td>120</td><td>48</td></tr>
</table></div></body></html>
"""


def _capture(html: str = CHART_HTML) -> Capture:
    cap = Capture(status="ok", final_url="http://localhost/store/shirt.html", http_status=200)
    cap.text = "Oxford Button-Down Shirt. Size guide: garment measurements."
    cap.html = html
    return cap


def _extraction(with_sizes: bool = False) -> PageExtraction:
    sizes = (
        [RawSize(label="M", measurements=[RawMeasure(label="Chest", value_min=999.0)])] if with_sizes else []
    )
    return PageExtraction(product=ExtractedProduct(title="Oxford Button-Down Shirt",
                                                   category=Category.shirt, sizes=sizes))


def test_chart_is_filled_from_html_when_the_model_returned_none():
    ext = _extraction()
    assert _fill_chart_from_html(ext, _capture()) is True
    assert [s.label for s in ext.product.sizes] == ["S", "M", "L"]
    assert ext.product.unit == "cm"
    assert ext.product.chart_type == "garment"
    assert ext.product.chart_type_evidence


def test_a_chart_the_model_did_return_is_never_overwritten():
    ext = _extraction(with_sizes=True)
    assert _fill_chart_from_html(ext, _capture()) is False
    assert ext.product.sizes[0].measurements[0].value_min == 999.0


def test_no_html_or_no_table_changes_nothing():
    ext = _extraction()
    assert _fill_chart_from_html(ext, _capture(html="")) is False
    assert _fill_chart_from_html(ext, _capture(html="<p>no tables</p>")) is False
    assert ext.product.sizes == []


def test_read_url_returns_a_usable_product_and_flags_the_provenance(monkeypatch):
    """End to end through read_url with a stubbed capture and a model that returns no chart."""
    monkeypatch.setattr(pipeline_mod, "extract", lambda *a, **k: _extraction())
    resp = asyncio.run(read_url(
        "http://localhost/store/shirt.html", ReadCache(), capture_fn=lambda url: _capture()
    ))

    assert resp.status == "ok" and resp.product is not None
    assert resp.product.size_chart is not None, "the HTML chart should have been recovered"
    assert sorted(resp.product.size_chart.sizes) == ["L", "M", "S"]
    assert resp.product.size_chart.sizes["M"] and resp.product.size_chart.chart_type == "garment"
    assert "size_chart_from_html" in resp.flags
    assert "no_size_chart" not in resp.flags


def test_read_url_without_a_recoverable_chart_still_reports_no_size_chart(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "extract", lambda *a, **k: _extraction())
    resp = asyncio.run(read_url(
        "http://localhost/store/plain.html", ReadCache(),
        capture_fn=lambda url: _capture(html="<p>no tables here</p>"),
    ))
    assert resp.status == "ok"
    assert resp.product is not None and resp.product.size_chart is None
    assert "size_chart_from_html" not in resp.flags


def test_a_failed_extraction_is_not_cached_as_unsupported(monkeypatch):
    """A transient model failure must not pin "not a garment" to the URL for 7 days.

    assemble() returns unsupported whenever category is None, and read_url caches that like any
    other result. With the cache now on disk (readerlib/cache.py), one bad extraction would make
    the demo store's shirt unreadable for a week -- even though its own garment size chart had
    just been parsed out of the page.
    """
    no_category = PageExtraction(product=ExtractedProduct(title="Oxford Button-Down Shirt", category=None))
    monkeypatch.setattr(pipeline_mod, "extract", lambda *a, **k: no_category)
    cache = ReadCache()
    url = "http://localhost/store/shirt.html"

    resp = asyncio.run(read_url(url, cache, capture_fn=lambda u: _capture()))

    assert resp.status == "error", "a missing category with a parsed garment chart is a failed read"
    assert "extraction_incomplete" in (resp.reason or "")
    assert cache.get(url) is None, "a failed read must not be cached"


def test_a_page_that_really_is_not_a_garment_is_still_cached_as_unsupported(monkeypatch):
    """The genuine case must keep working: the model read the page and it is not a garment."""
    toaster = PageExtraction(product=ExtractedProduct(title="Two-Slice Toaster", category=None))
    monkeypatch.setattr(pipeline_mod, "extract", lambda *a, **k: toaster)
    cache = ReadCache()
    url = "http://localhost/store/toaster.html"

    resp = asyncio.run(read_url(url, cache, capture_fn=lambda u: _capture(html="<p>no tables</p>")))

    assert resp.status == "unsupported"
    assert cache.get(url) is not None, "a real unsupported verdict should still be cached"
