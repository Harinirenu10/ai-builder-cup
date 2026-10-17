"""Static checks on the widget file and the mock storefront. No browser needed."""

import filecmp
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from apilib.models import FitQueryProduct
from fitcore.fit import evaluate
from fitcore.schemas import Measurements, Product
from tests.conftest import ROOT

WEB = ROOT / "web"
STORE = WEB / "store"
WIDGET = WEB / "widget" / "fit-widget.js"
SLUGS = ["kurta", "shirt", "trousers", "tshirt", "jacket", "blouse"]
sys.path.insert(0, str(STORE))


# ----------------------------------------------------------------------------- widget


def test_widget_is_under_15_kb_and_has_no_dependencies():
    src = WIDGET.read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)                       # drop the header comment
    assert WIDGET.stat().st_size < 15 * 1024, WIDGET.stat().st_size
    assert not re.search(r"(import\s*\(|require\s*\(|eval\s*\(|new Function|document\.write)", code)
    assert not re.search(r"https?://", code)                                # no hard-coded hosts: everything comes from data-* attributes
    assert "postMessage(" not in code                                        # it only listens; the consent popup is what posts


def test_widget_checks_message_origin_and_source():
    src = WIDGET.read_text(encoding="utf-8")
    assert "e.origin !== appOrigin" in src and "e.source !== this.popup" in src


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node is needed for a syntax check")
def test_widget_and_store_scripts_parse():
    import subprocess
    for f in (WIDGET, STORE / "store.js"):
        assert subprocess.run(["node", "--check", str(f)], capture_output=True).returncode == 0, f


# ----------------------------------------------------------------------------- storefront


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ld, self._in_ld, self.widget_attrs, self.script_attrs, self.rows, self.reviews, self.images = [], False, [], [], 0, 0, 0
        self.has_dialog = False
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True
        if tag == "script" and "data-merchant" in a:
            self.script_attrs.append(a)
        if "data-fit-product" in a:
            self.widget_attrs.append(a["data-fit-product"])
        if tag == "tr":
            self.rows += 1
        if tag == "div" and "review" in (a.get("class") or "").split():
            self.reviews += 1
        if tag == "img":
            self.images += 1
        if a.get("role") == "dialog":
            self.has_dialog = True

    def handle_endtag(self, tag):
        self._in_ld = False if tag == "script" else self._in_ld

    def handle_data(self, data):
        if self._in_ld:
            self.ld.append(data)
        else:
            self.text.append(data)


def parse(slug: str) -> Page:
    p = Page()
    p.feed((STORE / f"{slug}.html").read_text(encoding="utf-8"))
    return p


@pytest.mark.parametrize("slug", SLUGS)
def test_product_page_has_everything_the_brief_asks_for(slug):
    p = parse(slug)
    ld = json.loads("".join(p.ld))
    assert ld["@type"] == "Product" and ld["brand"]["name"] == "Demo Threads Co." and ld["material"]
    assert p.images >= 2                                     # front and back pictures
    assert p.has_dialog and p.rows >= 5                      # size chart in a dialog: header plus at least four sizes
    assert p.reviews >= 5
    text = " ".join(p.text)
    assert "Fabric:" in text and "SIMULATED" in text         # simulated data is labelled
    assert "Add to bag" in text and "Size guide" in text
    assert len(p.widget_attrs) == 1 and len(p.script_attrs) == 1 and p.script_attrs[0]["data-merchant"] == "pk_demo_threads"


@pytest.mark.parametrize("slug", SLUGS)
def test_widget_json_is_a_valid_fit_query_product_and_fits_the_sample_twin(slug):
    from html import unescape
    raw = parse(slug).widget_attrs[0]
    product = FitQueryProduct.model_validate(json.loads(unescape(raw)))
    twin = json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8"))
    body = Measurements.model_validate({"zones": twin["zones"]})
    result = evaluate(body, Product(id="t", brand="b", title="t", category=product.category, fit_type=product.fit_type, fabric=product.fabric,
                                    stretch_class=product.stretch_class, pre_shrunk=product.pre_shrunk, size_chart=product.size_chart))
    assert result.sizes and result.sizes[0].zones


def test_charts_mix_body_and_garment_and_exactly_one_is_in_inches():
    from html import unescape
    charts = {s: FitQueryProduct.model_validate(json.loads(unescape(parse(s).widget_attrs[0]))).size_chart for s in SLUGS}
    assert {c.chart_type for c in charts.values()} == {"body", "garment"}
    assert [s for s, c in charts.items() if c.original_unit == "in"] == ["kurta"]


def test_reviews_contain_the_zone_comments_the_reader_needs():
    all_reviews = " ".join(" ".join(parse(s).text) for s in SLUGS).lower()
    for phrase in ("runs small at the chest", "too short for my long torso", "shrank after the first wash", "sleeves are short", "waist runs small"):
        assert phrase in all_reviews, phrase


def test_six_distinct_products():
    cats = {json.loads("".join(parse(s).ld))["name"] for s in SLUGS}
    assert len(cats) == 6


def test_committed_pages_match_the_generator(tmp_path):
    """Edit build_store.py, not the HTML: regenerate and compare so the two cannot drift apart."""
    import build_store
    build_store.build(tmp_path, widget_src="../widget/fit-widget.js")   # same settings as the committed output
    for name in [f"{s}.html" for s in SLUGS] + ["index.html", "cart.html", "truth.json", "products.json"]:
        assert (tmp_path / name).read_text(encoding="utf-8") == (STORE / name).read_text(encoding="utf-8"), name
    for svg in (STORE / "img").glob("*.svg"):
        assert filecmp.cmp(svg, tmp_path / "img" / svg.name, shallow=False)
