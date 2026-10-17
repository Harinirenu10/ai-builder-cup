"""The reader can read the six mock storefront pages (web/store) over HTTP, widget script and all."""

import asyncio
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fitcore.fit import evaluate, from_review_signals
from fitcore.schemas import Measurement, Measurements, MeasurementZone as Z, Product
from readerlib.cache import MemoryStore, ReadCache
from readerlib.capture import capture_page
from readerlib.compare import STORE, compare_chart, load_truth
from readerlib.pipeline import read_url
from readerlib.replay import ReplayClient, fixture_path

TRUTH = load_truth(STORE)
NAMES = sorted(TRUTH)


@pytest.fixture(scope="session")
def web_server():
    """Serves web/ so that /store/kurta.html finds ../widget/fit-widget.js, as on the hosted site."""
    quiet = type("Quiet", (SimpleHTTPRequestHandler,), {"log_message": lambda *a, **k: None})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(quiet, directory=str(STORE.parent)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_six_products_with_recorded_responses():
    assert len(NAMES) == 6
    assert {TRUTH[n]["category"] for n in NAMES} == {"kurta", "shirt", "trousers", "tshirt", "jacket", "blouse"}
    assert all(fixture_path(n).is_file() for n in NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_page_is_readable_chart_opens_and_values_match(need_chromium, web_server, name):
    slug = name.removeprefix("store_")
    url = f"{web_server}/store/{slug}.html"
    cap = capture_page(url)
    assert cap.status == "ok" and cap.chart_opened and cap.chart_png and cap.json_ld
    assert "Size chart (opened):" in cap.text and "Customer reviews" in cap.text

    resp = asyncio.run(read_url(url, ReadCache(MemoryStore()), client=ReplayClient(fixture_path(name))))
    assert resp.status == "ok", resp.reason
    product = Product.model_validate(json.loads(resp.model_dump_json())["product"])
    truth = TRUTH[name]
    rows = compare_chart(product, truth)
    assert rows and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert product.category.value == truth["category"]
    assert product.size_chart.chart_type == truth["chart_type"] and product.size_chart.original_unit == truth["unit"]
    assert product.pre_shrunk == truth["pre_shrunk"]
    assert resp.reviews.reviews_read == truth["reviews_read"] and resp.reviews.zone_signals
    assert not [f for f in product.size_chart.flags if f.startswith("non_monotonic")]


def test_review_comments_become_fit_adjustments_for_a_matching_twin(need_chromium, web_server):
    """'runs small at chest' / 'shrank' on the kurta page reach the fit engine as signals."""
    name = "store_kurta"
    resp = asyncio.run(read_url(f"{web_server}/store/kurta.html", ReadCache(MemoryStore()), client=ReplayClient(fixture_path(name))))
    assert resp.reviews.shrinkage_mentions == 2
    signals = from_review_signals(resp.reviews)
    assert (Z.chest, "smaller") in {(s.zone, s.direction) for s in signals}
    twin = Measurements(zones={z: Measurement(value_cm=v, confidence=0.9, source="simulated")
                               for z, v in {Z.chest: 96, Z.waist: 82, Z.hip: 98, Z.shoulder_width: 42, Z.sleeve_length: 58, Z.torso_length: 46}.items()})
    assert evaluate(twin, resp.product).recommended_size
