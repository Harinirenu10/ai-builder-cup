import asyncio
import base64
import importlib.util
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fitcore.fit import apply_reviews, evaluate, from_review_signals
from fitcore.schemas import Measurement, Measurements, MeasurementZone as Z, Product
from readerlib.cache import MemoryStore, ReadCache, url_key
from readerlib.capture import Capture
from readerlib.compare import SAMPLES, compare_chart, load_truth
from readerlib.pipeline import read_image, read_url
from readerlib.replay import ReplayClient, fixture_path

SAMPLE_NAMES = sorted(load_truth())


def run(coro):
    return asyncio.run(coro)


def test_five_samples_present_with_fixtures():
    assert len(SAMPLE_NAMES) == 5
    for n in SAMPLE_NAMES:
        assert (SAMPLES / f"{n}.html").is_file() and fixture_path(n).is_file()


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_sample_gives_valid_product_json_matching_true_chart(need_chromium, server, name):
    truth = load_truth()[name]
    resp = run(read_url(f"{server}/{name}.html", ReadCache(MemoryStore()), client=ReplayClient(fixture_path(name))))
    assert resp.status == "ok", resp.reason
    product = Product.model_validate(json.loads(resp.model_dump_json())["product"])  # round-trips as valid Product JSON
    assert product.category.value
    rows = compare_chart(product, truth)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    if truth["chart_type"]:
        assert product.size_chart.chart_type == truth["chart_type"]
        assert product.size_chart.original_unit == truth["unit"]
    else:
        assert product.size_chart is None and "no_size_chart" in resp.flags
    assert product.pre_shrunk == truth["pre_shrunk"]
    assert resp.reviews.reviews_read == truth["reviews_read"]
    assert product.source_url.endswith(f"{name}.html")


def test_review_signals_flow_into_the_fit_engine(need_chromium, server):
    name = "demo_threads_denim"
    resp = run(read_url(f"{server}/{name}.html", ReadCache(MemoryStore()), client=ReplayClient(fixture_path(name))))
    signals = from_review_signals(resp.reviews)
    assert (Z.waist, "smaller") in {(s.zone, s.direction) for s in signals}
    twin = Measurements(zones={z: Measurement(value_cm=v, confidence=0.9, source="t")
                               for z, v in {Z.waist: 84, Z.hip: 100, Z.inseam: 79}.items()})
    adj, _ = apply_reviews(signals, twin)
    assert adj.cm[Z.waist] > 0
    result = evaluate(twin, resp.product, review_adjustments=adj)
    assert result.recommended_size in {s.size for s in result.sizes}


# ----------------------------------------------------------------------------- cache


class CountingCapture:
    def __init__(self, cap: Capture):
        self.cap, self.calls = cap, 0

    def __call__(self, url: str) -> Capture:
        self.calls += 1
        return self.cap


def test_second_read_is_served_from_cache_without_visiting_again():
    cache = ReadCache(MemoryStore())
    cap = CountingCapture(Capture(status="ok", text="x"))
    client = ReplayClient(fixture_path("demo_threads_tee"))
    first = run(read_url("http://127.0.0.1:1/p", cache, client=client, capture_fn=cap))
    second = run(read_url("http://127.0.0.1:1/p", cache, client=client, capture_fn=cap))
    assert first.cached is False and second.cached is True and cap.calls == 1 and client.calls == 1
    assert second.product == first.product
    assert cache.store.get(url_key("http://127.0.0.1:1/p")) is not None  # keyed by sha256(url)


def test_blocked_result_is_cached_so_the_site_is_not_hit_again():
    cache = ReadCache(MemoryStore())
    cap = CountingCapture(Capture(status="blocked", reason="captcha"))
    a = run(read_url("http://127.0.0.1:1/b", cache, capture_fn=cap))
    b = run(read_url("http://127.0.0.1:1/b", cache, capture_fn=cap))
    assert a.status == b.status == "blocked" and b.cached and cap.calls == 1


def test_errors_are_not_cached():
    cache = ReadCache(MemoryStore())
    cap = CountingCapture(Capture(status="error", reason="navigation_failed"))
    run(read_url("http://127.0.0.1:1/e", cache, capture_fn=cap))
    run(read_url("http://127.0.0.1:1/e", cache, capture_fn=cap))
    assert cap.calls == 2


def test_cache_entry_expires_after_seven_days():
    cache = ReadCache(MemoryStore())
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    cache.put("http://x/p", {"response": {"status": "blocked"}}, now=t0)
    assert cache.get("http://x/p", now=t0 + timedelta(days=6, hours=23)) is not None
    assert cache.get("http://x/p", now=t0 + timedelta(days=7, seconds=1)) is None


def test_unsafe_url_is_rejected_before_any_visit(monkeypatch):
    monkeypatch.setenv("READER_ALLOW_PRIVATE", "0")
    cap = CountingCapture(Capture(status="ok"))
    resp = run(read_url("http://169.254.169.254/latest", ReadCache(MemoryStore()), capture_fn=cap))
    assert resp.status == "error" and cap.calls == 0


def test_unsupported_category_is_reported(tmp_path):
    fx = tmp_path / "shoe.json"
    fx.write_text(json.dumps({"product": {"title": "Running shoe", "category": None}, "reviews": {}}), encoding="utf-8")
    resp = run(read_url("http://127.0.0.1:1/s", ReadCache(MemoryStore()), client=ReplayClient(fx),
                        capture_fn=CountingCapture(Capture(status="ok", text="Running shoe"))))
    assert resp.status == "unsupported" and resp.product is None


# ----------------------------------------------------------------------------- /read-image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_read_image_returns_product_without_reviews():
    resp = run(read_image(_png_b64(), "https://blocked.example/p/1", client=ReplayClient(fixture_path("demo_threads_overshirt"))))
    assert resp.status == "ok" and resp.reviews is None
    assert resp.product.size_chart.sizes["M"][Z.chest] == 112.0  # flat 56 doubled
    assert resp.product.source_url == "https://blocked.example/p/1"


def test_read_image_rejects_non_images():
    resp = run(read_image(base64.b64encode(b"not an image").decode()))
    assert resp.status == "error" and resp.reason == "not a valid image"


# ----------------------------------------------------------------------------- endpoints


@pytest.fixture
def api(monkeypatch):
    spec = importlib.util.spec_from_file_location("reader_main", Path(__file__).resolve().parents[1] / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_cache", ReadCache(MemoryStore()))
    return mod, TestClient(mod.app)


def test_read_endpoint_blocked_page(need_chromium, server, api):
    _, client = api
    r = client.post("/read", json={"url": server + "/_captcha"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "blocked" and body["reason"] == "captcha" and body["product"] is None


def test_read_endpoint_rejects_internal_url(api, monkeypatch):
    monkeypatch.setenv("READER_ALLOW_PRIVATE", "0")
    _, client = api
    assert client.post("/read", json={"url": "http://localhost/x"}).status_code == 422


def test_read_endpoint_product_and_cache(need_chromium, server, api, monkeypatch):
    mod, client = api
    real = mod.read_url
    replay = ReplayClient(fixture_path("demo_threads_tee"))
    monkeypatch.setattr(mod, "read_url", lambda url, cache: real(url, cache, client=replay))
    r = client.post("/read", json={"url": server + "/demo_threads_tee.html"})
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["product"]["category"] == "tshirt"
    assert client.post("/read", json={"url": server + "/demo_threads_tee.html"}).json()["cached"] is True


def test_read_image_endpoint(api, monkeypatch):
    mod, client = api
    real = mod.read_image
    replay = ReplayClient(fixture_path("demo_threads_denim"))
    monkeypatch.setattr(mod, "read_image", lambda image, page_url=None: real(image, page_url, client=replay))
    r = client.post("/read-image", json={"image": _png_b64()})
    assert r.status_code == 200 and r.json()["product"]["size_chart"]["chart_type"] == "body"
    assert client.post("/read-image", json={"image": "!!!"}).status_code == 422
