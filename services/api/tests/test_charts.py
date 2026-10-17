"""Size chart studio: measure, grading check with reviews, publish, export, and put on the store."""

import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from apilib.charts import get_flag_explainer, get_measurer
from apilib.stores import MemoryStore
from fitcore.chart.grading import check_grading
from fitcore.garment.landmarks import FileLandmarkFinder
from fitcore.garment.measure import measure_garment
from fitcore.garment.synthetic import render_flatlay
from fitcore.schemas import Category, MeasurementZone as Z
from tests.conftest import KEY, ORIGIN, user

STAFF = "staff-1"
SHIRT_ZONES = (Z.chest, Z.waist, Z.shoulder_width, Z.sleeve_length, Z.garment_length)
SIZES = {"S": (108, 104, 44, 60, 72), "M": (114, 110, 46, 61.5, 74), "L": (120, 116, 48, 63, 76), "XL": (126, 122, 50, 64.5, 78)}
XL_WRONG = (118, 122, 50, 64.5, 78)       # the XL sample measures 118 chest: 2 cm smaller than L


def truth(values):
    return dict(zip(SHIRT_ZONES, values))


@pytest.fixture
def photos():
    """size label -> synthetic photo. The XL is the mislabeled one."""
    out = {}
    for i, (label, vals) in enumerate({**SIZES, "XL": XL_WRONG}.items()):
        out[label] = render_flatlay(Category.shirt, truth(vals), seed=20 + i, tilt=0.03)
    return out


@pytest.fixture
def studio(client, app_module, store, photos):
    """Staff account, a measurer that places landmarks from the generator's truth (no model), and the photos by hash."""
    store.update("merchants", "m_demo", {"staff_uids": [STAFF]})
    by_hash = {hashlib.sha256(p.jpeg).hexdigest(): p for p in photos.values()}

    def measurer(image: bytes, category: Category):
        photo = by_hash[hashlib.sha256(image).hexdigest()]
        return measure_garment(image, category, finder=FileLandmarkFinder({k: list(v) for k, v in photo.landmarks.items()}))

    app_module.app.dependency_overrides[get_measurer] = lambda: measurer
    return client


def b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


def staff(client, method, path, uid=STAFF, **kw):
    return getattr(client, method)(path, headers=user(uid), **kw)


def measure(client, photo, label, category="shirt"):
    return staff(client, "post", "/v1/biz/charts/measure", json={"category": category, "size_label": label, "image": b64(photo.jpeg)})


def table_from(results: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {label: {z: v["value_cm"] for z, v in r["zones"].items()} for label, r in results.items()}


# ----------------------------------------------------------------------------- measure


def test_measure_returns_zones_confidence_and_an_overlay(studio, photos):
    r = measure(studio, photos["M"], "M")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size_label"] == "M" and set(body["zones"]) == {z.value for z in SHIRT_ZONES} and body["unsupported"] == []
    for z, v in truth(SIZES["M"]).items():
        assert abs(body["zones"][z.value]["value_cm"] - v) <= 2.5 and 0 < body["zones"][z.value]["confidence"] <= 1
    png = cv2.imdecode(np.frombuffer(base64.b64decode(body["overlay_png_base64"]), np.uint8), cv2.IMREAD_COLOR)
    assert png is not None and max(png.shape[:2]) <= 1100 and body["seconds"] >= 0


def test_measure_needs_a_staff_account_and_a_real_image(studio, photos):
    assert studio.post("/v1/biz/charts/measure", json={"category": "shirt", "size_label": "M", "image": b64(photos["M"].jpeg)}).status_code == 401
    assert measure(studio, photos["M"], "M").status_code == 200
    r = studio.post("/v1/biz/charts/measure", headers=user("alice"), json={"category": "shirt", "size_label": "M", "image": b64(photos["M"].jpeg)})
    assert r.status_code == 403
    assert staff(studio, "post", "/v1/biz/charts/measure", json={"category": "shirt", "size_label": "M", "image": "!!!"}).status_code == 422
    assert staff(studio, "post", "/v1/biz/charts/measure", json={"category": "bogus", "size_label": "M", "image": b64(photos["M"].jpeg)}).status_code == 422


def test_a_photo_without_the_sheet_is_refused_with_advice(client, store):
    """The real measurer, not the stand-in: the A4 sheet is looked for before any model is asked."""
    store.update("merchants", "m_demo", {"staff_uids": [STAFF]})
    photo = render_flatlay(Category.shirt, truth(SIZES["M"]), seed=1, with_a4=False)
    r = measure(client, photo, "M")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_a4" and "A4" in r.json()["detail"]["message"]


def test_a_failing_landmark_model_is_a_502_not_a_crash(client, app_module, store, photos):
    store.update("merchants", "m_demo", {"staff_uids": [STAFF]})

    def broken(image, category):
        raise ConnectionError("model down")

    app_module.app.dependency_overrides[get_measurer] = lambda: broken
    r = measure(client, photos["M"], "M")
    assert r.status_code == 502 and r.json()["detail"]["code"] == "measuring_failed"


# ----------------------------------------------------------------------------- grading check


def test_grading_check_flags_the_smaller_xl_with_a_sentence(studio):
    sizes = {s: {z.value: v for z, v in truth(vals).items()} for s, vals in {**SIZES, "XL": XL_WRONG}.items()}
    r = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": sizes})
    assert r.status_code == 200
    flags = r.json()["flags"]
    assert [(f["size"], f["previous_size"], f["zone"], f["kind"], f["delta_cm"]) for f in flags] == [("XL", "L", "chest", "decrease", -2.0)]
    assert flags[0]["explanation"] == "XL chest is 2.0 cm smaller than L: likely a mislabeled sample or a measuring slip." and flags[0]["explained_by"] == "template"


def test_explain_true_uses_the_model_and_explain_false_never_does(studio, app_module):
    calls = []

    def fake(flags):
        calls.append(len(flags))
        for f in flags:
            f.explanation, f.explained_by = "XL chest is 2.0 cm smaller than L, which points to a mislabeled sample.", "model"
        return flags

    app_module.app.dependency_overrides[get_flag_explainer] = lambda: fake
    sizes = {"L": {"chest": 120.0}, "XL": {"chest": 118.0}}
    quick = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": sizes}).json()["flags"]
    assert calls == [] and quick[0]["explained_by"] == "template"
    full = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": sizes, "explain": True}).json()["flags"]
    assert calls == [1] and full[0]["explained_by"] == "model"


def seed_product_page(store, url, old_sizes, directions=(("chest", "runs_small", 3),)):
    store.set("products", hashlib.sha256(url.encode()).hexdigest(), {"url": url, "fetched_at": "2026-10-17T00:00:00+00:00", "response": {
        "status": "ok", "product": {"size_chart": {"sizes": old_sizes, "chart_type": "garment", "unit": "cm"}},
        "reviews": {"reviews_read": 6, "zone_signals": [{"zone": z, "direction": d, "strength": 0.6, "mention_count": n, "reviewer_context": []} for z, d, n in directions]}}})


def test_reviews_are_shown_beside_the_chart_when_the_product_page_was_read(studio, store):
    url = "https://store.example/shirt.html"
    seed_product_page(store, url, {"S": {"chest": 110}, "M": {"chest": 116}, "L": {"chest": 122}, "XL": {"chest": 128}})
    sizes = {"S": {"chest": 108.0}, "M": {"chest": 114.0}, "L": {"chest": 120.0}, "XL": {"chest": 126.0}}
    body = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": sizes, "product_url": url}).json()
    assert body["review_source"] == "product_page" and len(body["review_notes"]) == 1
    note = body["review_notes"][0]
    assert note["text"] == "3 reviewers say it runs small at the chest; your measured chest is 2.0 cm below your old chart." and note["agreement"] == "agrees"


def test_no_stored_product_page_means_no_notes_and_the_request_can_carry_its_own(studio):
    sizes = {"M": {"chest": 114.0}, "L": {"chest": 120.0}}
    none = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": sizes, "product_url": "https://nowhere.example/x"}).json()
    assert none["review_notes"] == [] and none["review_source"] == "none"
    own = staff(studio, "post", "/v1/biz/charts/grading-check", json={
        "category": "shirt", "sizes": sizes, "old_chart": {"sizes": {"M": {"chest": 116}, "L": {"chest": 122}}, "chart_type": "garment"},
        "review_signals": {"reviews_read": 4, "zone_signals": [{"zone": "chest", "direction": "tight", "strength": 0.7, "mention_count": 2}]}}).json()
    assert own["review_source"] == "request" and "2 reviewers say it feels tight at the chest" in own["review_notes"][0]["text"]


# ----------------------------------------------------------------------------- publish, list, export


def publish(client, sizes, **extra):
    body = {"category": "shirt", "name": "Oxford Button-Down Shirt", "product_ref": "dtc-shirt", "sizes": sizes, **extra}
    return staff(client, "post", "/v1/biz/charts", json=body)


GOOD = {s: {z.value: v for z, v in truth(vals).items()} for s, vals in SIZES.items()}


def test_publish_saves_a_chart_with_a_version_and_the_flags_at_the_time(studio, store):
    first = publish(studio, GOOD, source="measured")
    assert first.status_code == 200, first.text
    c = first.json()
    assert (c["version"], c["status"], c["live"], c["chart_type"], c["created_by"], c["grading_flags"]) == (1, "published", False, "garment", STAFF, [])
    assert list(c["sizes"]) == ["S", "M", "L", "XL"] and c["sizes"]["M"]["chest"] == 114.0
    bad = {s: dict(row) for s, row in GOOD.items()}
    bad["XL"]["chest"] = 119.0
    second = publish(studio, bad, source="edited").json()
    assert second["version"] == 2 and [f["kind"] for f in second["grading_flags"]] == ["decrease"]
    assert store.get("charts", c["id"])["merchant_id"] == "m_demo"
    assert [x["version"] for x in staff(studio, "get", "/v1/biz/charts").json()] == [2, 1]


def test_publish_rejects_impossible_numbers_and_needs_staff(studio):
    assert publish(studio, {"M": {"chest": -5}}).status_code == 422
    assert publish(studio, {"M": {"chest": 900}}).status_code == 422
    assert publish(studio, {"M": {}}).status_code == 422
    assert studio.post("/v1/biz/charts", json={"category": "shirt", "name": "x", "sizes": GOOD}).status_code == 401
    assert staff(studio, "post", "/v1/biz/charts", uid="alice", json={"category": "shirt", "name": "x", "sizes": GOOD}).status_code == 403


def test_charts_belong_to_their_merchant(studio, store):
    cid = publish(studio, GOOD).json()["id"]
    store.set("merchants", "m_other", {"name": "Other", "publishable_key": "pk_o", "allowed_origins": [], "staff_uids": ["other-staff"]})
    assert staff(studio, "get", f"/v1/biz/charts/{cid}", uid="other-staff").status_code == 404
    assert staff(studio, "get", "/v1/biz/charts", uid="other-staff").json() == []
    assert staff(studio, "post", f"/v1/biz/charts/{cid}/use-on-store", uid="other-staff").status_code == 404


def test_export_json_and_html_in_cm_and_inches(studio):
    cid = publish(studio, GOOD, name="Shirt <b>&</b>").json()["id"]
    cm = staff(studio, "get", f"/v1/biz/charts/{cid}/export").json()
    assert cm["unit"] == "cm" and cm["json"]["sizes"]["M"]["chest"] == 114.0 and cm["json"]["chart_type"] == "garment" and cm["json"]["product_ref"] == "dtc-shirt"
    assert "<table" in cm["html"] and "Chest (cm)" in cm["html"] and "<td>114</td>" in cm["html"] and 'scope="row">XL<' in cm["html"]
    assert "<b>" not in cm["html"] and "&lt;b&gt;" in cm["html"]                                 # the name is escaped
    inch = staff(studio, "get", f"/v1/biz/charts/{cid}/export?unit=in").json()
    assert inch["json"]["sizes"]["M"]["chest"] == pytest.approx(114 / 2.54, abs=0.05) and "Chest (in)" in inch["html"]


# ----------------------------------------------------------------------------- the store


def store_get(client, ref="dtc-shirt", key=KEY, origin=ORIGIN):
    return client.get(f"/v1/store/charts/{ref}", headers={"X-Merchant-Key": key, "Origin": origin})


def test_use_on_store_makes_the_chart_live_and_the_store_can_read_it(studio):
    assert store_get(studio).status_code == 204                                                  # nothing live yet: empty, not an error
    a = publish(studio, GOOD).json()
    assert store_get(studio).status_code == 204                                                  # published is not live
    live = staff(studio, "post", f"/v1/biz/charts/{a['id']}/use-on-store").json()
    assert live["live"] is True
    got = store_get(studio).json()
    assert got["chart_id"] == a["id"] and got["chart_type"] == "garment" and got["unit"] == "cm" and got["sizes"]["L"]["chest"] == 120.0

    newer = {s: {**row, "chest": row["chest"] + 1} for s, row in GOOD.items()}
    b = publish(studio, newer).json()
    staff(studio, "post", f"/v1/biz/charts/{b['id']}/use-on-store")
    assert store_get(studio).json()["chart_id"] == b["id"] and store_get(studio).json()["sizes"]["L"]["chest"] == 121.0
    assert staff(studio, "get", f"/v1/biz/charts/{a['id']}").json()["live"] is False             # only one live chart per product
    staff(studio, "post", f"/v1/biz/charts/{b['id']}/remove-from-store")
    assert store_get(studio).status_code == 204


def test_use_on_store_needs_a_product_ref(studio):
    cid = publish(studio, GOOD, product_ref=None).json()["id"]
    r = staff(studio, "post", f"/v1/biz/charts/{cid}/use-on-store")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "no_product_ref"


def test_the_store_endpoint_checks_the_merchant_key_and_origin(studio):
    cid = publish(studio, GOOD).json()["id"]
    staff(studio, "post", f"/v1/biz/charts/{cid}/use-on-store")
    assert store_get(studio, origin="https://evil.example").status_code == 403
    assert store_get(studio, key="pk_nope").status_code == 401
    assert studio.get("/v1/store/charts/dtc-shirt").status_code == 401
    other = studio.get("/v1/store/charts/dtc-shirt", headers={"X-Merchant-Key": "pk_test_other", "Origin": "https://other.example"})
    assert other.status_code == 204                                                              # another merchant has no such live chart


def test_the_store_response_carries_no_body_data(studio):
    cid = publish(studio, GOOD).json()["id"]
    staff(studio, "post", f"/v1/biz/charts/{cid}/use-on-store")
    text = store_get(studio).text
    assert "twin" not in text and "owner" not in text and "uid" not in text and "created_by" not in text


# ----------------------------------------------------------------------------- the whole studio flow


def test_four_photos_become_a_published_chart_in_under_a_minute_with_the_anomaly_flagged(studio, photos, store):
    t0 = time.monotonic()
    labels = ["S", "M", "L", "XL"]
    with ThreadPoolExecutor(max_workers=4) as pool:                                              # the studio measures all sizes at once
        replies = list(pool.map(lambda s: measure(studio, photos[s], s), labels))
    assert [r.status_code for r in replies] == [200] * 4, [r.text for r in replies if r.status_code != 200]
    measured = table_from({s: r.json() for s, r in zip(labels, replies)})

    check = staff(studio, "post", "/v1/biz/charts/grading-check", json={"category": "shirt", "sizes": measured}).json()
    flagged = {(f["size"], f["zone"], f["kind"]) for f in check["flags"]}
    assert ("XL", "chest", "decrease") in flagged, check["flags"]                                # the planted mislabeled sample is caught

    chart = publish(studio, measured, source="measured").json()
    staff(studio, "post", f"/v1/biz/charts/{chart['id']}/use-on-store")
    assert time.monotonic() - t0 < 60
    assert chart["version"] == 1 and ("XL", "chest", "decrease") in {(f["size"], f["zone"], f["kind"]) for f in chart["grading_flags"]}
    live = store_get(studio).json()
    assert list(live["sizes"]) == labels
    for label, vals in {**SIZES, "XL": XL_WRONG}.items():
        for z, v in truth(vals).items():
            assert abs(live["sizes"][label][z.value] - v) <= 2.5
    assert check_grading(Category.shirt, {s: {Z(z): v for z, v in row.items()} for s, row in live["sizes"].items()})   # the live chart still has the flag
