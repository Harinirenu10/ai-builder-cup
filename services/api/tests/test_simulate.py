"""Pre-production return simulation through the API."""

import json
import time
from unittest.mock import MagicMock

import pytest

import apilib.simulate as sim_module
from apilib.simulate import get_memo_writer, get_population
from fitcore.population import generate_standin
from fitcore.sim_memo import Memo, MemoSection, cites_only_the_facts, memo_facts
from tests.conftest import ROOT, user

STAFF = "staff-1"
POP = generate_standin()
SHIRT_CM = {"S": {"chest": 108, "waist": 104, "shoulder_width": 44, "sleeve_length": 60, "garment_length": 72},
            "M": {"chest": 114, "waist": 110, "shoulder_width": 46, "sleeve_length": 61.5, "garment_length": 74},
            "L": {"chest": 120, "waist": 116, "shoulder_width": 48, "sleeve_length": 63, "garment_length": 76},
            "XL": {"chest": 126, "waist": 122, "shoulder_width": 50, "sleeve_length": 64.5, "garment_length": 78}}


@pytest.fixture
def lab(client, app_module, store):
    store.update("merchants", "m_demo", {"staff_uids": [STAFF]})
    app_module.app.dependency_overrides[get_population] = lambda: POP
    return client


def post(client, path, body, uid=STAFF):
    return client.post(path, json=body, headers=user(uid))


def inline(**extra):
    return {"chart": {"sizes": SHIRT_CM, "chart_type": "garment", "unit": "cm"}, "category": "shirt", **extra}


def test_needs_a_staff_account(lab):
    assert lab.post("/v1/biz/simulate", json=inline()).status_code == 401
    assert post(lab, "/v1/biz/simulate", inline(), uid="alice").status_code == 403
    assert lab.post("/v1/biz/simulate/memo", json=inline()).status_code == 401


def test_statistics_come_back_fast_with_the_banner_and_the_population(lab):
    started = time.perf_counter()
    r = post(lab, "/v1/biz/simulate", inline())
    assert r.status_code == 200, r.text
    assert time.perf_counter() - started < 20 and r.json()["seconds"] < 20
    b = r.json()
    assert b["n_bodies"] == 4000 and b["population_source"] == "stand-in" and b["reweighted"] is False
    assert b["banner"].startswith("Synthetic population from a statistical stand-in") and b["banner"].endswith("before relying on it.")
    assert set(b["size_mix"]) == {"S", "M", "L", "XL"} and abs(sum(b["size_mix"].values()) + b["no_recommendation_share"] - 1) < 2e-3
    assert 0 <= b["no_good_size_share"] <= 1 and b["coverage"] == pytest.approx(1 - b["no_good_size_share"], abs=1e-3)
    assert set(b["zone_fit"]["M"]) == {"chest", "waist", "shoulder_width", "sleeve_length", "garment_length"}
    assert b["regrade"]["coverage_after"] >= b["regrade"]["coverage_before"]


def test_a_chart_in_inches_gives_the_same_answer_as_the_same_chart_in_cm(lab):
    in_chart = {"sizes": {s: {z: round(v / 2.54, 4) for z, v in row.items()} for s, row in SHIRT_CM.items()}, "chart_type": "garment", "unit": "in"}
    a = post(lab, "/v1/biz/simulate", inline(regrade=False)).json()
    b = post(lab, "/v1/biz/simulate", {"chart": in_chart, "category": "shirt", "regrade": False}).json()
    assert a["size_mix"] == b["size_mix"] and a["coverage"] == b["coverage"]


def test_options_target_regrade_and_fit_details(lab):
    assert post(lab, "/v1/biz/simulate", inline(target=True)).json()["reweighted"] is True
    assert post(lab, "/v1/biz/simulate", inline(regrade=False)).json()["regrade"] is None
    rayon = post(lab, "/v1/biz/simulate", inline(regrade=False, fabric=[{"fiber": "rayon", "percent": 100}], pre_shrunk=False)).json()
    plain = post(lab, "/v1/biz/simulate", inline(regrade=False, fabric=[{"fiber": "polyester", "percent": 100}])).json()
    assert rayon["size_mix"] != plain["size_mix"]                                            # washing shrinkage changes who fits


def test_a_published_studio_chart_can_be_simulated_by_id(lab, store):
    store.set("charts", "c1", {"merchant_id": "m_demo", "name": "Oxford shirt", "category": "shirt", "fit_type": "regular", "chart_type": "garment",
                               "sizes": SHIRT_CM, "version": 1, "status": "published", "live": False, "created_by": STAFF, "created_at": "2026-10-17T00:00:00+00:00"})
    by_id = post(lab, "/v1/biz/simulate", {"chart_id": "c1", "regrade": False}).json()
    direct = post(lab, "/v1/biz/simulate", inline(regrade=False)).json()
    assert by_id["size_mix"] == direct["size_mix"] and by_id["category"] == "shirt"
    store.set("charts", "c2", {"merchant_id": "m_other", "name": "x", "category": "shirt", "sizes": SHIRT_CM, "version": 1, "created_at": "2026-10-17T00:00:00+00:00"})
    assert post(lab, "/v1/biz/simulate", {"chart_id": "c2"}).status_code == 404            # another merchant's chart
    assert post(lab, "/v1/biz/simulate", {"chart_id": "nope"}).status_code == 404


@pytest.mark.parametrize("body,code", [
    ({"category": "shirt"}, "chart_required"),
    ({"chart": {"sizes": SHIRT_CM}}, "category_required"),
])
def test_missing_pieces_are_422s_with_a_code(lab, body, code):
    r = post(lab, "/v1/biz/simulate", body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == code


def test_bad_charts_are_refused(lab):
    assert post(lab, "/v1/biz/simulate", {"chart": {"sizes": {"M": {"not_a_zone": 5}}}, "category": "shirt"}).status_code == 422
    assert post(lab, "/v1/biz/simulate", {"chart": {"sizes": {}}, "category": "shirt"}).status_code == 422
    assert post(lab, "/v1/biz/simulate", {"chart": {"sizes": {f"S{i}": {"chest": 100} for i in range(20)}}, "category": "shirt"}).status_code == 422
    r = post(lab, "/v1/biz/simulate", {"chart": {"sizes": {"M": {"neck": 40}}}, "category": "tshirt"})       # a zone a t-shirt chart does not use
    assert r.status_code == 422 and r.json()["detail"]["code"] == "cannot_simulate"


def test_only_aggregates_leave_the_service(lab):
    r = post(lab, "/v1/biz/simulate", inline())
    text = r.text
    assert len(text) < 60_000
    for forbidden in ('"height', '"weights"', '"gender"', '"bodies"', '"betas"', '"torso_length"', '"inseam"'):
        assert forbidden not in text, forbidden


# ----------------------------------------------------------------------------- the memo


def fake_writer(calls):
    def write(result):
        calls.append(result)
        return Memo(headline="Shirt chart", sections=[MemoSection(heading="Coverage", body=f"The chart covers {round(result.coverage * 100, 1)}% of the bodies."),
                                                      MemoSection(heading="Caveat", body=result.banner)], source="model")
    return write


def test_the_memo_is_its_own_call_and_is_written_from_the_same_statistics(lab, app_module):
    calls = []
    app_module.app.dependency_overrides[get_memo_writer] = lambda: fake_writer(calls)
    stats = post(lab, "/v1/biz/simulate", inline()).json()
    assert calls == []                                                                      # the statistics never wait for the model
    r = post(lab, "/v1/biz/simulate/memo", inline())
    assert r.status_code == 200 and len(calls) == 1
    assert calls[0].coverage == stats["coverage"] and calls[0].size_mix == stats["size_mix"]
    assert r.json()["memo"]["source"] == "model" and r.json()["memo"]["sections"][-1]["body"] == stats["banner"]


def test_without_a_model_the_memo_is_the_template_and_still_cites_only_the_statistics(lab):
    r = post(lab, "/v1/biz/simulate/memo", inline())
    assert r.status_code == 200
    memo = r.json()["memo"]
    assert memo["source"] == "template" and memo["sections"][-1]["heading"] == "Caveat"
    facts = post(lab, "/v1/biz/simulate/facts", inline()).json()
    text = " ".join([memo["headline"]] + [s["heading"] + " " + s["body"] for s in memo["sections"] if s["heading"] != "Caveat"])
    assert cites_only_the_facts(text, facts)


def test_the_facts_endpoint_shows_exactly_what_the_memo_is_written_from(lab):
    facts = post(lab, "/v1/biz/simulate/facts", inline()).json()
    stats = post(lab, "/v1/biz/simulate", inline()).json()
    assert facts["covered_percent"] == round(stats["coverage"] * 100, 1) and facts["population"]["bodies"] == 4000


def test_the_memo_is_rate_limited_per_merchant(lab, store, app_module):
    app_module.app.dependency_overrides[get_memo_writer] = lambda: fake_writer([])
    store.update("merchants", "m_demo", {"rate_limit_per_min": 2})
    codes = [post(lab, "/v1/biz/simulate/memo", inline()).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ----------------------------------------------------------------------------- the cached population


def test_the_population_is_read_from_parquet_once_and_created_if_missing(tmp_path, monkeypatch):
    path = tmp_path / "pop.parquet"
    monkeypatch.setenv("POPULATION_PATH", str(path))
    monkeypatch.setattr(sim_module, "_population", None)
    first = get_population()
    assert path.is_file() and len(first) == 4000
    stamp = path.stat().st_mtime_ns
    assert get_population() is first                                                       # cached in the process
    monkeypatch.setattr(sim_module, "_population", None)
    again = get_population()
    assert path.stat().st_mtime_ns == stamp and len(again) == 4000                         # read back, not regenerated
    assert (again.height_cm == first.height_cm).all()
