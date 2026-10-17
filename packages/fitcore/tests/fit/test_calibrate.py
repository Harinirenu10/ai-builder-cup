import json
from unittest.mock import MagicMock

import pytest

from fitcore.fit.calibrate import CalibrationChart, calibrate, chart_eases, extract, propose
from fitcore.schemas import MeasurementZone as Z


def chart(src="a", unit="cm", basis="full", sizes=None, cat="kurta", fit="regular"):
    sizes = sizes or {"S": ({"chest": 90}, {"chest": 104}), "M": ({"chest": 96}, {"chest": 110})}
    return CalibrationChart(category=cat, fit_type=fit, unit=unit, garment_basis=basis, source=src,
                            sizes={k: {"body": b, "garment": g} for k, (b, g) in sizes.items()})


def test_ease_is_garment_minus_body():
    assert chart_eases(chart())[Z.chest] == [14.0, 14.0]


def test_inches_converted():
    assert chart_eases(chart(unit="in", sizes={"M": ({"chest": 36}, {"chest": 42})}))[Z.chest] == [15.24]


def test_flat_circumference_doubled_but_not_length():
    c = chart(basis="flat", sizes={"M": ({"chest": 96, "torso_length": 45}, {"chest": 55, "garment_length": 90})})
    e = chart_eases(c)
    assert e[Z.chest] == [14.0] and e[Z.garment_length] == [45.0]   # length compared with body torso_length


def test_zone_without_body_value_skipped():
    assert chart_eases(chart(sizes={"M": ({}, {"chest": 110})})) == {}


@pytest.mark.parametrize("xs,expected", [([10.0], (10.0, 10.0, 10.0)), ([8, 10, 14], (8, 10, 14)), ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (1.9, 5.5, 9.1))])
def test_propose(xs, expected):
    assert propose(xs) == pytest.approx(expected, abs=0.05)


def test_needs_samples_from_two_products():
    rows = calibrate([chart("a"), chart("a")])["rows"]          # same product twice
    assert not rows[0]["enough_data"]
    res = calibrate([chart("a"), chart("b")])
    assert res["rows"][0]["enough_data"]
    assert res["proposed_yaml"]["target_ease_cm"]["kurta"]["regular"]["chest"] == [14.0, 14.0, 14.0]


def test_does_not_modify_config():
    from fitcore.fit.config import get_config
    before = get_config().target_ease_cm["kurta"]["regular"][Z.chest].ideal
    calibrate([chart("a"), chart("b")])
    assert get_config().target_ease_cm["kurta"]["regular"][Z.chest].ideal == before


def test_extract_from_text_uses_llm_and_sets_source(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "t")
    page = tmp_path / "chart.html"
    page.write_text("<html><script>x()</script><table><tr><td>M</td><td>96</td></tr></table></html>", encoding="utf-8")
    draft = {"category": "kurta", "fit_type": "regular", "unit": "cm", "sizes": {"M": {"body": {"chest": 96}, "garment": {"chest": 110}}}}
    client = MagicMock()
    client.models.generate_content.return_value.text = json.dumps(draft)
    import fitcore.llm as llm
    monkeypatch.setattr(llm, "get_client", lambda use_api_key=False: client)
    out = extract(page)
    assert out.source == "extracted from chart.html" and out.sizes["M"].garment[Z.chest] == 110
    sent = client.models.generate_content.call_args.kwargs["contents"][0]
    assert "x()" not in sent and "96" in sent
