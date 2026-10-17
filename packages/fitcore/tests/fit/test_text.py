"""reviews.apply, explain() and tailor_card() — offline, Gemini mocked with recorded/derived fixtures."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from fitcore.fit import reviews
from fitcore.fit.engine import evaluate
from fitcore.fit.explain import explain
from fitcore.fit.tailor import render_png, tailor_card
from fitcore.schemas import ReviewSignal, TailorLines
from tests.fit.helpers import Z, body, product

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")


def mock_client(*texts: str) -> MagicMock:
    client = MagicMock()
    responses = []
    for t in texts:
        r = MagicMock()
        r.text = t
        responses.append(r)
    client.models.generate_content.side_effect = responses
    return client


# ----------------------------------------------------------------------------- reviews


def test_reviews_plain_signal_and_cap():
    sigs = [ReviewSignal(zone=Z.chest, direction="smaller", strength="strong")] * 3
    adj, notes = reviews.apply(sigs, body(chest=96))
    assert adj.cm[Z.chest] == 4.0 and len(notes) == 3  # 3 x 3.0 capped at 4.0


def test_reviews_larger_gives_negative_adjustment():
    adj, _ = reviews.apply([ReviewSignal(zone=Z.waist, direction="larger", strength="moderate")], body(waist=80))
    assert adj.cm[Z.waist] == pytest.approx(-1.5)


@pytest.mark.parametrize("torso,applies", [(52.0, True), (50.0, True), (49.9, False), (44.0, False)])
def test_reviews_long_torso_context_only_when_twin_matches(torso, applies):
    sig = ReviewSignal(zone=Z.garment_length, direction="smaller", strength="moderate", context="long_torso")
    adj, notes = reviews.apply([sig], body(torso_length=torso))
    assert (Z.garment_length in adj.cm) is applies
    assert notes and ("Skipped" in notes[0]) is (not applies)


def test_reviews_context_without_twin_measurement_is_skipped():
    sig = ReviewSignal(zone=Z.garment_length, direction="smaller", context="long_torso")
    adj, notes = reviews.apply([sig], body(chest=96))
    assert adj.cm == {} and "no torso length" in notes[0]


# ----------------------------------------------------------------------------- explain


@pytest.fixture
def kurta_fit():
    b = body(chest=96, waist=82, hip=98, shoulder_width=42, sleeve_length=58, torso_length=46)
    sizes = {"L": {"chest": 115.5, "waist": 110, "hip": 116, "shoulder_width": 45.5, "sleeve_length": 60.5, "garment_length": 92}}
    return evaluate(b, product("kurta", sizes, fabric=[{"fiber": "rayon", "percent": 100}]))


def test_explain_uses_recorded_fixture(kurta_fit):
    client = mock_client((FIXTURES / "explain_kurta.json").read_text(encoding="utf-8"))
    out = explain(kurta_fit, "en", client=client)
    assert out.startswith("Size L") and "19.5" in out
    sent = client.models.generate_content.call_args.kwargs["contents"][0]
    assert "explanation" not in sent and json.loads(sent)["facts"] == kurta_fit.explanation_facts


@pytest.mark.parametrize("bad", [
    "This cut is flattering on most figures.",
    "Hides problem areas well.",
    "The chest has 99.0 cm of ease.",          # number not in the facts
    "   ",
])
def test_explain_falls_back_to_facts_when_guardrails_trip(kurta_fit, bad):
    out = explain(kurta_fit, "en", client=mock_client(json.dumps({"text": bad})))
    assert out == " ".join(kurta_fit.explanation_facts[:3])


def test_explain_empty_facts_makes_no_call():
    from fitcore.schemas import FitResult
    client = mock_client()
    assert explain(FitResult(category="kurta", fit_type="regular", sizes=[]), client=client) == ""
    assert not client.models.generate_content.called


# ----------------------------------------------------------------------------- tailor card


@pytest.fixture
def shirt_fit():
    b = body(chest=96, shoulder_width=42, sleeve_length=58, waist=82, torso_length=46, hip=98)
    sizes = {"M": {"chest": 112, "shoulder_width": 44, "sleeve_length": 61.5, "waist": 98, "garment_length": 72}}
    return b, evaluate(b, product("shirt", sizes))


def _translate(lines: TailorLines, **override) -> str:
    out = TailorLines(
        title="தையல் அட்டை",
        size_line=lines.size_line.replace("Size to buy", "வாங்க வேண்டிய அளவு"),
        alteration_lines=[l.replace("Shorten sleeves", "கைகளைக் குறைக்க") for l in lines.alteration_lines],
        measurement_lines=lines.measurement_lines,
    ).model_dump()
    out.update(override)
    return json.dumps(out, ensure_ascii=False)


def test_tailor_card_content_and_translation(shirt_fit):
    b, fit = shirt_fit
    en_probe = tailor_card(fit, b, ["en"])
    en = en_probe.lines["en"]
    card = tailor_card(fit, b, ["en", "ta"], client=mock_client(_translate(en)))
    assert card.size == "M" and set(card.lines) == {"en", "ta"}
    assert "Shorten sleeves 2.5 cm" in en.alteration_lines
    assert card.lines["ta"].alteration_lines[0].endswith("2.5 cm")
    # only what a tailor needs: sleeve (altered) and the hard zones; no hip, torso, waist
    assert set(card.measurements) == {Z.sleeve_length, Z.shoulder_width, Z.chest}
    assert card.measurements[Z.sleeve_length] == 58.0
    assert json.loads(card.model_dump_json())["size"] == "M"


@pytest.mark.parametrize("override", [
    {"alteration_lines": ["கைகளைக் குறைக்க 3 cm"]},            # number changed
    {"alteration_lines": ["கைகளைக் குறைக்க 2.5 செ.மீ"]},        # unit changed
    {"alteration_lines": []},                                      # line dropped
    {"size_line": "வாங்க வேண்டிய அளவு: L"},                       # size label changed
])
def test_tailor_translation_that_changes_numbers_units_or_size_is_dropped(shirt_fit, override):
    b, fit = shirt_fit
    en = tailor_card(fit, b, ["en"]).lines["en"]
    card = tailor_card(fit, b, ["en", "ta"], client=mock_client(_translate(en, **override)))
    assert set(card.lines) == {"en"} and card.notes


def test_tailor_translation_failure_keeps_english_card(shirt_fit):
    b, fit = shirt_fit
    client = MagicMock()
    client.models.generate_content.side_effect = ValueError("boom")
    card = tailor_card(fit, b, ["en", "ta"], client=client)
    assert set(card.lines) == {"en"} and "Translation to ta unavailable." in card.notes


def test_tailor_card_refuses_when_no_size():
    b = body(chest=100)
    fit = evaluate(b, product("tshirt", {"S": {"chest": 90}, "M": {"chest": 94}}))
    with pytest.raises(ValueError):
        tailor_card(fit, b)


def test_render_png(tmp_path, shirt_fit):
    b, fit = shirt_fit
    en = tailor_card(fit, b, ["en"]).lines["en"]
    card = tailor_card(fit, b, ["en", "ta"], client=mock_client(_translate(en)))
    path = render_png(card, tmp_path / "card.png")
    img = Image.open(path)
    assert img.format == "PNG" and img.width == 900 and img.height > 300
