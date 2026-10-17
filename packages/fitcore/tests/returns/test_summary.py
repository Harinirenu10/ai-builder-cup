import json
from unittest.mock import MagicMock

import pytest

from fitcore.returns.summary import BANNED, acceptable, case_facts, summarize, template_summary

FACTS = case_facts(product_title="Oxford Button-Down Shirt", ordered_size="M", reason="too_small", guarantee=True, recommended_size="M",
                   confidence=0.9, routing_route="auto_exchange", routing_status="approved_exchange",
                   decision_log=["Rule 1: the garment does not match its label."], spec_cm={"chest": 114.0}, measured_cm={"chest": 108.2},
                   has_note=True, days_since_order=3)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "t")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")


def client_returning(text: str) -> MagicMock:
    c = MagicMock()
    c.models.generate_content.return_value.text = json.dumps({"text": text})
    return c


GOOD = "The M shirt was reported too small. The guarantee applies, and the unboxing measurement is 5.8 cm under the size M chest spec of 114.0 cm, so the exchange was approved."


def test_facts_hold_numbers_and_no_free_text():
    assert FACTS["measured_minus_spec_cm"] == {"chest": -5.8} and FACTS["shopper_left_a_note"] is True
    assert "note" not in json.dumps(FACTS).replace("shopper_left_a_note", "")


def test_good_model_output_is_used():
    text, source = summarize(FACTS, client=client_returning(GOOD))
    assert source == "model" and text == GOOD


@pytest.mark.parametrize("bad", [
    "This looks like fraud by the shopper.", "The customer is lying about the size.", "Suspicious return, likely a scam.",
    "They are cheating the guarantee.", "The shopper is to blame.", "A fake claim.", "This is abuse of the policy.",
    "The garment would not flatter a heavier build.", "A dishonest report.",
])
def test_accusatory_or_body_judging_output_falls_back_to_template(bad):
    text, source = summarize(FACTS, client=client_returning(bad))
    assert source == "template" and not BANNED.search(text) and text == template_summary(FACTS)


def test_invented_numbers_fall_back():
    text, source = summarize(FACTS, client=client_returning("The shirt is 9.9 cm too small."))
    assert source == "template"


def test_empty_output_and_model_errors_fall_back():
    assert summarize(FACTS, client=client_returning("  "))[1] == "template"
    broken = MagicMock()
    broken.models.generate_content.side_effect = ValueError("boom")
    assert summarize(FACTS, client=broken)[1] == "template"


def test_template_itself_is_neutral_for_every_shape_of_facts():
    for guarantee in (True, False):
        for measured in (None, {"chest": 120.0}):
            f = case_facts(product_title="Tee", ordered_size="L", reason="wrong_size_received", guarantee=guarantee, recommended_size="L" if guarantee else None,
                           confidence=0.8 if guarantee else None, routing_route="human_review", routing_status="pending_review", decision_log=[],
                           spec_cm={"chest": 118.0}, measured_cm=measured, has_note=False, days_since_order=2)
            assert not BANNED.search(template_summary(f)) and acceptable(template_summary(f), f)


@pytest.mark.parametrize("word", ["fraud", "fraudulent", "lying", "liar", "lie", "lies", "cheating", "scam", "dishonest", "suspicious", "fake", "abuse", "blame", "accused", "guilty", "weight", "flattering"])
def test_banned_list_catches_the_words_the_brief_names_and_more(word):
    assert BANNED.search(f"the shopper is {word}")


def test_neutral_words_are_not_banned():
    assert not BANNED.search("The garment measures 5.8 cm under the ordered size spec; the exchange was approved by rule 1.")


def test_recorded_real_gemini_summary_passes_the_guardrails():
    """A response recorded from the live flash model (tests/fixtures/llm/returns_summary.json) is neutral and uses only the facts."""
    from pathlib import Path
    fx = json.loads((Path(__file__).parent.parent / "fixtures" / "llm" / "returns_summary.json").read_text(encoding="utf-8"))
    text, source = summarize(fx["facts"], client=client_returning(fx["response"]["text"]))
    assert source == "model" and text == fx["response"]["text"]
    assert not BANNED.search(text)
