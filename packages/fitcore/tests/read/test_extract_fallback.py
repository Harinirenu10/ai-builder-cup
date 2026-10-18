"""Tests for the retry in fitcore.read.extract when a call returns nothing usable.

generate_json's own Ollama-to-Gemini fallback only fires on an exception. A small local model
instead answers the schema with an empty product, which counts as success, so the fallback could
never see it -- the reader then reported a page it had captured fine as unreadable.
"""

from __future__ import annotations

import pytest

from fitcore import llm
from fitcore.read import extract as extract_mod
from fitcore.read.extract import extract
from fitcore.read.models import ExtractedProduct, PageExtraction, RawMeasure, RawSize
from fitcore.schemas import Category

EMPTY = PageExtraction(product=ExtractedProduct(title="", category=None))
GOOD = PageExtraction(product=ExtractedProduct(
    title="Oxford Button-Down Shirt", category=Category.shirt, unit="cm",
    sizes=[RawSize(label="M", measurements=[RawMeasure(label="Chest (cm)", value_min=114.0)])],
))


@pytest.fixture
def calls(monkeypatch):
    """Record every generate_json call and answer from a queued list of results."""
    recorded: list[dict] = []
    queue: list[PageExtraction] = []

    def fake_generate_json(role, parts, schema, system=None, client=None, ollama_transport=None,
                           abort_event=None, timeout=None, backend=None):
        recorded.append({"role": role, "backend": backend})
        return queue.pop(0)

    monkeypatch.setattr(llm, "generate_json", fake_generate_json)
    monkeypatch.setattr(extract_mod.llm, "generate_json", fake_generate_json)
    return recorded, queue


def test_a_usable_extraction_is_returned_without_a_second_call(calls, monkeypatch):
    recorded, queue = calls
    queue.append(GOOD)
    monkeypatch.setattr(extract_mod.llm, "get_backend", lambda: "ollama")
    monkeypatch.setattr(extract_mod.llm, "gemini_backend_for_role", lambda role: "vertex")

    got = extract("page text", None, None, "http://shop.example/p")
    assert got.product.category == Category.shirt
    assert len(recorded) == 1, "a usable result must not trigger a retry"
    assert recorded[0]["backend"] is None


def test_an_empty_extraction_is_retried_on_gemini(calls, monkeypatch):
    recorded, queue = calls
    queue.extend([EMPTY, GOOD])
    monkeypatch.setattr(extract_mod.llm, "get_backend", lambda: "ollama")
    monkeypatch.setattr(extract_mod.llm, "gemini_backend_for_role", lambda role: "vertex")

    got = extract("page text", None, None, "http://shop.example/p")
    assert got.product.category == Category.shirt, "the retry's result should be returned"
    assert len(recorded) == 2
    assert recorded[0]["backend"] is None       # first attempt: whatever is configured
    assert recorded[1]["backend"] == "vertex"   # retry: explicitly on Gemini


def test_no_retry_when_no_gemini_model_is_configured(calls, monkeypatch):
    """The real local setup: GEMINI_*_MODEL unset, so there is nowhere to fall back to."""
    recorded, queue = calls
    queue.append(EMPTY)
    monkeypatch.setattr(extract_mod.llm, "get_backend", lambda: "ollama")
    monkeypatch.setattr(extract_mod.llm, "gemini_backend_for_role", lambda role: None)

    got = extract("page text", None, None, "http://shop.example/p")
    assert got.product.category is None
    assert len(recorded) == 1, "must not spend a round trip when no Gemini is configured"


def test_no_retry_when_the_first_attempt_was_already_gemini(calls, monkeypatch):
    recorded, queue = calls
    queue.append(EMPTY)
    monkeypatch.setattr(extract_mod.llm, "get_backend", lambda: "vertex")
    monkeypatch.setattr(extract_mod.llm, "gemini_backend_for_role", lambda role: "vertex")

    extract("page text", None, None, "http://shop.example/p")
    assert len(recorded) == 1, "Gemini returning nothing must not be retried on Gemini"


def test_a_missing_size_chart_alone_is_not_treated_as_failure(calls, monkeypatch):
    """Plenty of pages genuinely have no chart; tables.py covers the ones published as HTML."""
    recorded, queue = calls
    queue.append(PageExtraction(product=ExtractedProduct(title="Tee", category=Category.tshirt)))
    monkeypatch.setattr(extract_mod.llm, "get_backend", lambda: "ollama")
    monkeypatch.setattr(extract_mod.llm, "gemini_backend_for_role", lambda role: "vertex")

    got = extract("page text", None, None, "http://shop.example/p")
    assert got.product.category == Category.tshirt and not got.product.sizes
    assert len(recorded) == 1


# ----------------------------------------------------------------- gemini_backend_for_role


def test_gemini_backend_for_role_reports_none_without_a_configured_model(monkeypatch):
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    assert llm.gemini_backend_for_role("flash") is None


def test_gemini_backend_for_role_prefers_vertex_then_the_dev_key(monkeypatch):
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "a-model-id")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    assert llm.gemini_backend_for_role("flash") == "vertex"

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "a-key")
    assert llm.gemini_backend_for_role("flash") == "gemini_api"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.gemini_backend_for_role("flash") is None


def test_generate_json_rejects_an_unknown_backend_override():
    with pytest.raises(ValueError, match="backend must be one of"):
        llm.generate_json("flash", "x", PageExtraction, backend="not-a-backend")
