import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("api_main", ROOT / "services" / "api" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return TestClient(mod.app)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for v in ("LLM_BACKEND", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "GEMINI_API_KEY"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def payload():
    return {
        "product": json.loads((ROOT / "samples" / "product_kurta.json").read_text(encoding="utf-8")),
        "twin": json.loads((ROOT / "samples" / "twin.json").read_text(encoding="utf-8")),
    }


def test_fit_returns_full_result(client, payload):
    r = client.post("/v1/fit", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["recommended_size"] == "L" and body["explanation"] is None


def test_fit_with_reviews_and_poses(client, payload):
    payload["review_signals"] = [{"zone": "chest", "direction": "smaller", "strength": "slight"}]
    payload["pose_deltas"] = {"poses": {"seated": {"minutes": 60, "delta_cm": {"hip": 2.0}}}}
    r = client.post("/v1/fit", json=payload)
    assert r.status_code == 200 and r.json()["review_notes"]


def test_biz_verdict_has_no_measurements(client, payload):
    r = client.post("/v1/biz/verdict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_size"] == "L" and body["zones"]
    text = json.dumps(body)
    for forbidden in ("ease", "value_cm", "margin", "shrink", "target_", "_cm"):
        assert forbidden not in text
    # no body measurement numbers leak
    for v in payload["twin"]["zones"].values():
        assert f'"{v["value_cm"]}"' not in text


def test_missing_size_chart_is_422(client, payload):
    payload["product"]["size_chart"] = None
    assert client.post("/v1/fit", json=payload).status_code == 422


def test_tailor_card_english(client, payload):
    r = client.post("/v1/tailor-card?languages=en", json=payload)
    assert r.status_code == 200 and r.json()["size"] == "L" and set(r.json()["lines"]) == {"en"}


def test_tailor_card_conflict_when_no_size(client, payload):
    payload["twin"]["zones"]["chest"]["value_cm"] = 200
    assert client.post("/v1/tailor-card?languages=en", json=payload).status_code == 409
