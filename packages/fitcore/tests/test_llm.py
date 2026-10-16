import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fitcore.schemas import FitResult
from fitcore.llm import validate_model_ids, generate_json


def test_validate_model_ids_rejects_gemini_1_and_2(monkeypatch):
    prohibited_models = [
        ("GEMINI_PRO_MODEL", "gemini-1.5-pro"),
        ("GEMINI_FLASH_MODEL", "gemini-2.0-flash"),
        ("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"),
        ("VTO_MODEL", "gemini-1.0-ultra"),
    ]
    for env_var, model_name in prohibited_models:
        monkeypatch.setenv(env_var, model_name)
        with pytest.raises(ValueError, match="strictly forbids gemini-1.x and gemini-2.x"):
            validate_model_ids()
        monkeypatch.delenv(env_var, raising=False)


def test_validate_model_ids_allows_clean_environment(monkeypatch):
    for var in ("GEMINI_PRO_MODEL", "GEMINI_FLASH_MODEL", "GEMINI_IMAGE_MODEL", "VTO_MODEL"):
        monkeypatch.delenv(var, raising=False)
    # Should not raise
    validate_model_ids()


def test_import_with_gemini_2_5_raises(monkeypatch):
    """Confirm Acceptance Check 5: importing fitcore.llm with a gemini-2.5 model in env raises a clear error."""
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
    import fitcore.llm
    with pytest.raises(ValueError, match="strictly forbids gemini-1.x and gemini-2.x"):
        importlib.reload(fitcore.llm)


def test_generate_json_with_recorded_fixture(monkeypatch):
    """Test generate_json offline with mock client and recorded fixture."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")

    fixture_path = Path(__file__).parent / "fixtures" / "llm" / "recommendation_fixture.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture_data = json.load(f)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(fixture_data)
    mock_response.usage_metadata.prompt_token_count = 120
    mock_response.usage_metadata.candidates_token_count = 45
    mock_client.models.generate_content.return_value = mock_response

    result = generate_json(
        role="flash",
        parts="Evaluate fit for chest=100cm",
        schema=FitResult,
        client=mock_client,
    )

    assert isinstance(result, FitResult)
    assert result.recommended_size == "M"
    assert result.confidence == 0.94
    assert mock_client.models.generate_content.called


def test_generate_json_retry_on_429(monkeypatch):
    """Verify exponential backoff retry on 429."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({"recommended_size": "L", "confidence": 0.8, "verdict": "Loose fit", "zone_deltas": {}})
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 20

    rate_limit_error = Exception("429 ResourceExhausted: rate limit exceeded")
    mock_client.models.generate_content.side_effect = [rate_limit_error, mock_response]

    result = generate_json(
        role="flash",
        parts="Test retry",
        schema=FitResult,
        client=mock_client,
    )
    assert result.recommended_size == "L"
    assert mock_client.models.generate_content.call_count == 2
