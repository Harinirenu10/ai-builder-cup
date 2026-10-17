import base64
import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from fitcore.schemas import FitResult
from fitcore.llm import (
    get_provider,
    get_ollama_model,
    get_ollama_base_url,
    validate_model_ids,
    generate_json,
)


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


def test_generate_json_gemini_with_recorded_fixture(monkeypatch):
    """Test generate_json offline with Gemini mock client and recorded fixture."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
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


def test_generate_json_gemini_retry_on_429(monkeypatch):
    """Verify exponential backoff retry on 429 for Gemini."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
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


# --- Ollama Provider Tests ---


def test_provider_selection(monkeypatch):
    """Test LLM_PROVIDER environment variable resolution and defaults."""
    # Default without env var should be gemini
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_provider() == "gemini"

    # Explicit gemini
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert get_provider() == "gemini"

    # Explicit ollama
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert get_provider() == "ollama"

    # Case insensitive
    monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
    assert get_provider() == "ollama"

    # Unsupported provider raises ValueError
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_llm")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_provider()


def test_ollama_endpoint_and_model_configuration(monkeypatch):
    """Test Ollama configuration variables and defaults."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert get_ollama_base_url() == "http://localhost:11434"
    assert get_ollama_model() == "gemma4:12b"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://internal-ai-gateway.local:8080/")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b")
    assert get_ollama_base_url() == "http://internal-ai-gateway.local:8080"
    assert get_ollama_model() == "gemma4:12b"


def test_generate_json_ollama_success(monkeypatch):
    """Test successful Ollama JSON generation and schema validation."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b")

    mock_resp_payload = {
        "model": "gemma4:12b",
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "recommended_size": "M",
                "confidence": 0.95,
                "verdict": "Tailored regular fit with standard ease.",
                "zone_deltas": {"chest": 4.5},
            }),
        },
        "done": True,
        "prompt_eval_count": 80,
        "eval_count": 35,
    }

    recorded_calls = []

    def mock_ollama_client(url, payload, headers):
        recorded_calls.append({"url": url, "payload": payload, "headers": headers})
        return mock_resp_payload

    result = generate_json(
        role="flash",
        parts="Evaluate chest=98cm",
        schema=FitResult,
        client=mock_ollama_client,
    )

    assert isinstance(result, FitResult)
    assert result.recommended_size == "M"
    assert result.confidence == 0.95
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["payload"]["model"] == "gemma4:12b"
    assert call["payload"]["format"] == "json"


def test_generate_json_ollama_handles_markdown_wrapped_json(monkeypatch):
    """Test Ollama JSON generation when model returns ```json ... ``` code fence."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    raw_json = json.dumps({
        "recommended_size": "S",
        "confidence": 0.91,
        "verdict": "Slim fit.",
        "zone_deltas": {},
    })
    markdown_wrapped = f"```json\n{raw_json}\n```"

    mock_client = {
        "model": "gemma4:12b",
        "message": {
            "role": "assistant",
            "content": markdown_wrapped,
        },
        "prompt_eval_count": 50,
        "eval_count": 20,
    }

    result = generate_json(
        role="flash",
        parts="Test markdown stripping",
        schema=FitResult,
        client=mock_client,
    )

    assert result.recommended_size == "S"
    assert result.confidence == 0.91


def test_generate_json_ollama_multimodal_images(monkeypatch):
    """Verify image bytes and Part structures are encoded to base64 images array in Ollama request."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    from google.genai import types

    img1_bytes = b"fake-jpeg-image-front"
    img2_bytes = b"fake-jpeg-image-back"

    parts = [
        "Inspect photos",
        "View: front",
        types.Part.from_bytes(data=img1_bytes, mime_type="image/jpeg"),
        "View: back",
        img2_bytes,
    ]

    recorded_payloads = []

    def mock_client(url, payload, headers):
        recorded_payloads.append(payload)
        return {
            "model": "gemma4:12b",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "recommended_size": "L",
                    "confidence": 0.88,
                    "verdict": "Comfortable room.",
                    "zone_deltas": {},
                }),
            },
        }

    generate_json(
        role="flash",
        parts=parts,
        schema=FitResult,
        client=mock_client,
    )

    assert len(recorded_payloads) == 1
    req = recorded_payloads[0]
    user_msg = [m for m in req["messages"] if m["role"] == "user"][0]

    # Verify text was concatenated
    assert "Inspect photos" in user_msg["content"]
    assert "View: front" in user_msg["content"]

    # Verify both images were converted to base64
    assert len(user_msg["images"]) == 2
    assert user_msg["images"][0] == base64.b64encode(img1_bytes).decode("utf-8")
    assert user_msg["images"][1] == base64.b64encode(img2_bytes).decode("utf-8")


def test_generate_json_ollama_retry_and_auth_header(monkeypatch):
    """Verify Ollama attaches Bearer token if OLLAMA_API_KEY is set and retries on 500 error."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-test-token")

    call_count = 0
    recorded_headers = []

    def mock_client(url, payload, headers):
        nonlocal call_count
        call_count += 1
        recorded_headers.append(headers)
        if call_count == 1:
            err = Exception("500 InternalServerError: Ollama temporary error")
            err.code = 500
            raise err
        return {
            "model": "gemma4:12b",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "recommended_size": "M",
                    "confidence": 0.90,
                    "verdict": "Good fit.",
                    "zone_deltas": {},
                }),
            },
        }

    result = generate_json(
        role="flash",
        parts="Test auth and retry",
        schema=FitResult,
        client=mock_client,
    )

    assert result.recommended_size == "M"
    assert call_count == 2
    assert recorded_headers[0]["Authorization"] == "Bearer secret-test-token"


def test_generate_json_ollama_retry_on_timeout(monkeypatch):
    """Verify Ollama retries on TimeoutError / socket timeout and succeeds."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    call_count = 0

    def mock_client(url, payload, headers):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("The read operation timed out")
        return {
            "model": "gemma4:12b",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "recommended_size": "XL",
                    "confidence": 0.85,
                    "verdict": "Relaxed silhouette.",
                    "zone_deltas": {},
                }),
            },
        }

    result = generate_json(
        role="flash",
        parts="Test timeout retry",
        schema=FitResult,
        client=mock_client,
    )

    assert result.recommended_size == "XL"
    assert call_count == 2


def test_generate_json_ollama_retry_on_malformed_json(monkeypatch):
    """Verify Ollama retries when model output on attempt 1 is malformed JSON."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    call_count = 0

    def mock_client(url, payload, headers):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Broken/truncated JSON on first attempt
            return {
                "model": "gemma4:12b",
                "message": {
                    "role": "assistant",
                    "content": '{"recommended_size": "M", "confidence": ',
                },
            }
        return {
            "model": "gemma4:12b",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "recommended_size": "M",
                    "confidence": 0.92,
                    "verdict": "Valid fit on retry.",
                    "zone_deltas": {},
                }),
            },
        }

    result = generate_json(
        role="flash",
        parts="Test malformed JSON retry",
        schema=FitResult,
        client=mock_client,
    )

    assert result.recommended_size == "M"
    assert result.confidence == 0.92
    assert call_count == 2


def test_generate_json_ollama_fails_after_max_retries_on_persistent_malformed_json(monkeypatch):
    """Verify Ollama fails safely and raises after max retries when JSON remains invalid."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    call_count = 0

    def mock_client(url, payload, headers):
        nonlocal call_count
        call_count += 1
        return {
            "model": "gemma4:12b",
            "message": {
                "role": "assistant",
                "content": "This is completely invalid text, not JSON at all.",
            },
        }

    with pytest.raises(Exception):
        generate_json(
            role="flash",
            parts="Test persistent broken JSON",
            schema=FitResult,
            client=mock_client,
        )

    assert call_count == 3

