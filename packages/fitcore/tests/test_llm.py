import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel
from fitcore.llm import validate_model_ids, generate_json


class Rec(BaseModel):
    recommended_size: str
    confidence: float = 1.0
    verdict: str = ""
    zone_deltas: dict[str, float] = {}


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
        schema=Rec,
        client=mock_client,
    )

    assert isinstance(result, Rec)
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
        schema=Rec,
        client=mock_client,
    )
    assert result.recommended_size == "L"
    assert mock_client.models.generate_content.call_count == 2


# ----------------------------------------------------------------------------- ollama (dev-only backend)


@pytest.fixture
def ollama_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma-test:12b")


def _reply(content: str) -> dict:
    return {"message": {"content": content}, "prompt_eval_count": 10, "eval_count": 5}


def test_default_backend_is_auto(monkeypatch):
    from fitcore.llm import get_backend
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert get_backend() == "auto"


def test_invalid_backend_rejected(monkeypatch):
    from fitcore.llm import get_backend
    monkeypatch.setenv("LLM_BACKEND", "openai")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        get_backend()


def test_ollama_generate_json_sends_schema_and_validates(ollama_env):
    sent = []

    def transport(payload):
        sent.append(payload)
        return _reply(json.dumps({"recommended_size": "M", "confidence": 0.7}))

    result = generate_json("flash", "hello", Rec, system="be brief", ollama_transport=transport)
    assert result.recommended_size == "M"
    p = sent[0]
    assert p["model"] == "gemma-test:12b" and p["stream"] is False
    assert p["format"]["properties"]["recommended_size"]
    assert p["messages"][0] == {"role": "system", "content": "be brief"}


def test_ollama_one_repair_retry_then_success(ollama_env):
    replies = iter([_reply("not json"), _reply(json.dumps({"recommended_size": "L"}))])
    seen = []

    def transport(payload):
        seen.append(len(payload["messages"]))
        return next(replies)

    assert generate_json("flash", "x", Rec, ollama_transport=transport).recommended_size == "L"
    assert seen == [1, 3]


def test_ollama_gives_up_after_one_repair(ollama_env):
    with pytest.raises(Exception):
        generate_json("flash", "x", Rec, ollama_transport=lambda payload: _reply("still not json"))


def _png(size=(3000, 2000)) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


def test_ollama_sends_images_base64_and_resized(ollama_env):
    import base64, io
    from PIL import Image
    sent = []

    def transport(payload):
        sent.append(payload)
        return _reply(json.dumps({"recommended_size": "M"}))

    generate_json("pro", ["describe", _png()], Rec, ollama_transport=transport)
    user = sent[0]["messages"][-1]
    assert user["content"] == "describe" and len(user["images"]) == 1
    img = Image.open(io.BytesIO(base64.b64decode(user["images"][0])))
    assert max(img.size) == 1280 and img.format == "JPEG"


def test_ollama_sends_think_false(ollama_env):
    """The configured Ollama-compatible endpoint's own guidance: send `"think": false` for
    faster responses. Must be present on every /api/chat call, not just the first of a retry."""
    sent = []

    def transport(payload):
        sent.append(payload)
        return _reply(json.dumps({"recommended_size": "M"}))

    generate_json("pro", "describe", Rec, ollama_transport=transport)
    assert sent[0]["think"] is False


def test_ollama_rejects_unsupported_parts(ollama_env):
    with pytest.raises(ValueError, match="text and image bytes"):
        generate_json("pro", ["x", 42], Rec, ollama_transport=lambda p: _reply("{}"))


def test_ollama_requires_url_and_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
        generate_json("flash", "x", Rec)


def test_ollama_retries_on_503(ollama_env, monkeypatch):
    monkeypatch.setattr("fitcore.llm.time.sleep", lambda s: None)
    calls = []

    def transport(payload):
        calls.append(1)
        if len(calls) == 1:
            raise Exception("503 ServiceUnavailable")
        return _reply(json.dumps({"recommended_size": "S"}))

    assert generate_json("flash", "x", Rec, ollama_transport=transport).recommended_size == "S"
    assert len(calls) == 2


def test_is_retryable_recognizes_timeouts():
    """Regression test: a read/connect timeout (urllib.request.urlopen's socket
    timeout on the Ollama path, or a urllib.error.URLError wrapping one) must be
    retryable. Confirmed via a real demo run: 'LLM [gemma4:12b] call failed
    permanently after 1 attempts: The read operation timed out' -- a single slow
    response was never retried because TimeoutError/socket.timeout carries none
    of the status codes or strings the retry check looked for."""
    import socket
    import urllib.error

    from fitcore.llm import _is_retryable

    assert _is_retryable(TimeoutError("The read operation timed out")) is True
    assert _is_retryable(socket.timeout("The read operation timed out")) is True
    assert _is_retryable(urllib.error.URLError(socket.timeout("timed out"))) is True
    # Must not become so broad that genuinely permanent errors retry too.
    assert _is_retryable(ValueError("LLM_BACKEND=ollama requires OLLAMA_BASE_URL and OLLAMA_MODEL.")) is False


def test_ollama_retries_on_timeout_then_succeeds(ollama_env, monkeypatch):
    """A single transient read timeout must be retried and the call must still
    succeed, instead of failing permanently after one attempt."""
    monkeypatch.setattr("fitcore.llm.time.sleep", lambda s: None)
    calls = []

    def transport(payload):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return _reply(json.dumps({"recommended_size": "M"}))

    assert generate_json("flash", "x", Rec, ollama_transport=transport).recommended_size == "M"
    assert len(calls) == 2


def test_ollama_gives_up_after_repeated_timeouts_bounded(ollama_env, monkeypatch):
    """Repeated timeouts must exhaust after the existing bounded retry count (3
    attempts) and raise -- not retry forever, and not silently hang. This is
    the 'job cannot hang indefinitely' guarantee at the LLM-client level."""
    monkeypatch.setattr("fitcore.llm.time.sleep", lambda s: None)
    calls = []

    def transport(payload):
        calls.append(1)
        raise TimeoutError("The read operation timed out")

    with pytest.raises(TimeoutError):
        generate_json("flash", "x", Rec, ollama_transport=transport)
    assert len(calls) == 3, f"expected exactly 3 bounded attempts, got {len(calls)}"


# ----------------------------------------------------------------------------- auto: Gemma first, Gemini fallback


def _gemini_client(payload: dict) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = json.dumps(payload)
    client.models.generate_content.return_value = resp
    return client


@pytest.fixture
def auto_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma-test:12b")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")


def test_auto_prefers_ollama_and_skips_gemini(auto_env):
    gem = _gemini_client({"recommended_size": "G"})
    out = generate_json("flash", "x", Rec, client=gem, ollama_transport=lambda p: _reply(json.dumps({"recommended_size": "O"})))
    assert out.recommended_size == "O" and not gem.models.generate_content.called


def test_auto_falls_back_to_gemini_when_ollama_fails(auto_env):
    def down(payload):
        raise ConnectionError("relay unreachable")

    gem = _gemini_client({"recommended_size": "G"})
    assert generate_json("flash", "x", Rec, client=gem, ollama_transport=down).recommended_size == "G"


def test_auto_sends_image_parts_to_ollama_first(auto_env):
    gem = _gemini_client({"recommended_size": "G"})
    out = generate_json("pro", ["look", _png((100, 100))], Rec, client=gem,
                        ollama_transport=lambda p: _reply(json.dumps({"recommended_size": "O"})))
    assert out.recommended_size == "O" and not gem.models.generate_content.called


def test_auto_image_falls_back_to_gemini_when_ollama_fails(auto_env, monkeypatch):
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-exp-custom")

    def down(payload):
        raise ConnectionError("relay unreachable")

    gem = _gemini_client({"recommended_size": "G"})
    assert generate_json("pro", ["look", _png((100, 100))], Rec, client=gem, ollama_transport=down).recommended_size == "G"


def test_auto_without_ollama_config_uses_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    gem = _gemini_client({"recommended_size": "G"})
    assert generate_json("flash", "x", Rec, client=gem).recommended_size == "G"


def test_gemini_api_key_calls_are_capped(monkeypatch):
    import fitcore.llm as llm
    monkeypatch.setattr(llm, "_api_key_calls", 0)
    monkeypatch.setenv("LLM_BACKEND", "gemini_api")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_API_MAX_CALLS", "2")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-exp-custom")
    gem = _gemini_client({"recommended_size": "G"})
    for _ in range(2):
        generate_json("flash", "x", Rec, client=gem)
    with pytest.raises(RuntimeError, match="GEMINI_API_MAX_CALLS"):
        generate_json("flash", "x", Rec, client=gem)
    assert gem.models.generate_content.call_count == 2


def test_gemini_api_backend_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gemini_api")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        generate_json("flash", "x", Rec, client=_gemini_client({}))


def test_gemini_path_wraps_image_bytes_as_inline_parts(monkeypatch):
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-exp-custom")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    gem = _gemini_client({"recommended_size": "G"})
    generate_json("pro", ["look", _png((64, 64))], Rec, client=gem)
    contents = gem.models.generate_content.call_args.kwargs["contents"]
    assert contents[0] == "look"
    inline = contents[1].inline_data
    assert inline.mime_type == "image/png" and inline.data[:4] == b"\x89PNG"
