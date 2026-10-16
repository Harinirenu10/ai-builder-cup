"""Gemini LLM wrapper for Google Cloud Enterprise Agent Platform (Vertex AI).

All Gemini calls across the system MUST route through this module.
Hard constraints:
- genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location="global")
- Model IDs from env vars only: GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_IMAGE_MODEL, VTO_MODEL
- Never hardcode model IDs. Never allow any gemini-1.x or gemini-2.x model.
- Structured JSON output via pydantic schemas.
- 3 retries with exponential backoff on 429 and 5xx errors.
- Logging of model ID, latency, and token counts.

Backends (LLM_BACKEND, default "auto"):
- auto: calls go to Ollama/Gemma (OLLAMA_BASE_URL, OLLAMA_MODEL) when configured, including image
  parts (sent as base64, resized to OLLAMA_MAX_IMAGE_PX, default 1280); if it is not configured or
  fails, Gemini is used.
- ollama: Ollama only (dev). vertex: Vertex AI only. gemini_api: Gemini Developer API key only.
Gemini uses the Developer API key (GEMINI_API_KEY) when set, else Vertex. The key is dev-only and is
used sparingly: every key call is logged and capped per process (GEMINI_API_MAX_CALLS, default 30).
Secrets come from the environment or a gitignored .env, never from code or committed files.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Any, Literal, TypeVar
from pydantic import BaseModel

logger = logging.getLogger("fitcore.llm")


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the nearest .env (gitignored) without overriding real env vars."""
    from pathlib import Path

    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


_load_dotenv()

# Regulated environment variables containing model IDs
MODEL_ENV_VARS = (
    "GEMINI_PRO_MODEL",
    "GEMINI_FLASH_MODEL",
    "GEMINI_IMAGE_MODEL",
    "VTO_MODEL",
)


def validate_model_ids() -> None:
    """Validate model IDs configured in the environment.

    Raises:
        ValueError: If any model ID begins with 'gemini-1' or 'gemini-2'.
    """
    for var in MODEL_ENV_VARS:
        val = os.environ.get(var)
        if not val:
            continue
        val_clean = val.strip().lower()
        if val_clean.startswith("gemini-1") or val_clean.startswith("gemini-2"):
            raise ValueError(
                f"Prohibited Gemini model '{val}' detected in environment variable {var}. "
                "Specification strictly forbids gemini-1.x and gemini-2.x models. "
                "Please configure an approved model in Agent Platform Model Garden."
            )


# Enforce validation on module import
validate_model_ids()

T = TypeVar("T", bound=BaseModel)


def get_client(use_api_key: bool = False) -> Any:
    """Instantiate the Gemini client: Developer API key client if requested and set, else Vertex AI."""
    from google import genai

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if use_api_key and key:
        return genai.Client(api_key=key)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required (or set GEMINI_API_KEY for dev).")
    return genai.Client(vertexai=True, project=project, location="global")


_api_key_calls = 0


def _count_api_key_call() -> None:
    """Sparing use of the dev API key: log each call and stop at GEMINI_API_MAX_CALLS per process."""
    global _api_key_calls
    limit = int(os.environ.get("GEMINI_API_MAX_CALLS", "30"))
    if _api_key_calls >= limit:
        raise RuntimeError(f"GEMINI_API_MAX_CALLS={limit} reached; refusing more Gemini API key calls in this process.")
    _api_key_calls += 1
    logger.info("Gemini API key call %d/%d", _api_key_calls, limit)


def get_model_for_role(role: Literal["pro", "flash"]) -> str:
    """Resolve and validate the model ID for the requested role."""
    var_name = "GEMINI_PRO_MODEL" if role == "pro" else "GEMINI_FLASH_MODEL"
    model_id = os.environ.get(var_name)
    if not model_id or not model_id.strip():
        raise ValueError(
            f"Model environment variable {var_name} is empty or not set. "
            "Configure a valid model ID from Agent Platform Model Garden."
        )

    model_clean = model_id.strip().lower()
    if model_clean.startswith("gemini-1") or model_clean.startswith("gemini-2"):
        raise ValueError(
            f"Prohibited Gemini model '{model_id}' in {var_name}. "
            "gemini-1.x and gemini-2.x models are prohibited."
        )
    return model_id.strip()


BACKENDS = ("auto", "vertex", "ollama", "gemini_api")


def get_backend() -> str:
    """Selected backend from LLM_BACKEND (default 'auto')."""
    backend = os.environ.get("LLM_BACKEND", "auto").strip().lower() or "auto"
    if backend not in BACKENDS:
        raise ValueError(f"LLM_BACKEND must be one of {', '.join(BACKENDS)}, got '{backend}'.")
    return backend


def _ollama_configured() -> bool:
    return bool(os.environ.get("OLLAMA_BASE_URL", "").strip() and os.environ.get("OLLAMA_MODEL", "").strip())


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", getattr(exc, "status_code", None))
    status_str = str(exc)
    return code in (429, 500, 502, 503, 504) or any(
        err in status_str
        for err in ("429", "500", "502", "503", "504", "ResourceExhausted", "InternalServerError", "ServiceUnavailable")
    )


def _with_retries(label: str, role: str, call):
    """Run call() with 3 attempts and exponential backoff on retryable errors; log latency."""
    max_retries = 3
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        start_time = time.perf_counter()
        try:
            return call(start_time)
        except Exception as exc:
            if _is_retryable(exc) and attempt < max_retries:
                logger.warning(
                    "LLM [%s] call failed with retryable status (attempt %d/%d): %s. Retrying in %.2fs...",
                    label, attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error("LLM [%s] call failed permanently after %d attempts: %s", label, attempt, exc)
                raise


def generate_json(
    role: Literal["pro", "flash"],
    parts: Any,
    schema: type[T],
    system: str | None = None,
    client: Any | None = None,
    ollama_transport: Any | None = None,
) -> T:
    """Generate structured JSON conforming to a Pydantic schema.

    See the module docstring for LLM_BACKEND. With "auto", Gemma on Ollama is tried first for
    text-only calls and Gemini is the fallback.

    Args:
        role: "pro" or "flash" selector
        parts: Text string, byte payload, or list of content parts
        schema: Pydantic model class for response schema enforcement
        system: Optional system instruction prompt
        client: Optional pre-configured genai.Client (used for mocking/testing)
        ollama_transport: Optional callable(payload) -> dict replacing the HTTP call (tests)

    Returns:
        Instance of the specified Pydantic schema
    """
    backend = get_backend()
    if backend == "ollama":
        return _generate_json_ollama(role, parts, schema, system, ollama_transport)
    if backend == "auto" and _ollama_configured() and _ollama_supported(parts):
        try:
            return _generate_json_ollama(role, parts, schema, system, ollama_transport)
        except Exception as exc:
            logger.warning("Ollama unavailable or failed (%s); falling back to Gemini", exc)
    use_key = backend in ("auto", "gemini_api") and bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if backend == "gemini_api" and not use_key:
        raise ValueError("LLM_BACKEND=gemini_api requires GEMINI_API_KEY.")
    return _generate_json_vertex(role, parts, schema, system, client, use_key)


def _image_bytes(part: Any) -> bytes | None:
    """Raw bytes of an image part: bytes, or a genai Part carrying inline_data."""
    if isinstance(part, (bytes, bytearray)):
        return bytes(part)
    inline = getattr(part, "inline_data", None)
    data = getattr(inline, "data", None)
    return bytes(data) if isinstance(data, (bytes, bytearray)) else None


def _ollama_supported(parts: Any) -> bool:
    items = parts if isinstance(parts, list) else [parts]
    return all(isinstance(p, str) or _image_bytes(p) is not None for p in items)


def _encode_image(data: bytes) -> str:
    """Base64 of the image, downscaled so the long side is at most OLLAMA_MAX_IMAGE_PX (keeps the 16k context free)."""
    import base64
    import io

    from PIL import Image, ImageOps

    limit = int(os.environ.get("OLLAMA_MAX_IMAGE_PX", "1280"))
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if max(img.size) > limit:
        img.thumbnail((limit, limit))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_json_vertex(role, parts, schema, system, client, use_api_key=False):
    from google.genai import types

    model_id = get_model_for_role(role)
    if client is None:
        client = get_client(use_api_key)

    # Format content parts
    contents: list[Any] = []
    if isinstance(parts, (str, bytes)):
        contents = [parts]
    elif isinstance(parts, list):
        contents = parts
    else:
        contents = [parts]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system,
    )

    def call(start_time: float):
        if use_api_key:
            _count_api_key_call()
        response = client.models.generate_content(model=model_id, contents=contents, config=config)
        elapsed = time.perf_counter() - start_time
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        logger.info(
            "Gemini [%s] role=%s latency=%.3fs prompt_tokens=%s output_tokens=%s",
            model_id, role, elapsed, input_tokens, output_tokens,
        )
        response_text = response.text if hasattr(response, "text") else str(response)
        return schema.model_validate_json(response_text)

    return _with_retries(model_id, role, call)


def _ollama_post(base_url: str, payload: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    # A browser-less default User-Agent is blocked by some Cloudflare-fronted relays (HTTP 403).
    headers = {"Content-Type": "application/json", "User-Agent": "fitcore/0.1 (+ollama-dev)"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=float(os.environ.get("OLLAMA_TIMEOUT_S", "120"))) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _generate_json_ollama(role, parts, schema, system, transport):
    """Dev-only Ollama path. Text and image parts (images as base64); one repair retry if the JSON does not validate."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("LLM_BACKEND=ollama requires OLLAMA_BASE_URL and OLLAMA_MODEL.")
    items = parts if isinstance(parts, list) else [parts]
    if not _ollama_supported(parts):
        raise ValueError("The ollama backend handles text and image bytes only; other part types need Gemini.")
    texts = [p for p in items if isinstance(p, str)]
    images = [_encode_image(b) for b in (_image_bytes(p) for p in items) if b is not None]
    post = transport or (lambda payload: _ollama_post(base_url, payload, os.environ.get("OLLAMA_API_KEY")))

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    user: dict[str, Any] = {"role": "user", "content": "\n\n".join(texts)}
    if images:
        user["images"] = images
    messages.append(user)

    last_error: Exception | None = None
    for repair in range(2):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0},
        }

        def call(start_time: float, payload=payload):
            data = post(payload)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Ollama [%s] role=%s latency=%.3fs prompt_tokens=%s output_tokens=%s",
                model, role, elapsed, data.get("prompt_eval_count"), data.get("eval_count"),
            )
            return data["message"]["content"]

        text = _with_retries(model, role, call)
        try:
            return schema.model_validate_json(text)
        except Exception as exc:  # invalid JSON from a small model: ask once more with the error
            last_error = exc
            logger.warning("Ollama [%s] returned JSON that failed validation (attempt %d/2): %s", model, repair + 1, exc)
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That reply did not match the schema ({exc}). Reply again with only valid JSON."},
            ]
    assert last_error is not None
    raise last_error
