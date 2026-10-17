"""Unified LLM wrapper for Measured Fit supporting Gemini (Vertex AI) and Ollama.

All LLM calls across the system MUST route through this module.
Providers:
- "gemini": Google Cloud Enterprise Agent Platform (Vertex AI) using google-genai SDK.
  Hard constraints:
  - genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location="global")
  - Model IDs from env vars only: GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_IMAGE_MODEL, VTO_MODEL
  - Never hardcode model IDs. Never allow any gemini-1.x or gemini-2.x model.
- "ollama": Internal Ollama endpoint with model 'gemma4:12b' for local development & testing.
  Configured via:
  - LLM_PROVIDER=ollama
  - OLLAMA_BASE_URL (default: http://localhost:11434)
  - OLLAMA_MODEL (default: gemma4:12b)
  - OLLAMA_TIMEOUT (default: 60.0)
  - OLLAMA_API_KEY (optional auth header for protected internal endpoints)

Common guarantees across both providers:
- Structured JSON output enforced via Pydantic schemas.
- 3 retries with exponential backoff on retryable status codes (429, 5xx) and timeouts.
- Standardized latency, model, and token count logging.
- Support for multimodal parts (text, image bytes, and types.Part).
- Production credentials kept out of source code and log statements.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import time
from typing import Any, Literal, TypeVar
import urllib.error
import urllib.request
from pydantic import BaseModel, ValidationError

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("fitcore.llm")


# Regulated environment variables containing Gemini model IDs
MODEL_ENV_VARS = (
    "GEMINI_PRO_MODEL",
    "GEMINI_FLASH_MODEL",
    "GEMINI_IMAGE_MODEL",
    "VTO_MODEL",
)

DEFAULT_OLLAMA_MODEL = "gemma4:12b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

T = TypeVar("T", bound=BaseModel)


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


def get_provider() -> Literal["gemini", "ollama"]:
    """Resolve the active LLM provider from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in ("gemini", "ollama"):
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Allowed values are 'gemini' or 'ollama'."
        )
    return provider  # type: ignore


def get_ollama_base_url() -> str:
    """Resolve the Ollama endpoint base URL."""
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")


def get_ollama_model() -> str:
    """Resolve the Ollama model name."""
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def get_client() -> Any:
    """Instantiate Vertex AI Gemini client using google-genai SDK."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required.")
    return genai.Client(vertexai=True, project=project, location="global")


def get_model_for_role(role: Literal["pro", "flash"]) -> str:
    """Resolve and validate the Gemini model ID for the requested role."""
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


def _extract_text_and_images(parts: Any) -> tuple[str, list[str]]:
    """Extract concatenated text and base64-encoded images from multimodal parts.

    Supports strings, raw bytes, dicts, and google.genai.types.Part.
    """
    texts: list[str] = []
    images: list[str] = []

    def _process_item(item: Any) -> None:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, bytes):
            images.append(base64.b64encode(item).decode("utf-8"))
        elif hasattr(item, "inline_data") and getattr(item.inline_data, "data", None):
            data = item.inline_data.data
            if isinstance(data, bytes):
                images.append(base64.b64encode(data).decode("utf-8"))
            elif isinstance(data, str):
                images.append(data)
        elif hasattr(item, "text") and getattr(item, "text", None):
            texts.append(item.text)
        elif isinstance(item, dict):
            if "text" in item:
                texts.append(str(item["text"]))
            if "image" in item:
                img_val = item["image"]
                if isinstance(img_val, bytes):
                    images.append(base64.b64encode(img_val).decode("utf-8"))
                elif isinstance(img_val, str):
                    images.append(img_val)
        else:
            texts.append(str(item))

    if isinstance(parts, (str, bytes)):
        _process_item(parts)
    elif isinstance(parts, list):
        for p in parts:
            _process_item(p)
    else:
        _process_item(parts)

    return "\n\n".join(texts), images


def _clean_json_text(text: str) -> str:
    """Strip markdown code block markers (```json ... ```) from model response."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _generate_json_gemini(
    role: Literal["pro", "flash"],
    parts: Any,
    schema: type[T],
    system: str | None = None,
    client: Any | None = None,
) -> T:
    """Generate structured JSON conforming to a Pydantic schema using Gemini on Vertex AI."""
    from google.genai import types

    model_id = get_model_for_role(role)
    if client is None:
        client = get_client()

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

    max_retries = 3
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
            elapsed = time.perf_counter() - start_time

            # Log latency and token usage
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            logger.info(
                "Gemini [%s] role=%s latency=%.3fs prompt_tokens=%s output_tokens=%s",
                model_id,
                role,
                elapsed,
                input_tokens,
                output_tokens,
            )

            response_text = response.text if hasattr(response, "text") else str(response)
            clean_text = _clean_json_text(response_text)
            return schema.model_validate_json(clean_text)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            code = getattr(exc, "code", getattr(exc, "status_code", None))
            status_str = str(exc)
            is_timeout = (
                isinstance(exc, (TimeoutError, socket.timeout))
                or "timeout" in status_str.lower()
                or "timed out" in status_str.lower()
            )
            is_malformed_json = isinstance(exc, (json.JSONDecodeError, ValueError, ValidationError))
            is_retryable = (
                is_timeout
                or is_malformed_json
                or code in (429, 500, 502, 503, 504)
                or any(
                    err in status_str
                    for err in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "ResourceExhausted",
                        "InternalServerError",
                        "ServiceUnavailable",
                    )
                )
            )


            if is_retryable and attempt < max_retries:
                logger.warning(
                    "Gemini [%s] call failed with retryable status (attempt %d/%d): %s. Retrying in %.2fs...",
                    model_id,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(
                    "Gemini [%s] call failed permanently after %d attempts: %s",
                    model_id,
                    attempt,
                    exc,
                )
                raise


def _generate_json_ollama(
    role: Literal["pro", "flash"],
    parts: Any,
    schema: type[T],
    system: str | None = None,
    client: Any | None = None,
) -> T:
    """Generate structured JSON conforming to a Pydantic schema using Ollama (/api/chat)."""
    model = get_ollama_model()
    base_url = get_ollama_base_url()
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "60.0"))
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    prompt_text, images = _extract_text_and_images(parts)

    schema_dict = schema.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    system_content = (
        f"{system}\n" if system else "You are an expert AI assistant that outputs structured data.\n"
    )
    system_content += (
        f"You MUST respond ONLY with a single valid JSON object that strictly adheres to this schema:\n"
        f"{schema_json}\n"
        "Do not output markdown explanations, code blocks, or preamble. Return raw JSON only."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]

    user_msg: dict[str, Any] = {"role": "user", "content": prompt_text}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "fitcore/0.1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req_data = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/api/chat"

    max_retries = 3
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        start_time = time.perf_counter()
        try:
            if client is not None:
                # Mock or injected test client
                if callable(client):
                    response_dict = client(url=url, payload=payload, headers=headers)
                elif hasattr(client, "chat"):
                    response_dict = client.chat(payload)
                elif hasattr(client, "post"):
                    res = client.post(url, json=payload, headers=headers, timeout=timeout)
                    response_dict = res.json()
                elif isinstance(client, dict):
                    response_dict = client
                else:
                    raise TypeError(f"Unsupported mock client type: {type(client)}")
            else:
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_body = resp.read().decode("utf-8")
                    response_dict = json.loads(resp_body)

            elapsed = time.perf_counter() - start_time
            prompt_tokens = response_dict.get("prompt_eval_count")
            output_tokens = response_dict.get("eval_count")

            logger.info(
                "Ollama [%s] role=%s latency=%.3fs prompt_tokens=%s output_tokens=%s",
                model,
                role,
                elapsed,
                prompt_tokens,
                output_tokens,
            )

            raw_content = response_dict.get("message", {}).get("content", "")
            if not raw_content:
                raise ValueError("Ollama returned an empty response content.")

            clean_content = _clean_json_text(raw_content)
            return schema.model_validate_json(clean_content)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            code = getattr(exc, "code", getattr(exc, "status_code", None))
            status_str = str(exc)
            is_timeout = (
                isinstance(exc, (TimeoutError, socket.timeout))
                or (isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)))
                or "timeout" in status_str.lower()
                or "timed out" in status_str.lower()
            )
            is_malformed_json = isinstance(exc, (json.JSONDecodeError, ValueError, ValidationError))
            is_retryable = (
                is_timeout
                or is_malformed_json
                or code in (429, 500, 502, 503, 504)
                or any(
                    err in status_str
                    for err in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "ConnectionRefused",
                        "timed out",
                        "Timeout",
                        "RemoteDisconnected",
                        "URLError",
                    )
                )
            )


            if is_retryable and attempt < max_retries:
                logger.warning(
                    "Ollama [%s] call failed with retryable error (attempt %d/%d): %s. Retrying in %.2fs...",
                    model,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(
                    "Ollama [%s] call failed permanently after %d attempts: %s",
                    model,
                    attempt,
                    exc,
                )
                raise


def generate_json(
    role: Literal["pro", "flash"],
    parts: Any,
    schema: type[T],
    system: str | None = None,
    client: Any | None = None,
) -> T:
    """Generate structured JSON conforming to a Pydantic schema using configured provider.

    Provider selection:
    - LLM_PROVIDER="gemini" (default): Routes to Vertex AI Gemini.
    - LLM_PROVIDER="ollama": Routes to local/internal Ollama (model: gemma4:12b).

    Args:
        role: "pro" or "flash" role designation
        parts: Text string, byte payload, list of content parts, or types.Part instances
        schema: Pydantic model class for response validation
        system: Optional system instruction prompt
        client: Optional pre-configured client or mock callable

    Returns:
        Instance of the specified Pydantic schema
    """
    provider = get_provider()
    if provider == "ollama":
        return _generate_json_ollama(
            role=role,
            parts=parts,
            schema=schema,
            system=system,
            client=client,
        )
    return _generate_json_gemini(
        role=role,
        parts=parts,
        schema=schema,
        system=system,
        client=client,
    )
