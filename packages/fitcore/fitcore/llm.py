"""Gemini LLM wrapper for Google Cloud Enterprise Agent Platform (Vertex AI).

All Gemini calls across the system MUST route through this module.
Hard constraints:
- genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location="global")
- Model IDs from env vars only: GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GEMINI_IMAGE_MODEL, VTO_MODEL
- Never hardcode model IDs. Never allow any gemini-1.x or gemini-2.x model.
- Structured JSON output via pydantic schemas.
- 3 retries with exponential backoff on 429 and 5xx errors.
- Logging of model ID, latency, and token counts.
"""

import logging
import os
import time
from typing import Any, Literal, TypeVar
from pydantic import BaseModel

logger = logging.getLogger("fitcore.llm")

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


def get_client() -> Any:
    """Instantiate Vertex AI Gemini client using google-genai SDK."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required.")
    return genai.Client(vertexai=True, project=project, location="global")


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


def generate_json(
    role: Literal["pro", "flash"],
    parts: Any,
    schema: type[T],
    system: str | None = None,
    client: Any | None = None,
) -> T:
    """Generate structured JSON conforming to a Pydantic schema using Gemini.

    Args:
        role: "pro" or "flash" selector
        parts: Text string, byte payload, or list of content parts
        schema: Pydantic model class for response schema enforcement
        system: Optional system instruction prompt
        client: Optional pre-configured genai.Client (used for mocking/testing)

    Returns:
        Instance of the specified Pydantic schema
    """
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
            return schema.model_validate_json(response_text)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            code = getattr(exc, "code", getattr(exc, "status_code", None))
            status_str = str(exc)
            is_retryable = (
                code in (429, 500, 502, 503, 504)
                or any(err in status_str for err in ("429", "500", "502", "503", "504", "ResourceExhausted", "InternalServerError", "ServiceUnavailable"))
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
