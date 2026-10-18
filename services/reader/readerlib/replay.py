"""Offline stand-in for the Gemini client: returns a recorded response, so tests and the CLI run without the cloud."""

import os
from pathlib import Path
from types import SimpleNamespace

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "llm"


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"reader_{name}.json"


class ReplayClient:
    """Duck-types genai.Client for fitcore.llm.generate_json (vertex path)."""

    def __init__(self, path: Path):
        self._text = path.read_text(encoding="utf-8")
        self.models = SimpleNamespace(generate_content=self._generate)
        self.calls = 0

    def _generate(self, model, contents, config):  # noqa: ARG002
        self.calls += 1
        return SimpleNamespace(text=self._text, usage_metadata=None)


def use_offline_env() -> None:
    """generate_json needs a model id and the vertex backend to accept an injected client."""
    os.environ["LLM_BACKEND"] = "vertex"
    os.environ["GEMINI_FLASH_MODEL"] = os.environ.get("GEMINI_FLASH_MODEL") or "recorded-fixture"
