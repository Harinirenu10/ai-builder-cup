import pytest


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    """Tests are offline: ignore any backend, Ollama or API key settings from the developer's shell or .env."""
    for var in ("LLM_BACKEND", "LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_MAX_CALLS"):
        monkeypatch.delenv(var, raising=False)
