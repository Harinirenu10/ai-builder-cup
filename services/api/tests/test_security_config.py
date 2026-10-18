"""Regression tests for apilib/auth.py::validate_security_config -- the two startup safeguards:

1. AUTH_MODE=dev (the `Bearer dev:<uid>` shortcut) must never be reachable on a real Cloud Run
   deployment (detected via the `K_SERVICE` env var Cloud Run sets automatically).
2. CONSENT_JWT_SECRET must be set and at least MIN_CONSENT_SECRET_LEN characters whenever
   AUTH_MODE is not the literal string "dev".

Unit-level tests call validate_security_config() directly with monkeypatched env vars (fast,
precise about which of the two checks fired). The two subprocess tests at the bottom additionally
prove the real failure mode end to end -- a real `python -c "import main"`, mirroring
test_app_startup.py's own pattern -- so this is verified as an actual import-time crash, not just
a unit assertion about the helper function in isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from apilib.auth import MIN_CONSENT_SECRET_LEN, is_production, validate_security_config

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("CONSENT_JWT_SECRET", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)


# ----------------------------------------------------------------------------- is_production()


def test_is_production_false_without_k_service():
    assert is_production() is False


def test_is_production_true_with_k_service(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "fit-api")
    assert is_production() is True


# ----------------------------------------------------------------------------- validate_security_config()


def test_safe_local_development_passes(monkeypatch):
    """The documented local dev setup: AUTH_MODE=dev, no K_SERVICE, no CONSENT_JWT_SECRET at all."""
    monkeypatch.setenv("AUTH_MODE", "dev")
    validate_security_config()  # must not raise


def test_production_with_dev_auth_enabled_fails(monkeypatch):
    """The exact unsafe configuration this check exists to catch: AUTH_MODE=dev reachable on a
    real Cloud Run revision."""
    monkeypatch.setenv("K_SERVICE", "fit-api")
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("CONSENT_JWT_SECRET", "x" * MIN_CONSENT_SECRET_LEN)  # not what should matter here
    with pytest.raises(RuntimeError, match="AUTH_MODE=dev"):
        validate_security_config()


def test_missing_secret_outside_dev_fails(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "production")  # anything other than the literal string "dev"
    with pytest.raises(RuntimeError, match="CONSENT_JWT_SECRET is not set"):
        validate_security_config()


def test_missing_secret_with_auth_mode_unset_fails(monkeypatch):
    """AUTH_MODE unset entirely (the real Cloud Run default) is also "not dev"."""
    with pytest.raises(RuntimeError, match="CONSENT_JWT_SECRET is not set"):
        validate_security_config()


def test_weak_secret_outside_dev_fails(monkeypatch):
    monkeypatch.setenv("CONSENT_JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="too short"):
        validate_security_config()


def test_secret_exactly_at_threshold_passes(monkeypatch):
    monkeypatch.setenv("CONSENT_JWT_SECRET", "x" * MIN_CONSENT_SECRET_LEN)
    validate_security_config()  # must not raise


def test_valid_production_configuration_passes(monkeypatch):
    """K_SERVICE set (real Cloud Run), AUTH_MODE unset (never "dev" in prod), a strong secret."""
    monkeypatch.setenv("K_SERVICE", "fit-api")
    monkeypatch.setenv("CONSENT_JWT_SECRET", "a-sufficiently-long-random-production-secret-value")
    validate_security_config()  # must not raise


def test_error_messages_never_contain_the_secret_value(monkeypatch):
    secret = "short-but-very-identifiable-secret-xyz"
    monkeypatch.setenv("CONSENT_JWT_SECRET", secret[:10])  # still short enough to fail
    with pytest.raises(RuntimeError) as exc_info:
        validate_security_config()
    assert secret not in str(exc_info.value)
    assert secret[:10] not in str(exc_info.value)


# ----------------------------------------------------------------------------- real process-level startup failure


_IMPORT_CHECK = "import main; assert main.app is not None"


def _run(code: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AUTH_MODE", "CONSENT_JWT_SECRET", "K_SERVICE")}
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code.replace("import main", "import services.api.main as main")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_real_process_fails_to_start_with_dev_auth_in_production():
    """The actual failure a bad Cloud Run deploy would hit: the process crashes on import instead
    of silently starting with an open dev-auth backdoor."""
    result = _run(_IMPORT_CHECK, {"K_SERVICE": "fit-api", "AUTH_MODE": "dev", "CONSENT_JWT_SECRET": "x" * 40})
    assert result.returncode != 0
    assert "AUTH_MODE=dev" in result.stderr


def test_real_process_starts_with_a_valid_production_configuration():
    result = _run(_IMPORT_CHECK, {"K_SERVICE": "fit-api", "CONSENT_JWT_SECRET": "a-sufficiently-long-random-production-secret-value"})
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
