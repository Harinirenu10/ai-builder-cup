"""Regression tests for the two documented ways to launch services/api.

Both broke independently before this file existed, in a way the rest of the
test suite never caught: pytest's own import machinery (rootdir insertion
from `python -m pytest`, plus conftest.py's `sys.path.insert` for `apilib`)
quietly provides exactly the sys.path entries a real launch doesn't get, so
every other test in this directory passes regardless of whether the app can
actually be started. These tests use a subprocess with a controlled
environment instead, so they fail the same way a real launch would.

- infra/Makefile's `dev` target runs `uvicorn services.api.main:app` from the
  repo root. Before the fix this raised `ModuleNotFoundError: No module named
  'apilib'`, because nothing put services/api itself on sys.path.
- services/api/Dockerfile runs `uvicorn main:app` with this directory
  flattened to /app. Before the fix this raised `ModuleNotFoundError: No
  module named 'services'`, because main.py imports
  services.geometry.cloud_client unconditionally and services/geometry was
  never part of the image. This test builds a throwaway directory that
  mirrors exactly what the Dockerfile's COPY instructions produce, rather
  than relying on Docker being installed, so it stays fast and catches a
  regression if someone adds a new cross-service import without updating the
  Dockerfile to match.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parents[1]
GEOMETRY_DIR = REPO_ROOT / "services" / "geometry"

_ENV_EXTRA = {"AUTH_MODE": "dev", "CONSENT_JWT_SECRET": "test-secret-" + "x" * 24}
_IMPORT_CHECK = "import main; assert main.app is not None"


def _subprocess_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(_ENV_EXTRA)
    return env


def _run(code: str, cwd: Path, import_as: str = "main") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code.replace("import main", f"import {import_as} as main")],
        cwd=str(cwd),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_make_dev_style_launch_imports_from_repo_root():
    """`uvicorn services.api.main:app`, run with the repo root as cwd (infra/Makefile's `dev` target)."""
    result = _run(_IMPORT_CHECK, REPO_ROOT, import_as="services.api.main")
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_dockerfile_style_launch_imports_from_flattened_app_dir(tmp_path):
    """`uvicorn main:app`, run with services/api flattened to /app (services/api/Dockerfile).

    Mirrors the Dockerfile's own COPY instructions into a throwaway directory
    instead of requiring Docker: `COPY services/api/ /app/` plus the
    lightweight services.geometry.cloud_client slice main.py needs.
    """
    app_dir = tmp_path / "app"
    shutil.copytree(
        API_DIR,
        app_dir,
        ignore=shutil.ignore_patterns("tests", "__pycache__", ".pytest_cache", ".data"),
    )
    geometry_dest = app_dir / "services" / "geometry"
    geometry_dest.mkdir(parents=True)
    shutil.copy(GEOMETRY_DIR / "__init__.py", geometry_dest / "__init__.py")
    shutil.copy(GEOMETRY_DIR / "cloud_client.py", geometry_dest / "cloud_client.py")

    result = _run(_IMPORT_CHECK, app_dir)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
