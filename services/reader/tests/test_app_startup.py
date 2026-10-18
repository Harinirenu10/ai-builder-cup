"""Regression test for infra/Makefile's `dev` target launch of services/reader.

`uvicorn services.reader.main:app`, run from the repo root, used to raise
`ModuleNotFoundError: No module named 'readerlib'`, because nothing put
services/reader itself on sys.path when the process's cwd is the repo root
rather than this directory. pytest's own import machinery papers over this
(rootdir insertion from `python -m pytest` already puts the repo root on
sys.path, and this directory's conftest.py separately inserts
services/reader for readerlib), so a normal test here would never catch a
regression -- this uses a subprocess with a controlled environment instead,
so it fails the same way a real launch would.

services/reader/Dockerfile's own launch style (`uvicorn main:app` with this
directory as cwd) was never broken and needs no equivalent synthesized-image
test: unlike services/api, main.py here has no cross-service import, so cwd
alone is sufficient and the existing test suite already exercises it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

READER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = READER_DIR.parents[1]


def test_make_dev_style_launch_imports_from_repo_root():
    """`uvicorn services.reader.main:app`, run with the repo root as cwd (infra/Makefile's `dev` target)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", "import services.reader.main as m; assert m.app is not None"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
