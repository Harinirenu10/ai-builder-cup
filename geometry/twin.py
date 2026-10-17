"""CLI forwarding entrypoint for python -m geometry.twin."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory and services are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICES_DIR = REPO_ROOT / "services"

for p in (str(REPO_ROOT), str(SERVICES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.geometry.twin import main, parse_args, run_offline_pipeline

if __name__ == "__main__":
    sys.exit(main())
