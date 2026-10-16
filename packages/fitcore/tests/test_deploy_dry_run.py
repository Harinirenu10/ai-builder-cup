import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


def test_makefile_exists_and_contains_required_flags():
    repo_root = Path(__file__).resolve().parents[3]
    makefile_path = repo_root / "infra" / "Makefile"
    assert makefile_path.exists(), f"infra/Makefile not found at {makefile_path}"

    content = makefile_path.read_text(encoding="utf-8")
    assert "--region=" in content
    assert "--min-instances=0" in content
    assert "--max-instances=3" in content
    assert "--cpu-boost" in content
    assert "--cpu=4" in content
    assert "--memory=8Gi" in content
    assert "--timeout=900" in content
    assert "DRY_RUN" in content
    assert "deploy-api:" in content
    assert "deploy-geometry:" in content
    assert "deploy-reader:" in content
    assert "deploy-web:" in content
    assert "create-queue:" in content


def test_deploy_api_dry_run_command_output():
    repo_root = Path(__file__).resolve().parents[3]
    makefile_path = repo_root / "infra" / "Makefile"

    # Find make executable (native make, gnuwin32, or git-provided make)
    make_cmd = shutil.which("make")
    if not make_cmd:
        candidates = [
            r"C:\Program Files (x86)\GnuWin32\bin\make.exe",
            r"C:\GnuWin32\bin\make.exe",
            r"C:\Users\Harini.Ramamoorthy\AppData\Local\Programs\Git\usr\bin\make.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                make_cmd = c
                break
    env = os.environ.copy()
    env["GOOGLE_CLOUD_PROJECT"] = "test-project-123"
    env["REGION"] = "us-central1"
    env["DRY_RUN"] = "1"

    if not make_cmd:

        # Fall back to infra/make.py runner
        proc = subprocess.run(
            [sys.executable, str(repo_root / "infra" / "make.py"), "deploy-api", "DRY_RUN=1"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
    else:
        proc = subprocess.run(
            [make_cmd, "-f", str(makefile_path), "deploy-api"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )

    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"deploy-api dry run failed: {output}"
    assert "gcloud run deploy" in output
    assert "--min-instances=0" in output
    assert "--max-instances=3" in output
    assert "--cpu-boost" in output
    assert "--region=us-central1" in output

