#!/usr/bin/env python3
"""Cross-platform Makefile runner for systems without GNU Make installed."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "infra" / "Makefile"


def parse_makefile() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse basic variables and target recipes from infra/Makefile."""
    variables: dict[str, str] = {
        "REGION": os.environ.get("REGION", "us-central1"),
        "PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "API_SERVICE": "fit-api",
        "GEOMETRY_SERVICE": "fit-geometry",
        "READER_SERVICE": "fit-reader",
        "TASKS_QUEUE": os.environ.get("TASKS_QUEUE", "fit-jobs-queue"),
        "DRY_RUN": os.environ.get("DRY_RUN", "0"),
    }
    targets: dict[str, list[str]] = {}

    current_target = None
    with open(MAKEFILE, encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            # Parse target definition
            target_match = re.match(r"^([a-zA-Z0-9_\-]+)\s*:", line)
            if target_match and not line.startswith("\t"):
                current_target = target_match.group(1)
                targets[current_target] = []
                continue

            # Recipe lines start with tab
            if line.startswith("\t") and current_target:
                recipe = line.strip()
                if recipe:
                    targets[current_target].append(recipe)

    return variables, targets


def run_target(target_name: str, extra_args: dict[str, str]) -> int:
    variables, targets = parse_makefile()
    variables.update(extra_args)

    if target_name not in targets:
        print(f"Error: Target '{target_name}' not found in infra/Makefile.")
        print(f"Available targets: {', '.join(targets.keys())}")
        return 1

    dry_run = variables.get("DRY_RUN") == "1"
    region = variables.get("REGION", "us-central1")
    project = variables.get("PROJECT", "")
    api_service = variables.get("API_SERVICE", "fit-api")
    geometry_service = variables.get("GEOMETRY_SERVICE", "fit-geometry")
    reader_service = variables.get("READER_SERVICE", "fit-reader")
    tasks_queue = variables.get("TASKS_QUEUE", "fit-jobs-queue")

    base_deploy_flags = f"--region={region} --min-instances=0 --max-instances=3 --cpu-boost"
    geometry_flags = f"{base_deploy_flags} --cpu=4 --memory=8Gi --timeout=900"
    project_flag = f" --project={project}" if project else ""

    commands: list[tuple[str, Path]] = []
    if target_name == "deploy-api":
        cmd = f"gcloud run deploy {api_service} --source=. --dockerfile=services/api/Dockerfile {base_deploy_flags}{project_flag}"
        commands.append((f"[DRY_RUN=1] {cmd}" if dry_run else cmd, REPO_ROOT))
    elif target_name == "deploy-geometry":
        cmd = f"gcloud run deploy {geometry_service} --source=. --dockerfile=services/geometry/Dockerfile {geometry_flags}{project_flag}"
        commands.append((f"[DRY_RUN=1] {cmd}" if dry_run else cmd, REPO_ROOT))
    elif target_name == "deploy-reader":
        cmd = f"gcloud run deploy {reader_service} --source=. --dockerfile=services/reader/Dockerfile {base_deploy_flags}{project_flag}"
        commands.append((f"[DRY_RUN=1] {cmd}" if dry_run else cmd, REPO_ROOT))
    elif target_name == "deploy-web":
        commands.append(("npm run build", REPO_ROOT / "web"))
        dep_cmd = f"firebase deploy --only hosting{project_flag}"
        commands.append((f"[DRY_RUN=1] {dep_cmd}" if dry_run else dep_cmd, REPO_ROOT))
    elif target_name == "create-queue":
        cmd = f"gcloud tasks queues create {tasks_queue} --location={region}{project_flag}"
        commands.append((f"[DRY_RUN=1] {cmd}" if dry_run else cmd, REPO_ROOT))
    elif target_name == "test":
        pytest_cmd = "pytest" if shutil.which("pytest") else f'"{sys.executable}" -m pytest'
        commands.append((f"{pytest_cmd} packages/fitcore/tests -v", REPO_ROOT))
        commands.append((f"{pytest_cmd} services/api/tests -v", REPO_ROOT))
    elif target_name == "dev":
        print("Dev target: Spawns microservices with uvicorn --reload and Firebase emulators.")
        commands.append(("uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload", REPO_ROOT))

    for cmd, cwd in commands:
        if cmd.startswith("[DRY_RUN=1]"):
            print(cmd)
        else:
            print(f"Running (cwd={cwd.name}): {cmd}")
            ret = subprocess.run(cmd, shell=True, cwd=cwd)
            if ret.returncode != 0:
                return ret.returncode
    return 0



if __name__ == "__main__":
    args = sys.argv[1:]
    target = "help"
    overrides: dict[str, str] = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            overrides[k] = v
        else:
            target = a
    sys.exit(run_target(target, overrides))
