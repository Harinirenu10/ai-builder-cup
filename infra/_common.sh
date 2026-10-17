#!/usr/bin/env bash
# Shared settings for the infra scripts. Source it; do not run it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
: "${PROJECT:?set GOOGLE_CLOUD_PROJECT (or PROJECT)}"
REGION="${REGION:-us-central1}"
API_SERVICE="${API_SERVICE:-fit-api}"
GEOMETRY_SERVICE="${GEOMETRY_SERVICE:-fit-geometry}"
READER_SERVICE="${READER_SERVICE:-fit-reader}"
TASKS_QUEUE="${TASKS_QUEUE:-fit-jobs-queue}"
REPO="${AR_REPO:-fit}"

# Run a command, or print it when DRY_RUN=1.
run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN=1] $*"; else "$@"; fi; }

# Cloud Run service name for a short key.
svc_name() {
  case "$1" in
    api) echo "$API_SERVICE";;
    geometry) echo "$GEOMETRY_SERVICE";;
    reader) echo "$READER_SERVICE";;
    *) echo "unknown service '$1' (api|geometry|reader)" >&2; return 1;;
  esac
}

# Revision that currently carries the "submitted" tag; empty when there is none.
submitted_revision() {
  gcloud run services describe "$1" --region "$REGION" --project "$PROJECT" --format=json 2>/dev/null |
    python -c 'import json,sys
d=json.load(sys.stdin)
for t in d.get("status",{}).get("traffic",[]):
    if t.get("tag")=="submitted":
        print(t.get("revisionName",""))
        break' 2>/dev/null || true
}
