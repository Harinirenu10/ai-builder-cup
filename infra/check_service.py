"""Reads `gcloud run services describe --format=json` on stdin and checks the scaling spec. Used by verify_infra.sh.

    gcloud run services describe fit-api --region R --format=json | python infra/check_service.py api fit-api
"""

import json
import sys


def check(doc: dict, key: str) -> tuple[list[str], dict[str, str]]:
    tpl = doc["spec"]["template"]
    ann = tpl["metadata"].get("annotations", {})
    container = tpl["spec"]["containers"][0]
    limits = container.get("resources", {}).get("limits", {})
    errs = []
    if ann.get("autoscaling.knative.dev/maxScale") != "3":
        errs.append("max instances is not 3")
    if ann.get("run.googleapis.com/startup-cpu-boost") != "true":
        errs.append("startup CPU boost is off")
    if key == "geometry" and (limits.get("cpu") != "4" or limits.get("memory") != "8Gi" or tpl["spec"].get("timeoutSeconds") != 900):
        errs.append("geometry is not 4 vCPU / 8Gi / 900 s")
    return errs, {"min": ann.get("autoscaling.knative.dev/minScale", "0"), "max": ann.get("autoscaling.knative.dev/maxScale", "?")}


if __name__ == "__main__":
    key, name = sys.argv[1:3]
    errors, info = check(json.load(sys.stdin), key)
    print(name, f"min={info['min']} max={info['max']}", "OK" if not errors else "FAIL: " + ", ".join(errors))
    sys.exit(1 if errors else 0)
