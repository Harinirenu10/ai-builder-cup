#!/usr/bin/env bash
# infra/set_min_instances.sh N [api|reader|geometry ...]     default services: api reader
# If a revision is tagged "submitted", the change is made on a copy of that revision (same image) that takes over the tag and 100% of
# traffic, so the pinned code never changes. Settings other than the image come from the service's current template: if a later
# candidate deploy changed env vars or resources, redeploy the submitted build before relying on this.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
n="${1:?usage: set_min_instances.sh N [api|reader|geometry ...]}"
shift || true
keys=("$@")
if [ ${#keys[@]} -eq 0 ]; then keys=(api reader); fi
for k in "${keys[@]}"; do
  name="$(svc_name "$k")"
  rev="$(submitted_revision "$name")"
  if [ -n "$rev" ]; then
    image="$(gcloud run revisions describe "$rev" --region "$REGION" --project "$PROJECT" --format='value(spec.containers[0].image)')"
    run gcloud run services update "$name" --region "$REGION" --project "$PROJECT" --image "$image" --min-instances="$n" --no-traffic --tag=submitted
    newrev="$(gcloud run services describe "$name" --region "$REGION" --project "$PROJECT" --format='value(status.latestCreatedRevisionName)')"
    run gcloud run services update-traffic "$name" --region "$REGION" --project "$PROJECT" --to-revisions="$newrev=100"
  else
    run gcloud run services update "$name" --region "$REGION" --project "$PROJECT" --min-instances="$n"
  fi
done
