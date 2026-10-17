#!/usr/bin/env bash
# Tag each service's current serving revision "submitted" and pin 100% of traffic to it.
# After this, infra/deploy.sh deploys with --no-traffic and a tag, so judges keep seeing the submitted build.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
keys=("$@")
if [ ${#keys[@]} -eq 0 ]; then keys=(api reader geometry); fi
for k in "${keys[@]}"; do
  name="$(svc_name "$k")"
  rev="$(gcloud run services describe "$name" --region "$REGION" --project "$PROJECT" --format='value(status.traffic[0].revisionName)')"
  [ -n "$rev" ] || { echo "No serving revision for $name" >&2; exit 1; }
  echo "$name: pinning $rev as 'submitted'"
  run gcloud run services update-traffic "$name" --region "$REGION" --project "$PROJECT" --set-tags="submitted=$rev"
  run gcloud run services update-traffic "$name" --region "$REGION" --project "$PROJECT" --to-revisions="$rev=100"
done
