#!/usr/bin/env bash
# Build and deploy one Cloud Run service:  infra/deploy.sh api|geometry|reader   (DRY_RUN=1 prints instead of running)
# Every service: min instances 0, max 3, startup CPU boost. Geometry: 4 vCPU, 8 GiB, 900 s timeout.
# Once infra/pin_submitted.sh has tagged "submitted", deploys use --no-traffic with a tag and never take traffic.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
key="${1:?usage: infra/deploy.sh api|geometry|reader}"
name="$(svc_name "$key")"
BASE_FLAGS="${BASE_DEPLOY_FLAGS:---region=$REGION --min-instances=0 --max-instances=3 --cpu-boost}"
case "$key" in
  api)      EXTRA="--memory=1Gi --cpu=1 --allow-unauthenticated";;      # public: Firebase Hosting rewrites /api/** to it
  reader)   EXTRA="--memory=2Gi --cpu=1 --no-allow-unauthenticated";;   # Chromium needs about 2 GiB; callers need an identity token
  geometry) EXTRA="--cpu=4 --memory=8Gi --timeout=900 --no-allow-unauthenticated";;
esac
ENV_FILE="infra/vars/$key.yaml"
if [ ! -f "$ENV_FILE" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "Missing $ENV_FILE. Copy infra/vars/$key.example.yaml and fill it in." >&2
  exit 1
fi
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$name:$(date +%Y%m%d-%H%M%S)"
TRAFFIC=""
if [ "${DRY_RUN:-0}" != "1" ] && [ -n "$(submitted_revision "$name")" ]; then
  TRAFFIC="--no-traffic --tag=cand-$(date +%m%d-%H%M%S)"
  echo "A 'submitted' revision is pinned: this deploy gets no traffic ($TRAFFIC)."
fi
run gcloud builds submit --project "$PROJECT" --config infra/cloudbuild.yaml --substitutions "_DOCKERFILE=services/$key/Dockerfile,_IMAGE=$IMAGE" .
# shellcheck disable=SC2086
run gcloud run deploy "$name" --image "$IMAGE" $BASE_FLAGS $EXTRA --env-vars-file "$ENV_FILE" $TRAFFIC --project "$PROJECT"
