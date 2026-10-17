#!/usr/bin/env bash
# One-time (idempotent) project setup. Needs gcloud logged in, GOOGLE_CLOUD_PROJECT, and BILLING_ACCOUNT (like 0X0X0X-0X0X0X-0X0X0X).
# Creates: the Artifact Registry repo Cloud Build pushes to, the Cloud Tasks queue for twin jobs, Firestore TTL policies for demo
# sandboxes, the roles the services need, and the budget alert. DRY_RUN=1 prints instead of running.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BUDGET_USD="${BUDGET_USD:-50}"
: "${BILLING_ACCOUNT:?set BILLING_ACCOUNT to create the budget alert}"

run gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com \
  cloudtasks.googleapis.com aiplatform.googleapis.com iamcredentials.googleapis.com billingbudgets.googleapis.com --project "$PROJECT"

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 ||
  run gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" --project "$PROJECT"

# Cloud Tasks queue for twin jobs.
if gcloud tasks queues describe "$TASKS_QUEUE" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  run gcloud tasks queues update "$TASKS_QUEUE" --location "$REGION" --project "$PROJECT" --max-concurrent-dispatches=3 --max-attempts=3
else
  run gcloud tasks queues create "$TASKS_QUEUE" --location "$REGION" --project "$PROJECT" --max-concurrent-dispatches=3 --max-attempts=3
fi

# The services run as the default compute service account. It reads and writes Firestore, calls Vertex, enqueues tasks, and signs
# the Firebase custom tokens that POST /v1/demo/session returns (Token Creator on itself).
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA="$NUMBER-compute@developer.gserviceaccount.com"
for role in roles/datastore.user roles/aiplatform.user roles/cloudtasks.enqueuer; do
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" --role "$role" --condition=None --quiet >/dev/null
done
run gcloud iam service-accounts add-iam-policy-binding "$SA" --member "serviceAccount:$SA" --role roles/iam.serviceAccountTokenCreator --project "$PROJECT" --quiet >/dev/null

# Demo sandboxes carry expire_at; Firestore deletes them once it passes.
for coll in twins wardrobe events fits orders returns unboxing_measurements gifts charts merchants; do
  run gcloud firestore fields ttls update expire_at --collection-group="$coll" --enable-ttl --project "$PROJECT" --async --quiet
done

# Budget alert on the whole project: emails the billing admins at 50%, 90% and 100% of BUDGET_USD.
if gcloud billing budgets list --billing-account "$BILLING_ACCOUNT" --format='value(displayName)' 2>/dev/null | grep -qx "fit-demo-$PROJECT"; then
  echo "Budget fit-demo-$PROJECT already exists."
else
  run gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" --display-name "fit-demo-$PROJECT" \
    --budget-amount "${BUDGET_USD}USD" --filter-projects "projects/$NUMBER" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
fi
echo "Setup done. Next: copy infra/vars/*.example.yaml to *.yaml, then make deploy-api deploy-reader deploy-geometry deploy-web."
