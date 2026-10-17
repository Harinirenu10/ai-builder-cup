#!/usr/bin/env bash
# Fails loudly if the deployed project does not match the spec: scaling, geometry size, queue, budget alert, hosting rewrite.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
: "${BILLING_ACCOUNT:?set BILLING_ACCOUNT}"
bad=0
fail() { echo "FAIL: $*" >&2; bad=1; }
for key in api reader geometry; do
  name="$(svc_name "$key")"
  gcloud run services describe "$name" --region "$REGION" --project "$PROJECT" --format=json |
    python "$(dirname "${BASH_SOURCE[0]}")/check_service.py" "$key" "$name" || bad=1
done
if gcloud tasks queues describe "$TASKS_QUEUE" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  echo "queue $TASKS_QUEUE OK"
else
  fail "queue $TASKS_QUEUE missing"
fi
if gcloud billing budgets list --billing-account "$BILLING_ACCOUNT" --format='value(displayName)' | grep -qx "fit-demo-$PROJECT"; then
  echo "budget fit-demo-$PROJECT OK"
else
  fail "budget alert missing"
fi
if grep -q "\"serviceId\": \"$API_SERVICE\"" firebase.json; then
  echo "hosting rewrite -> $API_SERVICE OK"
else
  fail "firebase.json /api rewrite does not point at $API_SERVICE"
fi
exit $bad
