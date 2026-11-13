#!/usr/bin/env bash
# Deploy CareSignal to Google Cloud Run, using the Gemini API for the agent.
#
#   PROJECT_ID=my-project ./deploy/gcp/deploy.sh
#
# Optional: REGION (default us-central1), SERVICE (default caresignal),
#           PUBLIC=1 to make the URL reachable without a Google login.
# Run from the repository root, in Cloud Shell or anywhere `gcloud` is logged in.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-caresignal}"
SECRET="caresignal-gemini-key"
SA="caresignal-run@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

# The Gemini key lives in Secret Manager, never in the image or the repo.
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  read -r -s -p "Gemini API key (input hidden): " GEMINI_KEY; echo
  printf '%s' "$GEMINI_KEY" | gcloud secrets create "$SECRET" --data-file=- --replication-policy=automatic
  unset GEMINI_KEY
fi

# Dedicated identity for the service, allowed to read only that secret.
if ! gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create caresignal-run --display-name="CareSignal Cloud Run"
fi
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor" >/dev/null

# The app has no login of its own, so the service is private unless PUBLIC=1.
if [ "${PUBLIC:-0}" = "1" ]; then ACCESS="--allow-unauthenticated"; else ACCESS="--no-allow-unauthenticated"; fi

# Builds the Dockerfile with Cloud Build, then deploys the image.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --service-account "$SA" \
  --memory 2Gi --cpu 1 --timeout 300 \
  --min-instances 0 --max-instances 2 \
  --set-env-vars CARESIGNAL_PROVIDER=gemini \
  --set-secrets "GEMINI_API_KEY=${SECRET}:latest" \
  $ACCESS

echo
echo "Deployed: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
if [ "${PUBLIC:-0}" != "1" ]; then
  echo "The service is private. Open it locally with:"
  echo "  gcloud run services proxy $SERVICE --region $REGION --port 8080"
fi
