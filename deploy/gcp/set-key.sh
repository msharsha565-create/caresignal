#!/usr/bin/env bash
# Replace the Gemini API key used by the deployed CareSignal service.
# Tests the key against Gemini first, so a bad paste is caught before it is saved.
#
#   ./deploy/gcp/set-key.sh
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-caresignal}"
SECRET="caresignal-gemini-key"
MODEL="${CARESIGNAL_GEMINI_MODEL:-gemini-3.5-flash-lite}"

read -r -s -p "Gemini API key (input hidden): " KEY; echo
KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"
if [ -z "$KEY" ]; then echo "No key entered; nothing changed."; exit 1; fi

echo "Testing the key against Gemini..."
BODY="$(mktemp)"
CODE="$(curl -sS -o "$BODY" -w '%{http_code}' \
  https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
  -H "Authorization: Bearer ${KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say ok\"}]}")"
if [ "$CODE" != "200" ]; then
  echo "Gemini rejected the request (HTTP ${CODE}); nothing was changed. Gemini said:"
  cat "$BODY"; echo; rm -f "$BODY"; unset KEY
  exit 1
fi
rm -f "$BODY"
echo "Key works."

printf '%s' "$KEY" | gcloud secrets versions add "$SECRET" --data-file=- >/dev/null
unset KEY
# New revision so the running service picks up the new key.
gcloud run services update "$SERVICE" --region "$REGION" \
  --update-secrets "GEMINI_API_KEY=${SECRET}:latest" >/dev/null
echo "Saved. Reload the app and ask the agent again."
