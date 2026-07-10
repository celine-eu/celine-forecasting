#!/usr/bin/env bash
#
# Local smoke test: fetches a JWT from the celine-policies Keycloak,
# then hits MLflow with it. Requires:
#   - Keycloak running at keycloak.celine.localhost  (celine-policies stack)
#   - MLflow running at localhost:5000               (docker compose up)
#
# Usage:
#   ./tests/smoke_local.sh              # default: admin user (is_admin=True)
#   ./tests/smoke_local.sh viewer       # viewer user (is_admin=False)
#   ./tests/smoke_local.sh editor       # editor user
#
set -euo pipefail

KC_BASE="http://keycloak.celine.localhost"
KC_REALM="celine"
KC_CLIENT_ID="oauth2_proxy"
KC_CLIENT_SECRET="oauth2_proxy"
MLFLOW_URL="http://localhost:5000"

USER="${1:-admin}"
PASS="${USER}"

echo "=== Fetching JWT for user '${USER}' from Keycloak ==="
TOKEN_RESPONSE=$(curl -sf \
  "${KC_BASE}/realms/${KC_REALM}/protocol/openid-connect/token" \
  -d "grant_type=password" \
  -d "client_id=${KC_CLIENT_ID}" \
  -d "client_secret=${KC_CLIENT_SECRET}" \
  -d "username=${USER}" \
  -d "password=${PASS}" \
  -d "scope=openid")

ACCESS_TOKEN=$(echo "${TOKEN_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "=== Token claims ==="
echo "${ACCESS_TOKEN}" | python3 -c "
import sys, json, base64
token = sys.stdin.read().strip()
payload = token.split('.')[1]
payload += '=' * (-len(payload) % 4)
claims = json.loads(base64.urlsafe_b64decode(payload))
print(json.dumps({k: claims[k] for k in ('sub','preferred_username','groups','organization') if k in claims}, indent=2))
"

echo ""
echo "=== Hitting MLflow API with JWT ==="
HTTP_CODE=$(curl -s -o /tmp/mlflow_response.json -w "%{http_code}" \
  -H "X-Auth-Request-Access-Token: ${ACCESS_TOKEN}" \
  "${MLFLOW_URL}/api/2.0/mlflow/experiments/search")

echo "HTTP ${HTTP_CODE}"
if [ "${HTTP_CODE}" = "200" ]; then
  echo "SUCCESS: Authenticated as '${USER}'"
  python3 -c "import json; print(json.dumps(json.load(open('/tmp/mlflow_response.json')), indent=2))" 2>/dev/null || cat /tmp/mlflow_response.json
else
  echo "Response:"
  cat /tmp/mlflow_response.json 2>/dev/null || true
fi

echo ""
echo "=== Testing without JWT (expect 401) ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  "${MLFLOW_URL}/api/2.0/mlflow/experiments/search")
echo "HTTP ${HTTP_CODE} (expected 401)"

echo ""
echo "=== Testing with service account (celine-cli) ==="
SA_TOKEN_RESPONSE=$(curl -sf \
  "${KC_BASE}/realms/${KC_REALM}/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=celine-cli" \
  -d "client_secret=celine-cli" \
  -d "scope=openid" 2>/dev/null || echo '{}')

SA_TOKEN=$(echo "${SA_TOKEN_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")

if [ -n "${SA_TOKEN}" ]; then
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-Auth-Request-Access-Token: ${SA_TOKEN}" \
    "${MLFLOW_URL}/api/2.0/mlflow/experiments/search")
  echo "HTTP ${HTTP_CODE} (service account celine-cli, expected 200)"
else
  echo "SKIP: celine-cli client not available"
fi
