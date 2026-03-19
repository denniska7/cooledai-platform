#!/bin/bash
# Quick access to telemetry from the cloud API (no SSH required).
# Usage: ./scripts/tail_telemetry.sh [hours]
#   hours: fetch telemetry for past N hours (default: 1)

HOURS="${1:-1}"
API_KEY="${COOLEDAI_API_KEY:-***REDACTED_API_KEY***}"
API_URL="${COOLEDAI_API_URL:-https://proactive-creativity-production.up.railway.app}"

echo "=== Telemetry logs (last ${HOURS}h): GPU temps, CPU temps, Fan RPM, GPU power ==="
curl -s -H "X-API-Key: $API_KEY" "${API_URL}/api/v1/telemetry-logs?hours=${HOURS}" | python3 -m json.tool
