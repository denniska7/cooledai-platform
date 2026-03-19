#!/bin/bash
# Quick access to recent telemetry from the cloud API (no SSH required).
# Usage: ./scripts/tail_telemetry.sh

API_KEY="${COOLEDAI_API_KEY:-sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo}"
API_URL="${COOLEDAI_API_URL:-https://proactive-creativity-production.up.railway.app}"

echo "=== Cloud telemetry (nodes/status) ==="
curl -s -H "X-API-Key: $API_KEY" "${API_URL}/api/v1/nodes/status" | python3 -m json.tool
