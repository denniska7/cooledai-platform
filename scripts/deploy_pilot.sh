#!/usr/bin/env bash
#
# Deploy CooledAI pilot node (cooledai-srv) — copy all required files.
# Run from project root. SSH to pilot and run setup manually (see COOLEDAI_NODE_SETUP.md).
#
# Usage:
#   ./scripts/deploy_pilot.sh
#   PILOT_HOST=192.168.12.100 ./scripts/deploy_pilot.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PILOT_HOST="${PILOT_HOST:-100.92.29.44}"
PILOT_USER="${PILOT_USER:-cooledaiadmin}"

FILES=(
  scripts/st550_telemetry.py
  scripts/start_telemetry.sh
  scripts/llama_workload_scheduler.py
  scripts/predictive_engine.py
  scripts/cooledai_agent.py
  scripts/start_ollama_dual_gpu.sh
)

echo "=== Deploying CooledAI pilot to ${PILOT_USER}@${PILOT_HOST} ==="
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found" >&2
    exit 1
  fi
done

scp -o StrictHostKeyChecking=no "${FILES[@]}" "${PILOT_USER}@${PILOT_HOST}:~/"

echo ""
echo "Done. Next steps:"
echo "  1. ssh ${PILOT_USER}@${PILOT_HOST}"
echo "  2. Follow COOLEDAI_NODE_SETUP.md section 2 (or run the one-liner from section 4)"
echo ""
