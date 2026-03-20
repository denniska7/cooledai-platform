#!/usr/bin/env bash
# Start dual-GPU Ollama (11434 + 11435) on Pilot and Control.
#
# Run FROM your Mac. Requires passwordless SSH — same env vars as run_paired_workload_on_both_st550.sh
#
# WARNING: This kills existing `ollama serve` processes on each node.
#
set -euo pipefail

PILOT="${COOLEDAI_PILOT_SSH:-cooledaiadmin@100.92.29.44}"
CONTROL="${COOLEDAI_CONTROL_SSH:-cooledaiadmin@192.168.12.101}"
REMOTE_DIR_NAME="${COOLEDAI_REMOTE_DIR:-coolingai_simulator}"

remote_dual_ollama() {
  local host="$1"
  echo "========== dual Ollama: ${host} =========="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$host" "bash -lc 'set -e; cd \"\$HOME/${REMOTE_DIR_NAME}\" && bash scripts/start_ollama_dual_gpu.sh'"
  echo ""
}

echo "Pilot:   $PILOT"
echo "Control: $CONTROL"
echo ""

remote_dual_ollama "$PILOT"
remote_dual_ollama "$CONTROL"

echo "Done. Pull the same model on each host (see scripts/GPU_LOAD_BALANCE.md), then run:"
echo "  ./scripts/run_paired_workload_on_both_st550.sh"
