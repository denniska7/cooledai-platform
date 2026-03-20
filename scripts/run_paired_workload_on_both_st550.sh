#!/usr/bin/env bash
# Run the same paired Llama workload (+5% intensity) on Pilot and Control.
#
# Run FROM your Mac (or any machine that can SSH to both servers).
# Requires: passwordless SSH (ssh-copy-id) to both hosts — do NOT put passwords in this repo.
#
# Defaults match COOLEDAI_NODE_SETUP.md (Tailscale pilot + LAN control).
# Override:
#   COOLEDAI_PILOT_SSH=cooledaiadmin@192.168.12.100 \  # optional LAN data port instead of Tailscale
#   COOLEDAI_CONTROL_SSH=cooledaiadmin@192.168.12.101 \
#   COOLEDAI_REMOTE_DIR=coolingai_simulator \
#   ./scripts/run_paired_workload_on_both_st550.sh
#
# Optional: run dual Ollama on each node first (same repo path):
#   ./scripts/run_dual_ollama_on_both_st550.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PILOT="${COOLEDAI_PILOT_SSH:-cooledaiadmin@100.92.29.44}"
CONTROL="${COOLEDAI_CONTROL_SSH:-cooledaiadmin@192.168.12.101}"
REMOTE_DIR_NAME="${COOLEDAI_REMOTE_DIR:-coolingai_simulator}"

remote_run_paired() {
  local host="$1"
  echo "========== ${host} =========="
  # Remote: ~ must expand on the server, not locally
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$host" "bash -lc 'set -e; cd \"\$HOME/${REMOTE_DIR_NAME}\" && bash scripts/start_st550_paired_workload.sh'" \
    || {
      echo "FAILED: ${host} — is the repo cloned at ~/${REMOTE_DIR_NAME} and is SSH key auth set up?" >&2
      return 1
    }
  echo ""
}

echo "Pilot:   $PILOT"
echo "Control: $CONTROL"
echo "Remote repo dir: ~/${REMOTE_DIR_NAME}"
echo ""

remote_run_paired "$PILOT"
remote_run_paired "$CONTROL"

echo "Done. On each node: tail -f /tmp/cooledai_llama_workload.log"
