#!/usr/bin/env bash
# ST550 Pilot (.100) + Control (.101) — run THE SAME workload for a fair compare.
#
# Uneven GPU power (e.g. 16.7W vs 9.6W avg) is usually:
#   - Control (or Pilot) missing OLLAMA_SPREAD_URLS → all inference on GPU0
#   - Scheduler not running on one host
#   - Different COOLEDAI_WORKLOAD_INTENSITY / OLLAMA_MODEL between nodes
#
# Usage (on BOTH servers, after dual Ollama is up — see start_ollama_dual_gpu.sh):
#   bash scripts/start_st550_paired_workload.sh
#
# Optional overrides (must match on both nodes):
#   COOLEDAI_WORKLOAD_INTENSITY=1.05   # default; was raised +5% from prior baselines
#   OLLAMA_MODEL=llama3
#   SCHEDULER_LOG=/tmp/cooledai_llama_workload.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCHEDULER="${REPO_ROOT}/scripts/llama_workload_scheduler.py"
LOG_FILE="${SCHEDULER_LOG:-/tmp/cooledai_llama_workload.log}"

export OLLAMA_SPREAD_URLS="${OLLAMA_SPREAD_URLS:-11434,11435}"
export COOLEDAI_WORKLOAD_INTENSITY="${COOLEDAI_WORKLOAD_INTENSITY:-1.05}"

if [[ ! -f "$SCHEDULER" ]]; then
  echo "ERROR: $SCHEDULER not found" >&2
  exit 1
fi

echo "[paired-workload] OLLAMA_SPREAD_URLS=$OLLAMA_SPREAD_URLS COOLEDAI_WORKLOAD_INTENSITY=$COOLEDAI_WORKLOAD_INTENSITY"
echo "[paired-workload] logging to $LOG_FILE"

pkill -f "llama_workload_scheduler.py" 2>/dev/null || true
sleep 1

PYTHON="${PYTHON:-python3}"
nohup "$PYTHON" "$SCHEDULER" >>"$LOG_FILE" 2>&1 &
echo "[paired-workload] started llama_workload_scheduler.py PID $!"
echo "[paired-workload] tail -f $LOG_FILE"
