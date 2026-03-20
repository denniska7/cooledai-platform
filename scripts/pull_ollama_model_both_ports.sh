#!/usr/bin/env bash
# After start_ollama_dual_gpu.sh, pull the workload model on BOTH Ollama instances
# (11434 and 11435). Ollama returns HTTP 404 on /api/generate if the model is missing.
#
# Usage on each node:
#   bash scripts/pull_ollama_model_both_ports.sh
#   OLLAMA_MODEL=llama3 bash scripts/pull_ollama_model_both_ports.sh
#
set -euo pipefail
MODEL="${OLLAMA_MODEL:-llama3}"
echo "[ollama-pull] model=$MODEL on :11434 (default) and :11435 (OLLAMA_HOST)"

ollama pull "$MODEL"

echo "[ollama-pull] second instance (GPU1)..."
OLLAMA_HOST="${OLLAMA_HOST_SECOND:-127.0.0.1:11435}" ollama pull "$MODEL"

echo "[ollama-pull] done. Check: ollama list && OLLAMA_HOST=127.0.0.1:11435 ollama list"
