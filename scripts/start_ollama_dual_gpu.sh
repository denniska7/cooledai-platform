#!/bin/bash
# Start two Ollama instances so workload spreads across both GPUs.
# GPU 0: default port 11434
# GPU 1: port 11435
#
# Then set OLLAMA_SPREAD_URLS="11434,11435" for llama_workload_scheduler.py
# to distribute inference evenly (avoids one GPU at 63°C, other at 30°C).

set -e
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# GPU 0 - default port
CUDA_VISIBLE_DEVICES=0 ollama serve &
PID0=$!
sleep 3

# GPU 1 - port 11435 (OLLAMA_HOST, not --port; some versions don't support --port)
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve &
PID1=$!

echo "Ollama GPU 0 (port 11434): PID $PID0"
echo "Ollama GPU 1 (port 11435): PID $PID1"
echo "Set OLLAMA_SPREAD_URLS=11434,11435 for workload scheduler"
