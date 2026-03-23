#!/bin/bash
# Restart thermal_workload_benchmark.py on both nodes simultaneously

NODE1_USER="cooledaiadmin"
NODE1_HOST="100.92.29.44"
NODE2_USER="cooleda"
NODE2_HOST="100.92.29.94"
MODEL="llama3.2:3b"
SCRIPT="~/thermal_workload_benchmark.py"
LOG="~/workload_benchmark.log"

RESTART_CMD="pkill -f thermal_workload_benchmark.py; pkill -f workload_sim.py; pkill -f llama_workload_scheduler.py; sleep 1; setsid python3 $SCRIPT --model $MODEL > $LOG 2>&1 & echo \"[\$(hostname)] Started PID \$!\""

echo "Restarting workload on both nodes..."

ssh ${NODE1_USER}@${NODE1_HOST} "$RESTART_CMD" &
ssh ${NODE2_USER}@${NODE2_HOST} "$RESTART_CMD" &

wait
echo "Done. Verifying..."
sleep 2

ssh ${NODE1_USER}@${NODE1_HOST} "ps aux | grep thermal_workload | grep -v grep || echo '[.44] NOT running'" &
ssh ${NODE2_USER}@${NODE2_HOST} "ps aux | grep thermal_workload | grep -v grep || echo '[.94] NOT running'" &

wait
