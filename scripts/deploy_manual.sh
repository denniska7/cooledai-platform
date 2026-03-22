#!/bin/bash
# Deploy to both nodes by hand. You type the password each time scp/ssh ask.
# Run from your Mac (same network as nodes): ./scripts/deploy_manual.sh
set -e
cd "$(dirname "$0")/.."

echo "===== Node .100 (Pilot) ====="
echo "You will be prompted for password (2x scp, 1x ssh, 1x sudo)."
read -r -p "Press Enter to start SCP (files 1/2)..."
scp -o StrictHostKeyChecking=no \
  scripts/st550_telemetry.py scripts/start_telemetry.sh \
  cooledaiadmin@192.168.12.100:~/

read -r -p "Press Enter to start SCP (files 2/2)..."
scp -o StrictHostKeyChecking=no \
  scripts/llama_workload_scheduler.py scripts/predictive_engine.py scripts/cooledai_agent.py \
  cooledaiadmin@192.168.12.100:~/

echo "SSH to .100 — when sudo asks for password, type it and press Enter (may ask twice: once for start_telemetry, once for agent)."
read -r -p "Press Enter to SSH and run remote commands..."
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.100 'sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive; pkill -f llama_workload_scheduler.py 2>/dev/null || true; pkill -f predictive_engine.py 2>/dev/null || true; pkill -f cooledai_agent.py 2>/dev/null || true; nohup python3 ~/predictive_engine.py > /tmp/cooledai_predictive_engine.log 2>&1 & nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 & sudo apt-get install -y ipmitool 2>/dev/null; sudo nohup python3 ~/cooledai_agent.py --api-url https://proactive-creativity-production.up.railway.app --api-key ***REDACTED_API_KEY*** --node-id ST550-CooledAI-Predictive --ipmi-variant lenovo --shutdown-on-critical >> /tmp/cooledai_agent.log 2>&1 & sleep 2; echo Done on .100'

echo ""
echo "===== Node .101 (Control) ====="
echo "You will be prompted for password (2x scp, 1x ssh, 1x sudo)."
read -r -p "Press Enter to start SCP (files 1/2)..."
scp -o StrictHostKeyChecking=no \
  scripts/st550_telemetry.py scripts/start_telemetry.sh \
  cooledaiadmin@192.168.12.101:~/

read -r -p "Press Enter to start SCP (files 2/2)..."
scp -o StrictHostKeyChecking=no \
  scripts/llama_workload_scheduler.py \
  cooledaiadmin@192.168.12.101:~/

echo "SSH to .101 — when sudo asks for password, type it and press Enter."
read -r -p "Press Enter to SSH and run remote commands..."
ssh -t -o StrictHostKeyChecking=no cooledaiadmin@192.168.12.101 'sudo bash ~/start_telemetry.sh --node-id ST550-Control-Traditional; pkill -f llama_workload_scheduler.py 2>/dev/null || true; nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 & sleep 2; echo Done on .101'

echo ""
echo "Deploy finished. Both nodes updated."
