#!/usr/bin/env bash
# Deploy updated cooledai_agent to pilot and restart.
# Run from project root. Requires SSH access to pilot.
# Uses a remote script so ssh -t gets a TTY for sudo password prompt.
set -euo pipefail
cd "$(dirname "$0")/.."
PILOT="${PILOT:-cooledaiadmin@100.92.29.44}"
REMOTE_SCRIPT="/tmp/cooledai_deploy_agent_$$.sh"
echo "Deploying cooledai_agent to $PILOT ..."
scp -o StrictHostKeyChecking=no scripts/cooledai_agent.py "$PILOT":~/
cat > "$REMOTE_SCRIPT" << 'SCRIPT'
#!/bin/bash
sudo pkill -f cooledai_agent.py 2>/dev/null || true
sleep 2
sudo bash -c 'nohup python3 /home/cooledaiadmin/cooledai_agent.py \
  --api-url https://proactive-creativity-production.up.railway.app \
  --api-key ***REDACTED_API_KEY*** \
  --node-id ST550-CooledAI-Predictive \
  --ipmi-variant lenovo \
  >> /tmp/cooledai_agent.log 2>&1 &'
sleep 2
pgrep -af cooledai_agent || echo "(no process found)"
sudo tail -5 /tmp/cooledai_agent.log
echo "Done."
SCRIPT
scp -o StrictHostKeyChecking=no "$REMOTE_SCRIPT" "$PILOT:/tmp/cooledai_deploy_agent.sh"
rm -f "$REMOTE_SCRIPT"
ssh -t -o StrictHostKeyChecking=no "$PILOT" "bash /tmp/cooledai_deploy_agent.sh; rm -f /tmp/cooledai_deploy_agent.sh"
