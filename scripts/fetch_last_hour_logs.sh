#!/bin/bash
# Fetch all logs from the past hour on both nodes (Pilot + Control).
# Run from your Mac. Pilot default: Tailscale 100.92.29.44; control: LAN .101.
# Override: COOLEDAI_PILOT_SSH=cooledaiadmin@192.168.12.100 COOLEDAI_CONTROL_SSH=...
#
# Usage: ./scripts/fetch_last_hour_logs.sh [output_dir]
# Output: writes to ./logs_last_hour_YYYYMMDD_HHMM/ or specified dir

set -e
NODE100="${COOLEDAI_PILOT_SSH:-cooledaiadmin@100.92.29.44}"
NODE101="${COOLEDAI_CONTROL_SSH:-cooledaiadmin@192.168.12.101}"
# ~360 agent lines/hour (10s), ~720 lenovo_live lines/hour (5s). Use 500/1000 to be safe.
AGENT_LINES=500
TELEMETRY_LINES=500
LENOVO_LINES=1000

OUT_DIR="${1:-./logs_last_hour_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT_DIR"

echo "Fetching last hour of logs from both nodes..."
echo "Output: $OUT_DIR"
echo ""

# Node 100 (Pilot)
echo "=== Node 100 (Pilot - CooledAI) ==="
ssh -o StrictHostKeyChecking=no "$NODE100" "
  echo '--- Agent log (last $AGENT_LINES lines) ---'
  tail -n $AGENT_LINES ~/cooledai_agent_pilot.log 2>/dev/null || tail -n $AGENT_LINES /tmp/cooledai_agent.log 2>/dev/null || echo '(no agent log)'
  echo ''
  echo '--- Telemetry log (last $TELEMETRY_LINES lines) ---'
  tail -n $TELEMETRY_LINES /var/log/cooledai_telemetry.log 2>/dev/null || tail -n $TELEMETRY_LINES /tmp/cooledai_telemetry.log 2>/dev/null || echo '(no telemetry log)'
  echo ''
  echo '--- Lenovo live (last $LENOVO_LINES lines, JSONL) ---'
  tail -n $LENOVO_LINES /var/log/lenovo_live.log 2>/dev/null || tail -n $LENOVO_LINES /tmp/lenovo_live.log 2>/dev/null || echo '(no lenovo_live log)'
" > "$OUT_DIR/node100_pilot.log" 2>&1 && echo "  -> $OUT_DIR/node100_pilot.log" || echo "  -> SSH failed (check network/password)"

# Node 101 (Control)
echo ""
echo "=== Node 101 (Control - Traditional) ==="
ssh -o StrictHostKeyChecking=no "$NODE101" "
  echo '--- Agent log (last $AGENT_LINES lines) ---'
  tail -n $AGENT_LINES ~/cooledai_agent_control.log 2>/dev/null || tail -n $AGENT_LINES /tmp/cooledai_agent.log 2>/dev/null || echo '(no agent log)'
  echo ''
  echo '--- Telemetry log (last $TELEMETRY_LINES lines) ---'
  tail -n $TELEMETRY_LINES /var/log/cooledai_telemetry.log 2>/dev/null || tail -n $TELEMETRY_LINES /tmp/cooledai_telemetry.log 2>/dev/null || echo '(no telemetry log)'
  echo ''
  echo '--- Lenovo live (last $LENOVO_LINES lines, JSONL) ---'
  tail -n $LENOVO_LINES /var/log/lenovo_live.log 2>/dev/null || tail -n $LENOVO_LINES /tmp/lenovo_live.log 2>/dev/null || echo '(no lenovo_live log)'
" > "$OUT_DIR/node101_control.log" 2>&1 && echo "  -> $OUT_DIR/node101_control.log" || echo "  -> SSH failed (check network/password)"

echo ""
echo "Done. View with:"
echo "  cat $OUT_DIR/node100_pilot.log"
echo "  cat $OUT_DIR/node101_control.log"
