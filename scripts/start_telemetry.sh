#!/usr/bin/env bash
#
# start_telemetry.sh — single-instance launcher for st550_telemetry.py
#
# Kills any existing instance, then starts a fresh one under nohup.
# Safe to call repeatedly (cron, systemd, or manual).
#
# Network topology:
#   eno1np0         → Router / Internet  (gw 192.168.12.1)
#   enx9a29a6faed33 → Laptop / Wrench   (direct link, no internet)
#
# This script ensures eno1np0 owns the default route so telemetry
# traffic never accidentally egresses via the Wrench port.
#
# Usage:
#   sudo bash scripts/start_telemetry.sh                    # defaults
#   sudo NODE_ID=ST550-Shadow-Node bash scripts/start_telemetry.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TELEMETRY_PY="${SCRIPT_DIR}/st550_telemetry.py"
LOG_FILE="/var/log/cooledai_telemetry.log"
PID_FILE="/tmp/cooledai_telemetry.pid"

DATA_IFACE="eno1np0"
ROUTER_GW="192.168.12.1"

# ── Ensure default route goes through the data port ─────────────────
CURRENT_GW_IFACE=$(ip route show default 2>/dev/null | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')

if [ "$CURRENT_GW_IFACE" != "$DATA_IFACE" ]; then
    echo "[start_telemetry] Default route is via '$CURRENT_GW_IFACE', fixing → $DATA_IFACE (gw $ROUTER_GW)"
    # Remove any existing defaults that go through the wrong interface
    ip route del default 2>/dev/null || true
    ip route add default via "$ROUTER_GW" dev "$DATA_IFACE" metric 100
    echo "[start_telemetry] Default route set: default via $ROUTER_GW dev $DATA_IFACE metric 100"
else
    echo "[start_telemetry] Default route already via $DATA_IFACE — OK"
fi

# ── Kill any running instance ───────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[start_telemetry] Stopping existing instance (PID $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Belt-and-suspenders: kill anything matching the script name
pkill -f "st550_telemetry.py" 2>/dev/null || true
sleep 1

# ── Verify the script exists ───────────────────────────────────────
if [ ! -f "$TELEMETRY_PY" ]; then
    echo "[start_telemetry] ERROR: $TELEMETRY_PY not found" >&2
    exit 1
fi

# ── Ensure log directory is writable ────────────────────────────────
LOG_DIR="$(dirname "$LOG_FILE")"
if [ ! -w "$LOG_DIR" ]; then
    LOG_FILE="/tmp/cooledai_telemetry.log"
    echo "[start_telemetry] /var/log not writable, logging to $LOG_FILE"
fi

# ── Launch under nohup (use system python for pip/nvidia-ml-py3) ────
PYTHON="${PYTHON:-/usr/bin/python3}"
PYTHONUNBUFFERED=1 nohup "$PYTHON" "$TELEMETRY_PY" "$@" >> "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "[start_telemetry] Started st550_telemetry.py (PID $NEW_PID)"
echo "[start_telemetry] Logs: $LOG_FILE"
echo "[start_telemetry] PID file: $PID_FILE"
echo "[start_telemetry] To stop:  kill $NEW_PID  (or bash $0 will auto-stop before restart)"
