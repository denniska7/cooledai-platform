#!/usr/bin/env bash
#
# test_connection.sh — Network continuity audit for ST550
#
# Verifies that eno1np0 (Router/Internet) can reach the internet.
# Run this to confirm the primary data path is alive before starting
# telemetry.  Safe to run from cron or systemd timer.
#
# Exit codes:
#   0 — success (eno1np0 can ping 8.8.8.8)
#   1 — failure (interface down, no route, or no reply)
#
# Usage:
#   bash scripts/test_connection.sh
#   bash scripts/test_connection.sh >> /var/log/cooledai_network.log 2>&1
#
set -euo pipefail

DATA_IFACE="${COOLEDAI_DATA_IFACE:-eno1np0}"
PING_TARGET="${COOLEDAI_PING_TARGET:-8.8.8.8}"
LOG_PREFIX="[network]"

# ── Verify primary route (eno1np0) vs secondary (enx/Wrench) ────────
DEFAULT_DEV=$(ip route show default 2>/dev/null | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
if [[ -n "$DEFAULT_DEV" && "$DEFAULT_DEV" != "$DATA_IFACE" ]]; then
    echo "$(date -Iseconds) $LOG_PREFIX WARN: Default route via $DEFAULT_DEV (expected $DATA_IFACE)"
fi

# ── Check interface exists and has an IP ─────────────────────────────
if ! ip -4 addr show "$DATA_IFACE" &>/dev/null; then
    echo "$(date -Iseconds) $LOG_PREFIX CRITICAL: $DATA_IFACE not found or has no IPv4"
    exit 1
fi

# ── Ping via the data interface ─────────────────────────────────────
if ping -c 1 -W 3 -I "$DATA_IFACE" "$PING_TARGET" &>/dev/null; then
    echo "$(date -Iseconds) $LOG_PREFIX SUCCESS: $DATA_IFACE → $PING_TARGET reachable"
    exit 0
else
    echo "$(date -Iseconds) $LOG_PREFIX FAIL: $DATA_IFACE → $PING_TARGET unreachable"
    exit 1
fi
