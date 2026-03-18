#!/usr/bin/env bash
#
# enforce_st550_route.sh — Lock default route to eno1np0 (Data Port / Router)
#
# Ensures the ST550's data port (eno1np0 → wifi router) ALWAYS owns the
# default route. Prevents the Wrench port (enx*) from hijacking it when
# connected, so you can unplug the Wrench without losing SSH or telemetry.
#
# Run at boot via systemd, and every 30s via timer (in case Wrench adds a route).
# Safe to run manually anytime.
#
set -euo pipefail

DATA_IFACE="eno1np0"
ROUTER_GW="192.168.12.1"

# ── Remove any default route NOT via the data port ─────────────────────
while true; do
    CURRENT_DEV=$(ip route show default 2>/dev/null | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' || true)
    if [[ -z "$CURRENT_DEV" ]]; then
        break
    fi
    if [[ "$CURRENT_DEV" == "$DATA_IFACE" ]]; then
        exit 0
    fi
    echo "[route] Removing wrong default via $CURRENT_DEV (Wrench?)"
    ip route del default 2>/dev/null || true
done

# ── Add default via data port ─────────────────────────────────────────
if ! ip route show default dev "$DATA_IFACE" &>/dev/null; then
    echo "[route] Adding default via $ROUTER_GW dev $DATA_IFACE"
    ip route add default via "$ROUTER_GW" dev "$DATA_IFACE" metric 100
fi
