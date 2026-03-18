#!/usr/bin/env bash
#
# setup_data_port_only.sh — Run on each ST550 to survive Wrench unplug
#
# Installs route enforcement so the data port (eno1np0 → router) always
# owns the default route. Run this ONCE while connected (Wrench or data).
#
# After setup:
#   - SSH via data port: ssh cooledaiadmin@192.168.12.100 (from wifi)
#   - Unplug Wrench anytime; telemetry and workload keep running
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="/opt/cooledai"

echo "[setup] Installing CooledAI data-port-only routing..."

# 1. Create install dir and copy scripts
sudo mkdir -p "$INSTALL_DIR"
sudo cp "$SCRIPT_DIR/enforce_st550_route.sh" "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR/enforce_st550_route.sh"

# 2. Run route fix now
echo "[setup] Applying route fix..."
sudo "$INSTALL_DIR/enforce_st550_route.sh"

# 3. Install systemd service + timer
sudo cp "$SCRIPT_DIR/cooledai-route.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/cooledai-route.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cooledai-route.service cooledai-route.timer
sudo systemctl start cooledai-route.timer

echo "[setup] Done. Default route locked to eno1np0 (data port)."
echo "[setup] Timer runs every 30s to prevent Wrench from hijacking."
echo ""
echo "  SSH via data port:  ssh cooledaiadmin@192.168.12.100"
echo "  (Use this from a machine on the same wifi — not the Wrench cable)"
echo ""
