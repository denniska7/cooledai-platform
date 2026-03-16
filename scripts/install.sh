#!/usr/bin/env bash
# -------------------------------------------------------------------
# CooledAI Agent — One-Line Installer
#
# Downloads the unified agent, writes credentials to an env file,
# installs a systemd service, and starts it.
#
# Usage:
#   curl -fsSL https://api.cooledai.com/install.sh | sudo bash -s -- \
#     --api-key "sk-..." --node-id "rack-12-node-03"
#
#   # Or interactively (prompts for key and node id):
#   sudo bash install.sh
# -------------------------------------------------------------------

set -euo pipefail

API_URL_DEFAULT="https://api.cooledai.com"
AGENT_URL="${COOLEDAI_AGENT_URL:-}"
INSTALL_DIR="/opt/cooledai"
ENV_DIR="/etc/cooledai"
ENV_FILE="${ENV_DIR}/agent.env"
SERVICE_FILE="/etc/systemd/system/cooledai-agent.service"
STATE_DIR="/var/lib/cooledai"

API_KEY=""
NODE_ID=""
API_URL="${API_URL_DEFAULT}"

# ---- Parse arguments ------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)  API_KEY="$2";  shift 2 ;;
        --node-id)  NODE_ID="$2";  shift 2 ;;
        --api-url)  API_URL="$2";  shift 2 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Resolve agent download URL (derives from API_URL) ---------------

if [[ -z "$AGENT_URL" ]]; then
    AGENT_URL="${API_URL}/cooledai_agent.py"
fi

# ---- Interactive prompts if not provided ----------------------------

if [[ -z "$API_KEY" ]]; then
    read -rp "CooledAI API Key: " API_KEY
fi
if [[ -z "$API_KEY" ]]; then
    echo "Error: API key is required."
    exit 1
fi

if [[ -z "$NODE_ID" ]]; then
    default_node="$(hostname -s 2>/dev/null || echo 'node-01')"
    read -rp "Node ID [${default_node}]: " NODE_ID
    NODE_ID="${NODE_ID:-$default_node}"
fi

# ---- Preflight checks ----------------------------------------------

echo ""
echo "=== CooledAI Agent Installer ==="
echo ""

# Must be root
if [[ "$(id -u)" -ne 0 ]]; then
    echo "Error: This installer must be run as root (sudo)."
    exit 1
fi

# Python 3.8+
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found."
    echo "  Ubuntu/Debian:  sudo apt install python3"
    echo "  RHEL/CentOS:    sudo yum install python3"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 8 ]]; }; then
    echo "Error: Python 3.8+ required (found ${PY_VERSION})."
    exit 1
fi
echo "[ok] Python ${PY_VERSION}"

# ---- Download agent -------------------------------------------------

mkdir -p "$INSTALL_DIR" "$ENV_DIR" "$STATE_DIR"

echo "[..] Downloading agent from ${AGENT_URL}"
if command -v curl &>/dev/null; then
    curl -fsSL -o "${INSTALL_DIR}/cooledai_agent.py" "$AGENT_URL"
elif command -v wget &>/dev/null; then
    wget -qO "${INSTALL_DIR}/cooledai_agent.py" "$AGENT_URL"
else
    echo "Error: Neither curl nor wget found."
    exit 1
fi
chmod 755 "${INSTALL_DIR}/cooledai_agent.py"
echo "[ok] Agent installed to ${INSTALL_DIR}/cooledai_agent.py"

# ---- Write env file -------------------------------------------------

cat > "$ENV_FILE" <<ENVEOF
COOLEDAI_API_KEY=${API_KEY}
COOLEDAI_NODE_ID=${NODE_ID}
COOLEDAI_API_URL=${API_URL}
ENVEOF

chmod 600 "$ENV_FILE"
echo "[ok] Credentials written to ${ENV_FILE} (mode 600)"

# ---- Install systemd service ----------------------------------------

cat > "$SERVICE_FILE" <<'SVCEOF'
[Unit]
Description=CooledAI Thermal Control Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/cooledai/agent.env
ExecStart=/usr/bin/python3 /opt/cooledai/cooledai_agent.py \
    --api-key ${COOLEDAI_API_KEY} \
    --node-id ${COOLEDAI_NODE_ID} \
    --api-url ${COOLEDAI_API_URL}
Restart=always
RestartSec=10
WatchdogSec=300
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cooledai-agent

[Install]
WantedBy=multi-user.target
SVCEOF

echo "[ok] systemd unit installed to ${SERVICE_FILE}"

# ---- Enable and start -----------------------------------------------

systemctl daemon-reload
systemctl enable cooledai-agent.service
systemctl start cooledai-agent.service

echo ""
echo "=== Installation complete ==="
echo ""
echo "  Node ID :  ${NODE_ID}"
echo "  Config  :  ${ENV_FILE}"
echo "  Agent   :  ${INSTALL_DIR}/cooledai_agent.py"
echo "  Service :  cooledai-agent.service"
echo ""
echo "  Check status :  systemctl status cooledai-agent"
echo "  View logs    :  journalctl -u cooledai-agent -f"
echo "  Stop         :  sudo systemctl stop cooledai-agent"
echo "  Uninstall    :  sudo systemctl disable --now cooledai-agent && sudo rm -rf ${INSTALL_DIR} ${ENV_DIR} ${SERVICE_FILE}"
echo ""
