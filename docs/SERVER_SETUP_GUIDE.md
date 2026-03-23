# CooledAI — Server Setup & Connection Guide

## Overview

This guide provides step-by-step instructions for connecting a client's servers to the CooledAI platform after an agreement is signed. It covers network connectivity options, agent installation, telemetry verification, and ongoing management. Designed as a universal procedure that works across any supported server vendor.

---

## Step 1: Establish Secure Network Connectivity

CooledAI requires a secure, persistent network path between the client's server nodes and the CooledAI cloud platform. Choose one of the following options based on the client's security requirements.

### Option A: Tailscale Private Gateway (Recommended)

Tailscale creates a zero-config WireGuard mesh VPN — no firewall changes needed, no exposed ports.

**On each managed node:**

```bash
# 1. Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Authenticate to the CooledAI tailnet (or client's own tailnet)
sudo tailscale up --authkey=tskey-auth-XXXXXXXX --hostname=cooledai-node-01

# 3. Verify connectivity
tailscale status
ping 100.x.x.x  # CooledAI management server
```

**Tailscale ACL Configuration (client's admin console):**

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:cooledai-mgmt"],
      "dst": ["tag:cooledai-node:22,443"]
    }
  ],
  "tagOwners": {
    "tag:cooledai-mgmt": ["admin@cooledai.com"],
    "tag:cooledai-node": ["admin@client.com"]
  }
}
```

**Key advantages:**
- No firewall changes or port forwarding required
- Works behind NAT, double NAT, and CGN
- End-to-end encrypted (WireGuard)
- Client retains full ACL control over what CooledAI can access
- SSH and API traffic stays within the private tailnet

### Option B: WireGuard Site-to-Site VPN

For clients who prefer self-managed VPN infrastructure.

**On each managed node:**

```bash
# 1. Install WireGuard
sudo apt install wireguard -y

# 2. Generate keys
wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey

# 3. Configure /etc/wireguard/wg-cooledai.conf
[Interface]
PrivateKey = <node-private-key>
Address = 10.200.0.2/24

[Peer]
PublicKey = <cooledai-gateway-pubkey>
Endpoint = gateway.cooledai.com:51820
AllowedIPs = 10.200.0.0/24
PersistentKeepalive = 25

# 4. Enable and start
sudo systemctl enable --now wg-quick@wg-cooledai
```

### Option C: SSH Reverse Tunnel (Minimal Setup)

For quick evaluation or air-gapped environments where VPN isn't available.

```bash
# On each node: create a persistent reverse tunnel
# CooledAI SSH server assigns a unique port per node
autossh -M 0 -f -N -R <assigned-port>:localhost:22 \
  tunnel@gateway.cooledai.com \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -i /etc/cooledai/tunnel_key
```

### Option D: Direct Internet (HTTPS Only)

If the client allows outbound HTTPS, no VPN is needed — the agent POSTs telemetry directly to the CooledAI API.

```bash
# Verify outbound HTTPS connectivity
curl -s https://api.cooledai.com/health
# Expected: {"status": "ok"}
```

**Requirements:**
- Outbound HTTPS (port 443) to `api.cooledai.com`
- No inbound ports needed
- Agent authenticates with API key (no SSH access required)

---

## Step 2: Verify BMC/IPMI Access

The CooledAI agent controls fans via the server's Baseboard Management Controller (BMC). This must be accessible from the OS.

```bash
# 2.1 Identify BMC IP (usually on a dedicated management NIC)
ipmitool lan print 1 | grep "IP Address"

# 2.2 Test IPMI-over-LAN from the OS
ipmitool -I lanplus -H <bmc_ip> -U <bmc_user> -P <bmc_pass> power status
# Expected: "Chassis Power is on"

# 2.3 Test Redfish API
curl -sk -u <bmc_user>:<bmc_pass> https://<bmc_ip>/redfish/v1/Systems/1 | python3 -m json.tool | head -20

# 2.4 Verify fan control works (CRITICAL — test before deploying agent)
# Read current fan speed
ipmitool -I lanplus -H <bmc_ip> -U <bmc_user> -P <bmc_pass> sdr type Fan

# Attempt to set fan speed (vendor-specific command)
# Lenovo XCC:
ipmitool -I lanplus -H <bmc_ip> -U <bmc_user> -P <bmc_pass> raw 0x3a 0x07 0x01 0x28
# (0x28 = 40%)

# Wait 30 seconds, then re-read — RPM MUST change
sleep 30
ipmitool -I lanplus -H <bmc_ip> -U <bmc_user> -P <bmc_pass> sdr type Fan

# If RPMs did NOT change: check BIOS operating mode (see Step 3)
```

---

## Step 3: Configure BIOS Operating Mode

Many server vendors ship with a BIOS thermal mode that overrides external fan commands. This MUST be set correctly before the agent can control fans.

### Lenovo (XCC)

```bash
# Check current mode
curl -sk -u <user>:<pass> https://<bmc_ip>/redfish/v1/Systems/1/Bios | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['Attributes'].get('OperatingModes_ChooseOperatingMode', 'unknown'))"

# If "Efficiency_FavorPower" — change to CustomMode:
curl -sk -u <user>:<pass> \
  -X PATCH \
  -H "Content-Type: application/json" \
  -d '{"Attributes": {"OperatingModes_ChooseOperatingMode": "CustomMode"}}' \
  https://<bmc_ip>/redfish/v1/Systems/1/Bios/Pending

# Reboot required for BIOS change to take effect:
curl -sk -u <user>:<pass> \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"ResetType": "ForceRestart"}' \
  https://<bmc_ip>/redfish/v1/Systems/1/Actions/ComputerSystem.Reset
```

### Dell (iDRAC)

```bash
# Check thermal profile
racadm get System.ThermalSettings.ThermalProfile

# Set to custom:
racadm set System.ThermalSettings.ThermalProfile 4

# Disable third-party PCIe card thermal override:
racadm set System.ThermalSettings.ThirdPartyPCIFanResponse 0
```

### HPE (iLO)

```bash
# Check via Redfish
curl -sk -u <user>:<pass> https://<bmc_ip>/redfish/v1/Chassis/1/Thermal | python3 -m json.tool | grep -i fanmode

# Set via iLO web UI: System Information > Fan > Optimal Cooling
```

---

## Step 4: Install GPU Drivers & Tools

```bash
# 4.1 Verify NVIDIA driver is loaded
nvidia-smi

# 4.2 Verify nvidia-smi reports all expected GPUs
nvidia-smi --query-gpu=index,gpu_name,driver_version,power.limit --format=csv

# 4.3 If nvidia-smi is missing, install drivers:
# Ubuntu/Debian:
sudo apt install nvidia-driver-550 nvidia-utils-550 -y

# RHEL/Rocky:
sudo dnf install nvidia-driver nvidia-settings -y
```

---

## Step 5: Install the CooledAI Agent

```bash
# 5.1 Create directories
sudo mkdir -p /opt/cooledai /etc/cooledai /var/log/cooledai

# 5.2 Deploy agent code (from CooledAI delivery package)
sudo tar xzf cooledai-agent-latest.tar.gz -C /opt/cooledai/

# 5.3 Create environment configuration
sudo tee /etc/cooledai/agent.env > /dev/null << 'EOF'
# BMC / IPMI Configuration
XCC_BMC_HOST=<bmc_ip>
XCC_BMC_USER=<bmc_user>
XCC_BMC_PASS=<bmc_pass>

# Node Identity
NODE_ID=<unique-node-name>
NODE_ROLE=pilot

# CooledAI API
COOLEDAI_API_URL=https://api.cooledai.com
COOLEDAI_API_KEY=<provided-api-key>

# Fan Controller Settings
FAN_RATED_MAX_RPM=7000
FAN_BOOTSTRAP_FLOOR_PCT=30
CALIBRATION_WINDOW_MIN=30

# GPU Configuration
GPU_COUNT=2
GPU_POWER_LIMIT_W=75
EOF

sudo chmod 600 /etc/cooledai/agent.env

# 5.4 Install systemd service
sudo tee /etc/systemd/system/cooledai-agent.service > /dev/null << 'EOF'
[Unit]
Description=CooledAI Thermal Optimization Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/cooledai/agent.env
ExecStart=/usr/bin/python3 /opt/cooledai/scripts/cooledai_agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cooledai-agent
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

# 5.5 Install dependencies
sudo apt install -y ipmitool python3-pip 2>/dev/null || \
sudo dnf install -y ipmitool python3-pip 2>/dev/null
pip3 install pynvml requests

# 5.6 Delete any stale calibration profile
sudo rm -f /etc/cooledai/calibration_profile.json

# 5.7 Start the agent
sudo systemctl enable --now cooledai-agent

# 5.8 Verify it's running
systemctl status cooledai-agent
journalctl -u cooledai-agent -f --no-pager | head -30
```

---

## Step 6: Install Telemetry Reporter

The telemetry script runs alongside the agent and POSTs GPU/CPU/fan data to the CooledAI portal.

```bash
# 6.1 Copy telemetry script
sudo cp /opt/cooledai/scripts/st550_telemetry.py /opt/cooledai/

# 6.2 Start telemetry
sudo bash /opt/cooledai/scripts/start_telemetry.sh

# 6.3 Verify telemetry is posting
tail -20 /var/log/cooledai_telemetry.log
# Should show: [telemetry] POST 200 — successfully posting to API
```

---

## Step 7: Set Up Workload Simulator (Evaluation Only)

During the evaluation period, run an identical benchmark on both pilot (CooledAI) and control (traditional) nodes.

```bash
# 7.1 Install Ollama (GPU inference engine)
curl -fsSL https://ollama.com/install.sh | sh

# 7.2 Start dual-GPU Ollama instances
# GPU 0:
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > /tmp/ollama_gpu0.log 2>&1 &
sleep 3

# GPU 1:
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 OLLAMA_MODELS=/opt/ollama_gpu1/models \
  nohup ollama serve > /tmp/ollama_gpu1.log 2>&1 &
sleep 3

# 7.3 Pull the benchmark model on both instances
ollama pull llama3.2:3b
OLLAMA_HOST=0.0.0.0:11435 ollama pull llama3.2:3b

# 7.4 Start the benchmark (SAME command on both nodes)
nohup python3 /opt/cooledai/scripts/thermal_workload_benchmark.py \
  --model llama3.2:3b --seed 42 --ports 11434,11435 \
  > /tmp/workload_benchmark.log 2>&1 &
```

---

## Step 8: Verify Everything on the Portal

1. Log in to **https://portal.cooledai.com**
2. Navigate to **Live Comparison**
3. Confirm both nodes are reporting:
   - Green "Live" badge is active
   - GPU Temp, CPU Temp, Fan Speed, GPU Power all show data
   - CooledAI (green) and Control (red) lines are both visible
4. Check the **1H** view first for real-time data flow
5. After 24 hours, check **24H** view for trends
6. After 7 days, check **7D** view for the full evaluation picture

---

## Step 9: Ongoing Client Management

### Health Monitoring

```bash
# Check agent status
systemctl status cooledai-agent

# Check telemetry status
pgrep -a st550_telemetry

# Check recent agent decisions
journalctl -u cooledai-agent --since "10 minutes ago" --no-pager | grep FAN_DIAG | tail -5

# Check calibration profile health
cat /etc/cooledai/calibration_profile.json | python3 -m json.tool | grep -E "idle_rpm|ceiling_rpm|schema"
```

### Common Maintenance Tasks

| Task | Command |
|------|---------|
| Restart agent | `sudo systemctl restart cooledai-agent` |
| Delete stale profile | `sudo rm /etc/cooledai/calibration_profile.json && sudo systemctl restart cooledai-agent` |
| View live fan control | `journalctl -u cooledai-agent -f \| grep FAN_DIAG` |
| Check GPU temps | `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader` |
| Rotate telemetry log | `sudo mv /var/log/cooledai_telemetry.log /var/log/cooledai/archive/$(date +%Y%m%d).log && sudo bash /opt/cooledai/scripts/start_telemetry.sh` |
| Update agent code | `sudo systemctl stop cooledai-agent && sudo tar xzf cooledai-agent-vX.Y.Z.tar.gz -C /opt/cooledai/ && sudo systemctl start cooledai-agent` |

### Tailscale Node Management

```bash
# Check node connectivity
tailscale status

# Re-authenticate if expired
sudo tailscale up --authkey=tskey-auth-XXXXXXXX

# Remove a node from the tailnet (from admin console)
tailscale admin remove <node-id>
```

### Escalation Path

1. **Portal shows no data** → Check telemetry: `pgrep st550_telemetry`, check logs
2. **Fans not optimizing** → Check agent: `journalctl -u cooledai-agent -f`, verify BIOS mode
3. **Profile keeps getting rejected** → Delete profile, restart agent, monitor 30-min calibration
4. **Node unreachable** → Check Tailscale: `tailscale status`, check SSH: `ssh cooledaiadmin@<tailscale_ip>`
5. **GPU overheating** → Agent has automatic failsafe; if XCC command fails for 120s, reverts to BMC auto control

---

## Quick Reference Checklist

```
[ ] Network connectivity established (Tailscale / WireGuard / SSH tunnel / HTTPS)
[ ] BMC/IPMI accessible from OS (ipmitool works)
[ ] Fan control commands verified (RPMs actually change)
[ ] BIOS operating mode set correctly (CustomMode for Lenovo)
[ ] GPU drivers installed (nvidia-smi works)
[ ] Agent installed and configured (/etc/cooledai/agent.env)
[ ] Agent running (systemctl status cooledai-agent)
[ ] Telemetry posting to API (check portal for data)
[ ] Stale profile deleted (fresh calibration window)
[ ] Portal shows live data for this node
[ ] Workload benchmark running (evaluation period only)
[ ] Control node set up identically (for A/B comparison)
```
