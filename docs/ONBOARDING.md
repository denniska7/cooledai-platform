# CooledAI First-Run Onboarding Guide

This guide walks you through deploying CooledAI on a new server, from API key generation to verified operation.

## Prerequisites

- Server with NVIDIA GPUs and IPMI/Redfish-capable BMC
- Network access to BMC management interface
- Network access to CooledAI API (HTTPS outbound)
- Python 3.10+ installed on the server
- Root or sudo access for initial setup

## Step 1: Generate an API Key

Contact CooledAI to receive your client API key, or generate one via the admin endpoint:

```bash
curl -X POST https://api.cooledai.com/admin/keys \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"client_name": "Your Company", "owner_id": "your-client-id"}'
```

Save the returned key securely — it is shown only once.

## Step 2: Configure cooledai.yaml

Copy the template and fill in your values:

```bash
sudo mkdir -p /etc/cooledai
sudo cp configs/cooledai.yaml /etc/cooledai/cooledai.yaml
sudo nano /etc/cooledai/cooledai.yaml
```

Required fields:
```yaml
server:
  client_name: "Your Company"
  site: "Your Site Name"
  rack: "Rack ID"

hardware:
  gpu_count: 2          # Number of GPUs in this server
  max_fan_rpm: 7000     # Check your server's fan specs

alerts:
  warning_temp_c: 65    # GPU temp warning threshold
  critical_temp_c: 85   # GPU temp critical threshold
  webhook_url: ""       # Optional: Slack/Teams/PagerDuty webhook
```

## Step 3: Configure Agent Environment

```bash
sudo nano /etc/cooledai/agent.env
```

```bash
COOLEDAI_API_KEY=sk-your-api-key-here
COOLEDAI_API_URL=https://api.cooledai.com
COOLEDAI_NODE_ID=your-server-hostname
XCC_BMC_HOST=169.254.95.118        # Your BMC IP address
XCC_BMC_USER=USERID                 # Your BMC username
XCC_BMC_PASS=your-bmc-password      # Your BMC password
COOLEDAI_CONFIG=/etc/cooledai/cooledai.yaml
```

Set permissions:
```bash
sudo chmod 600 /etc/cooledai/agent.env
```

## Step 4: Run Preflight Checks

```bash
source /etc/cooledai/agent.env
python3 scripts/cooledai_preflight.py --config /etc/cooledai/cooledai.yaml
```

Expected output:
```
============================================================
  CooledAI Preflight Check
============================================================

[1/4] Validating config: /etc/cooledai/cooledai.yaml
  [PASS] Config is valid

[2/4] Testing XCC BMC connectivity
  [PASS] XCC BMC reachable

[3/4] Checking API health endpoint
  [PASS] API is responsive

[4/4] Sending test telemetry record
  [PASS] Telemetry accepted

All checks passed. CooledAI is ready for deployment.
```

Fix any failing checks before proceeding.

## Step 5: Create Service User

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin cooledai
sudo mkdir -p /var/lib/cooledai /var/log/cooledai
sudo chown cooledai:cooledai /var/lib/cooledai /var/log/cooledai
```

## Step 6: Install and Start the Agent

```bash
sudo cp scripts/cooledai-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cooledai-agent
sudo systemctl start cooledai-agent
```

## Step 7: Verify Operation

```bash
# Check service status
sudo systemctl status cooledai-agent

# Check telemetry is flowing
tail -f /var/log/cooledai_telemetry.log

# Check audit log
tail -f /var/log/cooledai/audit.log

# Verify on portal
# Visit https://cooledai.com/portal and confirm your node appears
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Agent won't start | `journalctl -u cooledai-agent -f` |
| No telemetry | Verify API key and URL in agent.env |
| BMC unreachable | Ping BMC IP; check credentials |
| Permission denied | Ensure agent.env has correct ownership |
| Rate limited (429) | Reduce telemetry frequency or contact support |

## Support

- Documentation: https://cooledai.com/docs
- Email: support@cooledai.com
