# CooledAI Security Model

## Architecture Overview

CooledAI operates exclusively at the BMC/IPMI out-of-band management
layer. The agent runs on the monitored server but communicates only with:

1. **The BMC** (Baseboard Management Controller) via IPMI/Redfish over the
   dedicated management network interface
2. **The CooledAI gateway** on port 8080 (or cloud API via HTTPS)

No other network connections are permitted.

## Network Access Requirements

| Destination | Port | Protocol | Purpose |
|---|---|---|---|
| BMC management IP | 623 | IPMI/UDP | Fan control, sensor reads |
| BMC management IP | 443 | HTTPS | Redfish API |
| CooledAI gateway | 8080 | HTTPS | Telemetry, config |
| CooledAI cloud | 443 | HTTPS | Dashboard, policy |

## Credentials

The agent holds two sets of credentials:

1. **BMC/IPMI credentials**: Username and password for the server's BMC.
   - Stored in /etc/cooledai/agent.env (mode 0600, owned by cooledai user)
   - Never logged, never included in error messages or stack traces
   - Credential scrubber active in logging configuration
   - Rotation: every 90 days (automated via gateway at scale)

2. **CooledAI API key**: Bearer token for the CooledAI platform.
   - Generated per client during onboarding
   - Stored as bcrypt hash on the server side
   - Rotation: every 90 days with 75-day warning

## Credential Rotation

### Manual (current — 2-node deployment)
```bash
# Generate new BMC password
NEW_PASS=$(openssl rand -base64 24)

# Update BMC via Redfish
curl -k -X PATCH https://<bmc>/redfish/v1/AccountService/Accounts/1 \
  -H 'Content-Type: application/json' \
  -d "{\"Password\": \"$NEW_PASS\"}"

# Update agent config
sudo sed -i "s/XCC_BMC_PASS=.*/XCC_BMC_PASS=$NEW_PASS/" /etc/cooledai/agent.env
sudo systemctl restart cooledai-agent
```

### Automated (at scale — via HashiCorp Vault)
See Phase 6.4 of the 5MW Scaling Plan.

## Agent Hardening (systemd)

The cooledai-agent systemd unit enforces:

```ini
[Service]
User=cooledai
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/cooledai /var/lib/cooledai
```

The agent runs as a dedicated `cooledai` service user with no sudo access
except for specific nvidia-smi and ipmitool commands via /etc/sudoers.d/.

## Audit Log

Every hardware command is recorded in an append-only JSONL log at
`/var/log/cooledai/audit/YYYY-MM-DD.jsonl` before execution.

Each entry contains:
- ISO timestamp
- node_id
- command_type (FAN_SPEED_SET, TEMP_READ, etc.)
- Exact command string
- Result code
- `data_plane_touched: false` (hardcoded, immutable)

The audit log is queryable via `GET /api/v1/audit/commands?date=YYYY-MM-DD`.

## Security Attestation

`GET /api/v1/security/attestation` generates a signed JSON report containing:
- Permitted hardware interfaces with exact scope
- 30-day command summary (types and counts only)
- Confirmation of zero OS-layer access
- Software version and integrity hash
- HMAC-SHA256 signature for tamper detection

This report can be handed directly to a security audit team.

## Reporting Vulnerabilities

security@cooledai.com
