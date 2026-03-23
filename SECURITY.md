# CooledAI Security Model

## Authentication

### API Key Authentication
- Every API endpoint (except `/health`) requires a valid `X-API-Key` header
- Keys are generated per-client using `secrets.token_urlsafe(32)` with `sk-` prefix
- Keys are stored as **bcrypt hashes** in `data/api_keys.json` — plaintext keys are never persisted
- Each key is bound to a `client_id` for tenant isolation
- Key rotation: keys expire after **90 days**; a warning header (`X-CooledAI-Key-Expiry-Warning`) is added at **75 days**
- Expired keys receive a `401 Unauthorized` response

### Portal Authentication
- The CooledAI web portal uses Clerk JWT authentication
- JWT tokens are verified against Clerk's public keys

## HTTPS

- Production deployments enforce HTTPS via `HTTPSRedirectMiddleware`
- HTTP requests receive a `301 Redirect` to the HTTPS equivalent
- `Strict-Transport-Security` header is set with `max-age=31536000`
- Controlled via `server.require_https` in `cooledai.yaml` (or `COOLEDAI_REQUIRE_HTTPS` env var)

## Rate Limiting

- **60 requests per minute** per API key (sliding window)
- Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header
- The health endpoint is exempt from rate limiting
- Configurable via `auth.rate_limit_per_minute` in `cooledai.yaml`

## Request Validation

- Global request body size limit: **1 MB**
- String field validation: max **256 characters** on API key management endpoints
- Telemetry records are validated against a field whitelist (see `PRIVACY.md`)

## BMC/XCC Security

### Credential Handling
- XCC/BMC credentials are loaded from environment variables (`XCC_BMC_HOST`, `XCC_BMC_USER`, `XCC_BMC_PASS`)
- Credentials **never** appear in log output — a credential scrubber masks them before any log line is written
- IPMI passwords are passed via environment variable (not command-line arguments) to prevent exposure in process listings

### HMAC Request Signing
- Outbound Redfish API calls include `X-CooledAI-Signature` (HMAC-SHA256) and `X-CooledAI-Timestamp` headers
- Prevents replay attacks — requests older than 60 seconds are rejected
- HMAC secret configured via `COOLEDAI_XCC_HMAC_SECRET` environment variable
- Signing is optional; when no secret is configured, requests proceed without signatures

### Network Access
The CooledAI agent requires network access to exactly two endpoints:
1. **BMC/XCC** (typically on `169.254.x.x` link-local or dedicated management VLAN) — for fan control and sensor reading
2. **CooledAI API** (HTTPS) — for telemetry upload and configuration

No other outbound network access is required or expected.

## Systemd Sandboxing

The `cooledai-agent.service` runs with hardened systemd settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| `NoNewPrivileges` | `yes` | Prevents privilege escalation |
| `ProtectSystem` | `strict` | Read-only filesystem except allowed paths |
| `ProtectHome` | `yes` | No access to user home directories |
| `PrivateTmp` | `yes` | Isolated /tmp namespace |
| `MemoryDenyWriteExecute` | `yes` | Prevents code injection |
| `RestrictSUIDSGID` | `yes` | No SUID/SGID binary execution |
| `ReadWritePaths` | `/var/lib/cooledai /var/log/cooledai` | Only paths the agent needs |
| `CapabilityBoundingSet` | `CAP_SYS_RAWIO CAP_NET_ADMIN` | Only for ipmitool BMC access |

The agent runs as a dedicated `cooledai` service user with no sudo access.

## Audit Log

### Location
`/var/log/cooledai/audit.log` (configurable via `audit.log_path` in `cooledai.yaml`)

### Format
Structured JSON lines (one entry per line), append-only:

```json
{"ts":"2026-03-23T18:00:00Z","type":"api_request","key_id":"sk-abc123...","client_id":"acme","endpoint":"/api/v1/telemetry","method":"POST","status_code":200,"ip":"10.0.1.5"}
{"ts":"2026-03-23T18:00:01Z","type":"fan_command","node_id":"node-01","target_rpm":4500,"method":"redfish","success":true,"error":null}
{"ts":"2026-03-23T18:00:02Z","type":"security_rejection","reason":"rate_limit_exceeded","key_id":"sk-xyz789...","ip":"10.0.1.5","endpoint":"/api/v1/telemetry"}
```

### Recorded Events
- Every API request (key ID, client ID, endpoint, method, status code, IP)
- Every XCC fan command (target RPM, method, success/failure)
- Every security rejection (reason, key ID, IP, endpoint)

### Rotation Policy
- **Daily rotation** recommended (configure via logrotate)
- **90-day retention** (configurable via `audit.retention_days`)
- Append-only — no existing entries are modified or deleted during normal operation

## Telemetry Data Isolation

See `PRIVACY.md` for the complete data collection policy and whitelist enforcement.

## Credential Rotation

| Credential | Rotation Period | How to Rotate |
|-----------|----------------|---------------|
| API Key | 90 days | Generate new key via admin endpoint; old key expires automatically |
| XCC BMC Password | Per organizational policy | Update `XCC_BMC_PASS` env var and restart agent |
| HMAC Secret | Per organizational policy | Update `COOLEDAI_XCC_HMAC_SECRET` env var and restart agent |

## Responsible Disclosure

Report security vulnerabilities to: security@cooledai.com
