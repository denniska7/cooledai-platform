# CooledAI Privacy Policy — Data Collection Statement

## What CooledAI Collects

CooledAI collects **only** thermal and power telemetry from monitored servers. Every data field is enforced by a code-level whitelist (`api/telemetry_whitelist.py`).

### Collected Fields

| Category | Fields |
|----------|--------|
| **GPU Thermal** | `gpu_temp_c` — GPU die temperature in Celsius |
| **GPU Power** | `gpu_power_w` — GPU power draw in watts |
| **GPU Memory** | `gpu_memory_utilization_pct` — VRAM utilization percentage |
| **CPU Thermal** | `cpu_temp_c` — CPU package temperature in Celsius |
| **Fan** | `fan_rpm`, `fan_duty_pct` — chassis fan speed and duty cycle |
| **Ambient** | `ambient_temp_c`, `inlet_temp_c`, `exhaust_temp_c` |
| **Power** | `power_draw_w`, `peak_power_w` — system-level power |
| **Metadata** | `node_id`, `timestamp_utc`, `agent_version` |

### Derived Metrics (computed server-side)

- Heating rate (dT/dt)
- Cooling efficiency (delta-T vs. fan RPM)
- PUE estimates
- Calibration profiles (thermal response curves)

## What CooledAI Does NOT Collect

CooledAI **never** reads, stores, transmits, or processes:

- Process lists or running applications
- Job queue contents (Slurm, Kubernetes, PBS, etc.)
- Container names, images, or orchestration metadata
- Application logs or stdout/stderr
- Network traffic, packet captures, or connection tables
- Filesystem contents, file names, or directory listings
- User accounts, authentication tokens, or credentials
- GPU memory contents or model weights
- Any data that reveals **what** is running on the server

## Enforcement

The telemetry whitelist is enforced at two levels:

1. **Agent-side** (defense-in-depth): The CooledAI agent filters outbound telemetry before transmission, sending only whitelisted fields.

2. **API-side** (authoritative): The CooledAI API validates every incoming telemetry record against `ALLOWED_TELEMETRY_FIELDS`. Fields outside the whitelist are:
   - Stripped from the record before storage
   - Logged as `[SECURITY] telemetry_violation: field={field} rejected`
   - In strict mode (`telemetry.strict_whitelist: true` in `cooledai.yaml`), the entire record is rejected with a `TelemetryViolationError`

## Data Storage & Retention

- Telemetry data is stored per-client with strict `client_id` isolation
- No API key can access telemetry belonging to a different client
- Data retention follows client agreement terms
- All stored data contains only the whitelisted fields listed above

## Data Transmission

- All telemetry is transmitted over HTTPS (TLS 1.2+)
- API authentication via Bearer token (bcrypt-hashed API keys)
- No telemetry is shared between clients or with third parties

## Contact

For privacy questions: privacy@cooledai.com
