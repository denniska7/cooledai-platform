# CooledAI Universal Protocol Gateway

Enterprise-grade edge agent for high-criticality data center environments.

## Architecture

```
gateway/
├── collectors/           # Protocol modules
│   ├── base_collector.py # BaseCollector + TelemetryObject
│   ├── bacnet_manager.py # BACnet (HVAC, BMS)
│   ├── snmp_manager.py   # SNMP v3 (AES/SHA)
│   └── redfish_manager.py # Redfish (servers)
├── control_gate.py       # SHADOW/PRODUCTION mode
├── telemetry_buffer.py   # SQLite store-and-forward
├── normalizer.py         # F→C, W→kW
├── heartbeat.py          # CPU/RAM every 60s
├── cloud_heartbeat.py    # Last successful cloud response (for SafetyWatchdog)
├── safety_watchdog.py    # Cloud connection monitor → REVERT_TO_DEFAULT
├── log_scrubber.py       # Scrub IPs/credentials
└── main.py               # Entry point
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROL_MODE` | `SHADOW` | `SHADOW` = intercept writes, log only. `PRODUCTION` = safety bounds then send |
| `COOLEDAI_BACKEND_URL` | `https://api.cooledai.com` | Railway backend URL |
| `HEARTBEAT_INTERVAL_SEC` | `60` | Agent health signal interval |
| `COOLEDAI_AGENT_ID` | `default` | Agent identifier |
| `COOLEDAI_CLOUD_LOSS_TIMEOUT_SEC` | `30` | SafetyWatchdog: trigger REVERT_TO_DEFAULT if no cloud response for this many seconds |

## Shadow vs Production

- **SHADOW**: All SET/WRITE commands intercepted, logged to `shadow_actions.log`, never sent to hardware.
- **PRODUCTION**: Commands executed only after Safety Bounds check (e.g., fan RPM never below 30%).

## SafetyWatchdog (Cloud Connection Monitor)

Runs in a **dedicated thread**, independent of the AI optimization loop. Monitors the heartbeat/connection to the cloud API:

- Every successful telemetry or heartbeat response to the cloud updates the "last successful cloud response" timestamp.
- If no successful response has been received for **30 seconds** (configurable via `COOLEDAI_CLOUD_LOSS_TIMEOUT_SEC`), the SafetyWatchdog triggers **REVERT_TO_DEFAULT** on all connected cooling units (100% cooling via each collector’s `write("revert_to_default", 100, "%")`).
- This ensures that loss of cloud connectivity fails safe: cooling reverts to a known default instead of leaving setpoints stale.

## Store-and-Forward

If backend connection fails, telemetry is stored in `telemetry_buffer.db`. On reconnect, data is burst in chronological order.

## Data Normalization

- Fahrenheit → Celsius
- Watts → Kilowatts
- Heartbeat: CPU/RAM health every 60 seconds

## Security

- **SNMP v3**: Enforced with AES/SHA encryption (no v1/v2c).
- **Log Scrubbing**: IP addresses and credentials removed before cloud transmission. Use `scrub_log_message()`.

## Run

```bash
pip install -r requirements-gateway.txt
export CONTROL_MODE=SHADOW
export COOLEDAI_BACKEND_URL=https://your-railway-url.up.railway.app
python -m gateway.main
```
