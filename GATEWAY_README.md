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

## Shadow vs Production

- **SHADOW**: All SET/WRITE commands intercepted, logged to `shadow_actions.log`, never sent to hardware.
- **PRODUCTION**: Commands executed only after Safety Bounds check (e.g., fan RPM never below 30%).

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
