# Accessing Recent Telemetry Logs

## 1. Cloud API (no SSH, works from anywhere)

### Telemetry logs – last hour (GPU temps, CPU temps, Fan RPM, GPU power)

```bash
./scripts/tail_telemetry.sh
# Or for 2 hours:
./scripts/tail_telemetry.sh 2
```

Or with curl:

```bash
curl -s -H "X-API-Key: <YOUR_COOLEDAI_API_KEY>" \
  "https://proactive-creativity-production.up.railway.app/api/v1/telemetry-logs?hours=1"
```

Returns all telemetry points for both nodes: `cooledai_gpu_temp_c`, `control_gpu_temp_c`, `cooledai_fan_rpm`, `control_fan_rpm`, `cooledai_cpu_temp_c`, `control_cpu_temp_c`, `cooledai_gpu_power_w`, `control_gpu_power_w`.

### ST550 telemetry agent logs (`st550_telemetry.py`)

- **`"received":2` with only `gpu` in the log** means the POST had **two GPU rows only** — no `/cpu` or `/fans`. Fix: `git pull` the latest script, **`sudo`** `start_telemetry.sh`, **`apt install ipmitool`**, and watch for `records=4 {'gpu': 2, 'cpu': 1, 'fans': 1}` (counts vary).
- **Connect / read timeouts to Railway** are often transient (cold start, Wi‑Fi). The script retries with backoff; optional env: `COOLEDAI_TELEMETRY_TIMEOUT_S=45`, `COOLEDAI_TELEMETRY_RETRIES=6`.

### Live snapshot (API key only)

```bash
curl -s -H "X-API-Key: <YOUR_COOLEDAI_API_KEY>" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status
```

Returns pilot and control node status plus latest telemetry: `temp_c`, `avg_gpu_temp_c`, `gpu_temps_c`, `fan_rpm`, `raw_fan_wattage`.

### Raw thermal history (requires portal login)

The `/api/v1/thermal-history?mode=raw&hours=24` endpoint returns time-series telemetry but requires a Clerk JWT. **Easiest access:** open the portal at https://cooledai-platform.vercel.app (or your deployed URL), sign in, and use the dashboard charts. The 1H/24H/7D views show raw telemetry.

### Optimize/control — verify policy fan floor (API key)

Isolates **cloud optimizer** vs **host fan path**. After deploying the latest API, `POST /api/v1/optimize/control` includes:

- `policy_soft_floor_rpm` — target minimum RPM under active GPU load (~2500 typical with 7000 rated max).
- `policy_floor_forced_after_layers` — `true` if slew/hysteresis was overridden to hit that floor.
- `policy_capacity_rpm` — rated capacity used for policy math.

```bash
export API_KEY="your-x-api-key"
export API_BASE="https://your-railway-or-api-host"

curl -s -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"temp_c":48,"fan_rpm":1932,"gpu_power_w":120,"max_fan_rpm":7000,"node_id":"ST550-CooledAI-Predictive"}' \
  "$API_BASE/api/v1/optimize/control" | python3 -m json.tool
```

If `policy_soft_floor_rpm` is correct but the **physical fan** barely moves, check IPMI/BIOS/PWM limits on the ST550 — the command may not reach the hardware.

---

## 2. Remote node logs (SSH required, same network)

### Pilot (CooledAI Predictive) — Tailscale `100.92.29.44`

| Log | Path | Contents |
|-----|------|----------|
| Agent | `~/cooledai_agent_pilot.log` or `/tmp/cooledai_agent.log` | Fan control, telemetry uploads, IPMI |
| Telemetry | `/var/log/cooledai_telemetry.log` or `/tmp/cooledai_telemetry.log` | st550_telemetry / start_telemetry output |
| Lenovo live | `/var/log/lenovo_live.log` or `/tmp/lenovo_live.log` | 5s samples (if lenovo_live.py running) |

### Node 101 (Control – Traditional)

| Log | Path | Contents |
|-----|------|----------|
| Agent | `~/cooledai_agent_control.log` or `/tmp/cooledai_agent.log` | Telemetry uploads |
| Telemetry | `/var/log/cooledai_telemetry.log` or `/tmp/cooledai_telemetry.log` | st550_telemetry output |
| Lenovo live | `/var/log/lenovo_live.log` or `/tmp/lenovo_live.log` | 5s samples (if lenovo_live.py running) |

### Last hour from both nodes (one command)

```bash
./scripts/fetch_last_hour_logs.sh
```

Writes agent, telemetry, and lenovo_live logs to `./logs_last_hour_YYYYMMDD_HHMM/`. Run from your Mac (Tailscale for pilot `100.92.29.44`, LAN for control). Use SSH keys or enter password when prompted.

### Quick tail commands

Pilot uses Tailscale **`100.92.29.44`** (or `192.168.12.100` from the LAN data port).

```bash
# Pilot – last 50 lines of agent log
ssh -t cooledaiadmin@100.92.29.44 'tail -50 ~/cooledai_agent_pilot.log 2>/dev/null || tail -50 /tmp/cooledai_agent.log'

# Control – last 50 lines of agent log
ssh -t cooledaiadmin@192.168.12.101 'tail -50 ~/cooledai_agent_control.log 2>/dev/null || tail -50 /tmp/cooledai_agent.log'

# Pilot – telemetry log
ssh -t cooledaiadmin@100.92.29.44 'tail -100 /var/log/cooledai_telemetry.log 2>/dev/null || tail -100 /tmp/cooledai_telemetry.log'

# Pilot – lenovo_live (JSON Lines, last 20 records)
ssh -t cooledaiadmin@100.92.29.44 'tail -20 /var/log/lenovo_live.log 2>/dev/null || tail -20 /tmp/lenovo_live.log'
```

---

## 3. Local script (optional)

Save as `scripts/tail_telemetry.sh` and run from the project root:

```bash
#!/bin/bash
API_KEY="${COOLEDAI_API_KEY:-<YOUR_COOLEDAI_API_KEY>}"
echo "=== Cloud status (live telemetry) ==="
curl -s -H "X-API-Key: $API_KEY" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status | python3 -m json.tool
```
