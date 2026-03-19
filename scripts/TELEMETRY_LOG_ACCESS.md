# Accessing Recent Telemetry Logs

## 1. Cloud API (no SSH, works from anywhere)

### Live snapshot (API key only)

```bash
curl -s -H "X-API-Key: ***REDACTED_API_KEY***" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status
```

Returns pilot and control node status plus latest telemetry: `temp_c`, `avg_gpu_temp_c`, `gpu_temps_c`, `fan_rpm`, `raw_fan_wattage`.

### Raw thermal history (requires portal login)

The `/api/v1/thermal-history?mode=raw&hours=24` endpoint returns time-series telemetry but requires a Clerk JWT. **Easiest access:** open the portal at https://cooledai-platform.vercel.app (or your deployed URL), sign in, and use the dashboard charts. The 1H/24H/7D views show raw telemetry.

---

## 2. Remote node logs (SSH required, same network)

### Node 100 (Pilot – CooledAI Predictive)

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

### Quick tail commands

```bash
# Node 100 – last 50 lines of agent log
ssh -t cooledaiadmin@192.168.12.100 'tail -50 ~/cooledai_agent_pilot.log 2>/dev/null || tail -50 /tmp/cooledai_agent.log'

# Node 101 – last 50 lines of agent log
ssh -t cooledaiadmin@192.168.12.101 'tail -50 ~/cooledai_agent_control.log 2>/dev/null || tail -50 /tmp/cooledai_agent.log'

# Node 100 – telemetry log
ssh -t cooledaiadmin@192.168.12.100 'tail -100 /var/log/cooledai_telemetry.log 2>/dev/null || tail -100 /tmp/cooledai_telemetry.log'

# Node 100 – lenovo_live (JSON Lines, last 20 records)
ssh -t cooledaiadmin@192.168.12.100 'tail -20 /var/log/lenovo_live.log 2>/dev/null || tail -20 /tmp/lenovo_live.log'
```

---

## 3. Local script (optional)

Save as `scripts/tail_telemetry.sh` and run from the project root:

```bash
#!/bin/bash
API_KEY="${COOLEDAI_API_KEY:-***REDACTED_API_KEY***}"
echo "=== Cloud status (live telemetry) ==="
curl -s -H "X-API-Key: $API_KEY" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status | python3 -m json.tool
```
