# Lenovo ThinkSystem ST550 — Fan Optimization Runbook

**Last Updated**: 2026-03-24
**BMC Firmware**: 1.90
**XCC Interface**: USB-over-Ethernet at `169.254.95.118`

---

## Quick Start: How to Enable Fan Optimization

Follow these steps in order. Skip steps already completed.

### Prerequisites

| Item | Pilot Node | Control Node |
|------|-----------|-------------|
| Tailscale IP | 100.92.29.44 | 100.123.202.94 |
| LAN IP | 192.168.12.100 | 192.168.12.101 |
| SSH | cooledaiadmin / <SSH_PASSWORD> | cooledaiadmin / <SSH_PASSWORD> |
| XCC BMC IP | 169.254.95.118 (USB-ethernet) | 169.254.95.118 (USB-ethernet) |
| XCC Credentials | USERID / <BMC_PASSWORD> | USERID / <BMC_PASSWORD> |
| Host USB-ethernet IP | 169.254.95.120 | 169.254.95.120 |

---

### Step 1: Set BIOS to CustomMode (ONE-TIME, requires reboot)

Fan optimization ONLY works when BIOS is in **CustomMode**. In any other mode
(Efficiency_FavorPower, Efficiency_FavorPerformance, etc.), IPMI fan commands
are silently ignored by the BMC.

**Check current mode:**
```bash
ssh cooledaiadmin@100.92.29.44
curl -skL -u USERID:'<BMC_PASSWORD>' \
  https://169.254.95.118/redfish/v1/Systems/1/Bios/ \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(d['Attributes']['OperatingModes_ChooseOperatingMode'])"
```

If it already says `CustomMode`, skip to Step 2.

**Change to CustomMode (requires reboot):**
```bash
# Stage the BIOS change (Lenovo uses /Bios/Pending, NOT /Bios/Settings)
curl -sk -u USERID:'<BMC_PASSWORD>' -X PATCH \
  -H 'Content-Type: application/json' \
  -d '{"Attributes":{"OperatingModes_ChooseOperatingMode":"CustomMode"}}' \
  'https://169.254.95.118/redfish/v1/Systems/1/Bios/Pending'

# Verify the pending change
curl -skL -u USERID:'<BMC_PASSWORD>' \
  https://169.254.95.118/redfish/v1/Systems/1/Bios/Pending \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print('Pending:', d['Attributes']['OperatingModes_ChooseOperatingMode'])"

# Reboot to apply (server takes 5-8 minutes to come back)
curl -sk -u USERID:'<BMC_PASSWORD>' -X POST \
  -H 'Content-Type: application/json' \
  -d '{"ResetType":"GracefulRestart"}' \
  'https://169.254.95.118/redfish/v1/Systems/1/Actions/ComputerSystem.Reset'
```

**After reboot, verify:**
```bash
curl -skL -u USERID:'<BMC_PASSWORD>' \
  https://169.254.95.118/redfish/v1/Systems/1/Bios/ \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(d['Attributes']['OperatingModes_ChooseOperatingMode'])"
# Should print: CustomMode
```

---

### Step 2: Configure Agent Environment

Ensure `/etc/cooledai/agent.env` has these settings:
```bash
ssh cooledaiadmin@100.92.29.44
cat /etc/cooledai/agent.env
```

Required variables:
```env
COOLEDAI_API_KEY=<YOUR_COOLEDAI_API_KEY>
COOLEDAI_NODE_ID=ST550-CooledAI-Predictive
COOLEDAI_API_URL=https://proactive-creativity-production.up.railway.app
COOLEDAI_GPU_POWER_MGMT=true
COOLEDAI_GPU_PERF_MODE=false
XCC_BMC_HOST=169.254.95.118
XCC_BMC_USER=USERID
XCC_BMC_PASS=<BMC_PASSWORD>
```

If any are missing, add them:
```bash
echo '<SSH_PASSWORD>' | sudo -S bash -c 'cat >> /etc/cooledai/agent.env << EOF
COOLEDAI_GPU_POWER_MGMT=true
COOLEDAI_GPU_PERF_MODE=false
XCC_BMC_HOST=169.254.95.118
XCC_BMC_USER=USERID
XCC_BMC_PASS=<BMC_PASSWORD>
EOF'
```

---

### Step 3: Deploy Latest Agent Code

```bash
# On the pilot node, pull latest code
cd ~/coolingai_simulator && git pull origin main

# Copy to systemd service location
echo '<SSH_PASSWORD>' | sudo -S cp scripts/cooledai_agent.py /opt/cooledai/cooledai_agent.py
echo '<SSH_PASSWORD>' | sudo -S cp core/optimization/thermal_calibrator.py /opt/cooledai/core/optimization/thermal_calibrator.py
echo '<SSH_PASSWORD>' | sudo -S cp core/optimization/gpu_power_governor.py /opt/cooledai/core/optimization/gpu_power_governor.py
echo '<SSH_PASSWORD>' | sudo -S cp core/hardware/xcc_fan_controller.py /opt/cooledai/core/hardware/xcc_fan_controller.py
```

---

### Step 4: Delete Stale Calibration Profile (if exists)

A stale profile with bad fan_idle_rpm will prevent optimization. Delete it
before starting the agent fresh:
```bash
echo '<SSH_PASSWORD>' | sudo -S rm -f /var/lib/cooledai/calibration_profile.json
```

---

### Step 5: Start/Restart the Agent via systemd

**The agent MUST run as root** via systemd for local IPMI (`/dev/ipmi0`) access.
Do NOT run manually as cooledaiadmin — that path lacks IPMI and falls back to
XCC-over-LAN which is less effective.

```bash
echo '<SSH_PASSWORD>' | sudo -S systemctl restart cooledai-agent
echo '<SSH_PASSWORD>' | sudo -S systemctl status cooledai-agent
```

Expected status: `Active: active (running)`

---

### Step 6: Verify Fan Optimization is Working

**Within first 10 seconds — check startup logs:**
```bash
echo '<SSH_PASSWORD>' | sudo -S journalctl -u cooledai-agent --since '30 sec ago' --no-pager | head -20
```

Look for these 4 lines:
1. `Discovered: CPU(sysfs:3zones) GPU(nvidia) IPMI(lenovo) Redfish(...) XCC_FAN(active)` — IPMI(lenovo) confirms local IPMI available
2. `[XCC_FAN] optimization_owns_control=True` — XCC taken over at startup
3. `[GPU_GOV] Governor initialized` — GPU power governor active
4. `CooledAI calibrating — using bootstrap defaults` — calibration starting (30 min window)

**After 30 seconds — check FAN_DIAG:**
```bash
echo '<SSH_PASSWORD>' | sudo -S journalctl -u cooledai-agent --since '10 sec ago' --no-pager | grep FAN_DIAG
```

Look for: `method=ipmi` (local IPMI) or `method=xcc_ipmi_lan` (LAN fallback)

**Check live telemetry on API:**
```bash
curl -s -H "X-API-Key: <YOUR_COOLEDAI_API_KEY>" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status \
  | python3 -m json.tool
```

CooledAI fan_rpm should be LOWER than Control fan_rpm.

---

### Step 7: Also Start Telemetry Script

The telemetry script (separate from the agent) posts GPU/CPU/fan data to the
Railway API portal. Start it alongside the agent:
```bash
cd ~/coolingai_simulator
nohup python3 -u scripts/st550_telemetry.py \
  --node-id ST550-CooledAI-Predictive > ~/cooledai_telemetry.log 2>&1 &
```

For the control node:
```bash
ssh cooledaiadmin@100.123.202.94
cd ~/coolingai_simulator
COOLEDAI_NODE_ID=ST550-Control-Traditional nohup python3 -u scripts/st550_telemetry.py \
  --node-id ST550-Control-Traditional > ~/cooledai_telemetry.log 2>&1 &
```

---

### Step 8: Start Identical Workloads on Both Nodes

```bash
# On BOTH nodes (pilot and control), run the same benchmark:
nohup python3 -u ~/thermal_workload_benchmark.py \
  --model llama3.2:3b --seed 42 --ports 11434,11435 > ~/workload.log 2>&1 &
```

Both Ollama instances must be running first:
```bash
# GPU 0 (default)
sudo systemctl start ollama  # or: ollama serve &

# GPU 1 (second instance on port 11435)
sudo CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 \
  OLLAMA_MODELS=/opt/ollama_gpu1/models ollama serve &
```

---

## How Fan Optimization Works (Technical)

### Fan Control Command

The working IPMI OEM command for Lenovo ThinkSystem ST550:
```
ipmitool raw 0x3a 0x07 0x01 {hex_pct}
```
- `0x3a` = Lenovo OEM NetFn
- `0x07` = Fan control command
- `0x01` = Manual mode enable
- `{hex_pct}` = Fan duty in hex (0x1e=30%, 0x32=50%, 0x64=100%)

**This command ONLY works when BIOS is in CustomMode.**

### Fan Control Priority in Agent (`set_fan_duty()`)

1. **Local IPMI** (`/dev/ipmi0`) — primary, requires root, uses `0x3a 0x07`
2. **XCC IPMI-over-LAN** — fallback when no `/dev/ipmi0` (e.g., agent run as non-root)
3. **PWM sysfs** — not available on ST550
4. **GPU fan** — not applicable for chassis fans

### Commands That Do NOT Work

| Command | Result | Notes |
|---------|--------|-------|
| `raw 0x32 0x9b 0x01` | Invalid command | Legacy Lenovo, not ThinkSystem |
| `raw 0x32 0x69 0x00 {pct}` | Invalid command | Legacy Lenovo |
| `raw 0x30 0x30 0x01 0x00` | Invalid command | Dell-style |
| `raw 0x30 0x70 0x66 ...` | Invalid command | SuperMicro-style |
| `PATCH /Chassis/1/Thermal/` | 405 Method Not Allowed | Firmware blocks Redfish writes |
| `POST SetFanControlMode` | 404 / 501 | OEM action not implemented |

### Agent Startup Sequence

1. Agent starts via systemd as root
2. XCC controller probes Redfish endpoints (PATCH 405, OEM 501)
3. XCC detects IPMI-over-LAN is reachable (chassis status OK)
4. XCC sends `set_manual_fan_percent(50)` → `optimization_owns_control=True`
5. Main loop begins: reads sensors → calls Railway API → gets target_duty → applies via local IPMI
6. ThermalCalibrator runs 30-min observation window with idle stepping
7. After calibration, profile is saved to `/var/lib/cooledai/calibration_profile.json`
8. Next restart loads saved profile and skips observation window

### Calibration Idle Stepping (Circular Reference Fix)

During the 30-min observation window, when GPU power drops below the active
compute trigger (~15W), the agent forces fans to the bootstrap floor (2,100 RPM)
instead of the optimizer's computed target. This ensures the calibrator measures
true hardware idle fan behavior, not its own inflated commands.

Log line: `[THERMAL_CAL] calibration_idle_step: gpu_power=5.4W fan_forced_to=30% (2100 RPM)`

### Profile Rejection on Load

On startup, if a saved profile has `fan_idle_rpm / fan_ceiling_rpm > 0.90`
(idle and ceiling within 10%), the profile is rejected as a circular reference
artifact and a fresh observation window runs instead.

Log line: `[THERMAL_CAL] Stale profile rejected: fan_idle_rpm=X within 10% of fan_ceiling_rpm=Y`

---

## Proven Results

### Session 23 (30,510 samples, 16 hours)

| Metric | CooledAI | Control | Delta |
|--------|---------|---------|-------|
| Mean Fan RPM | 2,735 | 3,216 | -481 (15% lower) |
| Min Fan RPM | 2,037 | 2,709 | -672 |
| Max Fan RPM | 4,458 | 4,308 | +150 |
| Fan Energy | 0.62x | 1.0x | **38% savings** |

### Current Live Telemetry

| Metric | CooledAI | Control | Delta |
|--------|---------|---------|-------|
| Fan RPM | 2,688 | 3,325 | -637 (19% lower) |
| GPU Temp | 58.5°C | 60.0°C | -1.5°C |
| Fan Energy | 0.53x | 1.0x | **47% savings** |

Fan power follows the cubic law: Power ~ RPM^3. A 19% RPM reduction = 47% energy savings.

---

## Troubleshooting

### "Fans stuck at 3850/3075/3225 RPM"

1. **Check BIOS mode** — must be `CustomMode`. If not, follow Step 1 above.
2. **Check agent is running as root via systemd** — `sudo systemctl status cooledai-agent`. PID should be owned by root.
3. **Check FAN_DIAG method** — must show `method=ipmi` (local) not `method=none`. If `none`, local IPMI isn't working.
4. **Check calibration state** — during the 30-min observation window, the brain uses cautionary cooling (+5-15% duty). Wait for calibration to complete.
5. **Stop the agent before manual testing** — `sudo systemctl stop cooledai-agent`. The agent sends commands every 3 seconds and will overwrite your test.

### "method=xcc_ipmi_lan instead of method=ipmi"

The agent doesn't detect local IPMI. Possible causes:
- Agent not running as root (systemd service runs as root, manual python3 does not)
- `/dev/ipmi0` doesn't exist — check: `ls -la /dev/ipmi0`
- IPMI kernel module not loaded — check: `lsmod | grep ipmi`

### "method=none on every FAN_DIAG"

No fan control path is working. Check:
1. `caps.ipmi` detection: `journalctl -u cooledai-agent | grep "Discovered:"`
2. IPMI variant: should show `IPMI(lenovo)` not `IPMI(generic)`
3. XCC availability: should show `XCC_FAN(active)`

### "API returns target_duty=47-48% instead of lower"

The Railway API brain is in cautionary cooling mode (low confidence). This happens after
every Railway redeploy (wipes in-memory thermal history). Wait 30+ minutes for the brain
to accumulate enough history and confidence will rise above the threshold, allowing the
optimizer's lower recommendations to flow through.

### "Stale profile rejected" on every restart

The previous calibration ran with the circular reference bug (optimizer commanding high
RPM during idle → calibrator measuring commanded RPM as idle baseline). Delete the bad
profile and let a clean observation run with the idle-stepping fix:
```bash
sudo rm /var/lib/cooledai/calibration_profile.json
sudo systemctl restart cooledai-agent
```

### "BIOS mode reverted after power cycle"

Some firmware versions lose BIOS settings on AC power loss. Re-run the Redfish PATCH
from Step 1 and reboot.

### "Redfish returns empty responses"

Add `-L` to curl to follow 301 redirects. Lenovo XCC redirects `/Thermal` to `/Thermal/`
(trailing slash required).

---

## BIOS Operating Modes Reference

| Mode | Fan Behavior | CooledAI Compatible |
|------|-------------|-------------------|
| `MinimalPower` | Lowest BMC curve | Unknown — not tested |
| `Efficiency_FavorPower` | Low BMC curve | NO — ignores IPMI commands |
| `Efficiency_FavorPerformance` | Moderate BMC curve | NO — ignores IPMI commands |
| `CustomMode` | Lower baseline, accepts IPMI | **YES — required for CooledAI** |
| `MaximumPerformance` | Highest BMC curve | Unknown — not tested |

### Measured Fan Speeds by Mode (at idle)

| Mode | Fan 1 | Fan 2 | Fan 3 | Average |
|------|-------|-------|-------|---------|
| Efficiency_FavorPerformance | 3,525 | 2,925 | 3,000 | 3,150 |
| CustomMode | 2,500 | 2,350 | 2,500 | 2,450 |

---

## Control Node Setup (No Optimization)

The control node runs telemetry only — no fan control, no optimization.
Its agent.env should have `COOLEDAI_CONTROL_ENABLED=false`:

```env
COOLEDAI_API_KEY=<YOUR_COOLEDAI_API_KEY>
COOLEDAI_NODE_ID=ST550-Control-Traditional
COOLEDAI_API_URL=https://proactive-creativity-production.up.railway.app
COOLEDAI_CONTROL_ENABLED=false
```

The control node's BIOS should stay in its default operating mode
(Efficiency_FavorPerformance) to represent the "traditional" cooling baseline.

---

## Firmware & Hardware Notes

- **BMC Firmware**: XCC 1.90
- **Redfish Version**: v1.0.2
- **BIOS PATCH Path**: `/redfish/v1/Systems/1/Bios/Pending` (NOT `/Bios/Settings` which returns 404)
- **BIOS Attribute Format**: Underscores (`OperatingModes_ChooseOperatingMode`), not dots
- **GPUs**: 2x Quadro P2000, 75W fixed TDP (min=max=75W, nvidia-smi -pl cannot reduce)
- **Fans**: 3 chassis fans, tach sensors at IPMI SDR addresses C0h/C1h/C2h
- **USB-Ethernet**: Host at 169.254.95.120, BMC at 169.254.95.118 (interface `enx9a29a6faed33`)
- **Local IPMI**: `/dev/ipmi0` available when running as root
- **XCC Web API**: `/api/login` endpoint exists but JWT auth is broken — use Redfish Basic Auth instead
