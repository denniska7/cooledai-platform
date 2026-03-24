# Lenovo ThinkSystem ST550 — Fan Control Reference

**Last Updated**: 2026-03-24
**BMC Firmware**: 1.90
**XCC Interface**: USB-over-Ethernet at `169.254.95.118` (link-local, interface `enx9a29a6faed33`)

---

## Executive Summary

The Lenovo ST550 BMC supports fan speed control via IPMI-over-LAN (0x3a 0x07 Lenovo OEM)
**when the BIOS Operating Mode is set to CustomMode**. In other modes (Efficiency_FavorPower,
Efficiency_FavorPerformance), the commands are accepted but silently ignored.

**Telemetry proof (Session 23, 30,510 samples over 16 hours):**
- CooledAI mean fan RPM: **2,735** vs Control: **3,216** — **481 RPM lower (15% reduction)**
- CooledAI fan range: **2,037 – 4,458 RPM** (continuous 21 RPM steps = active control)
- Control fan range: **2,709 – 4,308 RPM** (BMC auto only)
- Fan power saving via cubic law: (2735/3216)^3 = **0.62x — 38% less fan energy**

---

## What Works

### BIOS Operating Mode Change (via Redfish)

This is the **only method that actually changes fan behavior**.

**Credentials**: `USERID:***REDACTED_BMC_PASS***` (Redfish Basic Auth)

**Step 1 — Read current mode:**
```bash
curl -skL -u USERID:'***REDACTED_BMC_PASS***' \
  https://169.254.95.118/redfish/v1/Systems/1/Bios/ \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(d['Attributes']['OperatingModes_ChooseOperatingMode'])"
```

**Step 2 — Set new mode (pending reboot):**
```bash
curl -sk -u USERID:'***REDACTED_BMC_PASS***' -X PATCH \
  -H 'Content-Type: application/json' \
  -d '{"Attributes":{"OperatingModes_ChooseOperatingMode":"CustomMode"}}' \
  'https://169.254.95.118/redfish/v1/Systems/1/Bios/Pending'
```
- Returns HTTP 200 on success
- **IMPORTANT**: Uses `/Bios/Pending` (Lenovo-specific), NOT `/Bios/Settings` (returns 404)
- Attribute key uses underscores: `OperatingModes_ChooseOperatingMode`

**Step 3 — Verify pending change:**
```bash
curl -skL -u USERID:'***REDACTED_BMC_PASS***' \
  https://169.254.95.118/redfish/v1/Systems/1/Bios/Pending \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(d['Attributes']['OperatingModes_ChooseOperatingMode'])"
```

**Step 4 — Reboot to apply:**
```bash
curl -sk -u USERID:'***REDACTED_BMC_PASS***' -X POST \
  -H 'Content-Type: application/json' \
  -d '{"ResetType":"GracefulRestart"}' \
  'https://169.254.95.118/redfish/v1/Systems/1/Actions/ComputerSystem.Reset'
```
- Server takes ~5-8 minutes for full BIOS POST + OS boot
- SSH will drop during reboot

### Available Operating Modes

| Mode | Fan Behavior | Use Case |
|------|-------------|----------|
| `MinimalPower` | Lowest fan speeds | Quiet environments |
| `Efficiency_FavorPower` | Low fans, favor power savings | Default for efficiency |
| `Efficiency_FavorPerformance` | Moderate fans, favor performance | Balanced |
| `CustomMode` | **Lower baseline** (~24% reduction vs FavorPerformance) | **CooledAI recommended** |
| `MaximumPerformance` | Highest fan speeds | Maximum cooling |

### Measured Fan Speeds by Mode

| Mode | Fan 1 (RPM) | Fan 2 (RPM) | Fan 3 (RPM) | Average |
|------|------------|------------|------------|---------|
| Efficiency_FavorPerformance | 3,525 | 2,925 | 3,000 | 3,150 |
| CustomMode (idle) | 2,500 | 2,350 | 2,500 | 2,450 |
| CustomMode (under load) | 3,850 | 3,150 | 3,225 | 3,408 |

**Net effect of CustomMode**: ~24% RPM reduction at idle, ~8% at load. Fan power follows cubic law: (2450/3150)^3 = 0.47x — 53% less fan energy at idle.

---

## What Works — IPMI Fan Speed Override (CustomMode ONLY)

### IPMI OEM Commands (0x3a 0x07 — Lenovo OEM)

**These commands WORK in CustomMode but are IGNORED in other BIOS modes.**

```bash
# Step 1: Enable manual mode
ipmitool -I lanplus -H 169.254.95.118 -U USERID -P '***REDACTED_BMC_PASS***' \
  raw 0x3a 0x07 0x01 0x00

# Step 2: Set fan duty (hex percentage: 0x1e=30%, 0x32=50%, 0x64=100%)
ipmitool -I lanplus -H 169.254.95.118 -U USERID -P '***REDACTED_BMC_PASS***' \
  raw 0x3a 0x07 0x01 0x1e    # Set 30% (~2100 RPM)
```

**CRITICAL**: Must use `-I lanplus` (IPMI over LAN). Local `-I open` does NOT have
`/dev/ipmi0` on the ST550 — there is no in-band IPMI driver. All commands go through
the XCC BMC over the USB-ethernet interface at 169.254.95.118.

**IMPORTANT**: When testing manually via SSH, the CooledAI agent may be sending its
own fan commands every 3 seconds, immediately overwriting your test command. Stop the
agent first (`sudo systemctl stop cooledai-agent`) before manual testing.

### Telemetry-Confirmed Fan Ranges (Session 23)

| Duty Command | Approximate RPM | Notes |
|-------------|----------------|-------|
| 29% (0x1d) | ~2,037 RPM | Lowest observed (idle stepping) |
| 30% (0x1e) | ~2,100 RPM | Bootstrap floor |
| 38% (0x26) | ~2,667 RPM | Typical compute floor |
| 48% (0x30) | ~3,360 RPM | Sustained high load |
| 63% (0x3f) | ~4,400 RPM | Spike response |

---

## What Does NOT Work

### IPMI OEM Commands (0x32 — Lenovo Legacy)

```bash
ipmitool raw 0x32 0x9b 0x01       # "Invalid command"
ipmitool raw 0x32 0x69 0x00 0x19  # "Invalid command"
```

### IPMI Dell-Style Commands (0x30 0x30)

```bash
ipmitool raw 0x30 0x30 0x01 0x00  # "Invalid command"
ipmitool raw 0x30 0x30 0x02 0xff 0x1e  # "Invalid command"
```

### IPMI SuperMicro-Style Commands (0x30 0x70 0x66)

```bash
ipmitool raw 0x30 0x70 0x66 0x01 0x00 0x64  # "Invalid command"
```

### Redfish PATCH on Thermal

```bash
# All return 405 Method Not Allowed
curl -sk -X PATCH -u USERID:'***REDACTED_BMC_PASS***' \
  -H 'Content-Type: application/json' \
  -d '{"Fans":[{"MemberId":"0","Reading":20}]}' \
  'https://169.254.95.118/redfish/v1/Chassis/1/Thermal/'

curl -sk -X PATCH -u USERID:'***REDACTED_BMC_PASS***' \
  -H 'Content-Type: application/json' \
  -d '{"Oem":{"Lenovo":{"FanMinimumSpeed":20}}}' \
  'https://169.254.95.118/redfish/v1/Chassis/1/Thermal/'
```

### Redfish OEM Fan Endpoints

```bash
# All return 404 Not Found
/redfish/v1/Managers/1/Oem/Lenovo/FanControl
/redfish/v1/Managers/1/Oem/Lenovo/ThermalManagement
/redfish/v1/Managers/1/Actions/Oem/LenovoManager.SetFanControlMode
```

### XCC Web API (/api/login)

The XCC web interface at `https://169.254.95.118` has a `/api/login` endpoint that returns JWT tokens for a different auth system. **All credential combinations fail** (USERID/PASSW0RD, USERID/***REDACTED_BMC_PASS***, cooledaiadmin/***REDACTED_SSH_PASS***). Redfish Basic Auth works fine, but the XCC web API uses a separate JWT mechanism that appears locked/broken on this firmware.

### hwmon / PWM / sysfs

No fan control sysfs paths exist on the ST550:
- `/sys/class/hwmon/*/pwm*` — no entries
- `/sys/class/hwmon/*/fan*_target` — no entries
- The BMC manages fans entirely through its own firmware, not exposed to the OS

### nvidia-smi GPU Power Limiting

GPUs are 2x Quadro P2000 with fixed TDP: min=max=75W. `nvidia-smi -pl` cannot reduce below 75W.

---

## What Works for Monitoring (Read-Only)

### Redfish Thermal (GET)

```bash
curl -skL -u USERID:'***REDACTED_BMC_PASS***' \
  'https://169.254.95.118/redfish/v1/Chassis/1/Thermal/' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    [print(f'{f[\"Name\"]}: {f[\"Reading\"]}% ({f.get(\"Status\",{}).get(\"State\",\"?\")})')
     for f in d.get('Fans',[])]"
```
Returns fan percentages (0-100), temperatures (Ambient, Exhaust, CPU1, CPU2, DTS).

### IPMI SDR (local)

```bash
sudo ipmitool sdr type Fan
# Fan 1 Tach | C0h | ok | 29.1 | 3150 RPM
# Fan 2 Tach | C1h | ok | 29.2 | 3075 RPM
# Fan 3 Tach | C2h | ok | 29.3 | 3225 RPM
```

### IPMI over LAN (remote)

```bash
ipmitool -I lanplus -H 169.254.95.118 -U USERID -P '***REDACTED_BMC_PASS***' sdr type Fan
```

---

## Network & Credentials

| Interface | IP | Protocol | Credentials |
|-----------|-----|----------|-------------|
| XCC USB-Ethernet | 169.254.95.118 | Redfish HTTPS | USERID / ***REDACTED_BMC_PASS*** |
| XCC USB-Ethernet | 169.254.95.118 | IPMI lanplus (623) | USERID / ***REDACTED_BMC_PASS*** |
| OS SSH | 100.92.29.44 (Tailscale) | SSH | cooledaiadmin / ***REDACTED_SSH_PASS*** |
| OS SSH | 192.168.12.100 (LAN) | SSH | cooledaiadmin / ***REDACTED_SSH_PASS*** |

**Note**: The host OS IP on the USB-ethernet interface is `169.254.95.120`, BMC is at `169.254.95.118`.

---

## CooledAI Optimization Strategy for ST550

Since direct fan speed override is not possible, CooledAI optimizes through:

1. **BIOS Operating Mode** — Set to `CustomMode` for lowest safe BMC baseline (~24% fan energy reduction)
2. **GPU Clock Management** — Idle clock reduction, memory-bound clock optimization (via nvidia-smi -ac/-rac)
3. **Predictive Thermal Management** — Pre-cooling and workload-aware scheduling to keep BMC's thermal algorithm from overreacting
4. **Telemetry & Monitoring** — Rich Redfish + IPMI telemetry for dashboard, alerting, and trend analysis

---

## Troubleshooting

### Fans stuck at high RPM after BIOS change
The BIOS change requires a **reboot** to take effect. Check `/redfish/v1/Systems/1/Bios/Pending` to verify the pending change, then reboot via Redfish.

### BIOS mode reverted after power cycle
Some firmware versions revert BIOS settings on AC power loss. Re-apply the PATCH and reboot.

### Redfish returns empty responses
Add `-L` to curl to follow 301 redirects. Lenovo XCC redirects `/Thermal` to `/Thermal/` (trailing slash).

### "Invalid data field in request" on 0x3a 0x07 0x00 0x00
The auto-restore command format is wrong for this firmware version. The BMC doesn't support programmatic fan mode switching.

---

## Firmware-Specific Notes

- **XCC Firmware 1.90**: No writable fan endpoints via Redfish or IPMI OEM
- **Redfish v1.0.2**: Read-only Thermal, BIOS PATCH via `/Bios/Pending` (not `/Bios/Settings`)
- **BIOS PATCH path**: `/redfish/v1/Systems/1/Bios/Pending` — the standard `/Bios/Settings` returns 404
- **Attribute format**: Underscores (`OperatingModes_ChooseOperatingMode`), not dots
