# Client telemetry survey

## Purpose

Before integration, each customer completes **`CLIENT_TELEMETRY_SURVEY.yaml`** (copy → rename e.g. `acme_dc3_row4.yaml`). This tells engineering:

- Which **signals** exist vs are missing  
- **How** actuation works (IPMI, BACnet, SNMP, nothing)  
- **Cadence** and trust in sensors  

We map “available / no / unknown” to collector adapters and define a **minimum viable** path (often: GPU temp + power + one fan metric + one control knob).

## Spreadsheet option

If procurement prefers Excel/Sheets: duplicate the YAML sections as columns (`telemetry.gpu.available`, …). Export CSV only if keys stay stable; **YAML in git** is easier to diff.

## After submit

1. **Gap analysis** — which protocols to enable on the gateway (`bacnet_manager`, `snmp_manager`, `redfish_manager`, Modbus, agent-only).  
2. **Normalization** — set `max_fan_rpm_rated`, critical temp, and whether tach is trusted.  
3. **Cold-start FOPDT** — run workload + steps, export samples, run `scripts/cold_start_fopdt_calibration.py`.  
4. **Failure posture** — API/agent now return `failure_posture` when telemetry is stale, tach may not track command, or load rises at high duty.

## Related

- `docs/PINN_ROADMAP.md` — research vs production stack  
- `docs/FLEET_AND_ECONOMIZER.md` — multi-site / DR (later phase)  
