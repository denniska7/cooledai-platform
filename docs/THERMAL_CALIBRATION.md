# Thermal Auto-Calibration

CooledAI auto-discovers hardware-specific cooling thresholds by observing telemetry during a calibration window. This eliminates hardcoded RPM, temperature, and wattage constants that break when deployed on different hardware.

## How it works

### Calibration window

On startup the agent enters a **calibration window** (default 30 minutes, configurable via `COOLEDAI_CALIB_WINDOW_S`). During this time it collects every telemetry tick:

- Fan RPM readings (chassis IPMI tachometers)
- GPU power draw (W) from `nvidia-smi`
- GPU temperature (°C)
- CPU temperature (°C)

The system is **protected during calibration** using bootstrap defaults derived from the rated fan max RPM. Once the window completes, observed percentiles replace bootstrap values.

### The 9 derived thresholds

All thresholds are computed as percentages of observed anchors — no magic numbers.

| # | Threshold | Formula | Purpose |
|---|-----------|---------|---------|
| 1 | `active_compute_fan_floor_rpm` | `fan_idle + (fan_range × 0.12)` | Minimum RPM during active GPU compute |
| 2 | `active_compute_trigger_w` | `gpu_idle_w × 1.30` | Power level that activates compute floor |
| 3 | `spike_hold_fan_floor_rpm` | `fan_ceiling × 1.10` (capped at rated × 0.95) | Elevated RPM floor after thermal spike |
| 4 | `spike_trigger_temp_c` | `temp_mean + (temp_stdev × 2.5)` (clamped to [hw_limit×0.50, hw_limit×0.85]) | Temperature that triggers spike hold |
| 5 | `hysteresis_rpm` | `fan_range × 0.08` (clamped [20, 60]) | Dead zone to prevent fan chattering |
| 6 | `slew_rate_up_rpm_per_cycle` | `fan_range × 0.15` (min 30) | Max RPM increase per optimization cycle |
| 7 | `slew_rate_down_rpm_per_cycle` | `fan_range × 0.04` | Max RPM decrease per optimization cycle |
| 8 | `min_response_quantum_rpm` | `fan_range × 0.12` (min 30) | Minimum RPM change — smaller changes round up |
| 9 | `spike_hold_duration_s` | `spike_recovery_s × 1.50` (clamped [120, 600]) | How long to maintain elevated fan floor after spike |

### Observed anchors

| Anchor | Definition |
|--------|-----------|
| `fan_idle_rpm` | P10 of observed fan RPMs |
| `fan_ceiling_rpm` | P95 of observed fan RPMs |
| `fan_range_rpm` | `fan_ceiling - fan_idle` |
| `gpu_idle_w` | P10 of observed GPU power |
| `temp_mean_c` | Mean of observed GPU temps |
| `temp_stdev_c` | Standard deviation of GPU temps |
| `temp_p90_c` / `temp_p99_c` | P90 / P99 of GPU temps |
| `spike_recovery_s` | Median seconds from temp peak back to (mean + 1σ) |

### Safety clamps

Regardless of observations, these hard limits are enforced:

- `spike_trigger_temp_c` clamped to `[gpu_hw_thermal_limit × 0.50, gpu_hw_thermal_limit × 0.85]`
- `spike_hold_fan_floor` never exceeds `fan_rated_max × 0.95`
- `hysteresis_rpm` clamped to `[20, 60]`
- `slew_rate_up` minimum 30 RPM/cycle
- `min_response_quantum` minimum 30 RPM
- `spike_hold_duration_s` clamped to `[120, 600]`

## EWMA recalibration

Every 6 hours (configurable via `COOLEDAI_RECALIB_INTERVAL_S`), thresholds are recalculated from all accumulated telemetry and blended with existing values using **EWMA (α=0.25)**:

```
new_threshold = 0.25 × fresh_calculation + 0.75 × current_threshold
```

This ensures thresholds **drift smoothly** toward changing conditions (e.g. seasonal ambient changes, workload shifts) without sudden jumps that could destabilize cooling.

### Drift detection

If the 30-minute rolling mean GPU temperature shifts more than **5°C** from the calibrated `temp_mean_c`, an immediate EWMA re-observation is triggered regardless of the recalibration interval. This catches rapid environmental changes (HVAC failure, sudden load migration).

## Reading calibration log output

After initial calibration completes, the agent logs:

```
CooledAI calibration complete — thresholds set from observation:
  active_floor=1784 RPM | spike_hold=5060 RPM | trigger=47.5°C |
  hysteresis=256 RPM | slew_up=480 RPM/cycle | slew_down=128 RPM/cycle |
  min_response=384 RPM | hold_duration=180s
```

During calibration:

```
CooledAI calibrating — using bootstrap defaults for 1800s.
System is protected but not yet optimized for this hardware.
```

The `calibration_profile` dict is included in every optimize/control API response under `raw_metrics.calibration_profile` and includes `calibration_state` (`CALIBRATING` | `CALIBRATED` | `RECALIBRATING`), `calibration_progress_pct`, and all threshold values.

## Manual override env vars

For cases where auto-calibration should be bypassed (known hardware, testing, regulatory compliance):

| Variable | Effect |
|----------|--------|
| `COOLEDAI_ACTIVE_FLOOR_RPM` | Fixed active compute fan floor (RPM) |
| `COOLEDAI_SPIKE_HOLD_RPM` | Fixed spike hold fan floor (RPM) |
| `COOLEDAI_SPIKE_TRIGGER_C` | Fixed spike trigger temperature (°C) |

When any of these are set, the calibrator logs a WARNING and uses the fixed value for that parameter. Other parameters still auto-calibrate normally.

**When to use manual overrides:**
- Regulated environments requiring fixed setpoints
- Testing / validation against known baselines
- Hardware where sensor noise makes auto-calibration unreliable

## Troubleshooting

### Thresholds look wrong

1. Check `calibration_progress_pct` — if < 100%, calibration hasn't completed yet
2. Check `sample_count` — at least 10 samples are required
3. Verify sensors are reporting valid data (not stuck at 0 or N/A)
4. Check for manual override env vars that may be pinning values

### Calibration stuck

- Ensure the agent is receiving telemetry (fans, GPU power, temps)
- Check `COOLEDAI_CALIB_WINDOW_S` — default is 1800s (30 min)
- In dry-run mode, synthetic sensor data feeds calibration normally

### Manual override not taking effect

- Env vars must be set **before** the agent starts
- Check spelling: `COOLEDAI_ACTIVE_FLOOR_RPM`, `COOLEDAI_SPIKE_HOLD_RPM`, `COOLEDAI_SPIKE_TRIGGER_C`
- Look for the WARNING log line confirming override detection

### Thresholds drift too much

- Increase `COOLEDAI_RECALIB_INTERVAL_S` (default 21600 = 6h)
- EWMA α=0.25 means 75% of the old value is retained each recalibration
- Safety clamps prevent dangerous drift regardless of EWMA

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `COOLEDAI_CALIB_WINDOW_S` | 1800 | Initial calibration window (seconds) |
| `COOLEDAI_RECALIB_INTERVAL_S` | 21600 | Periodic recalibration interval (seconds) |
| `COOLEDAI_ACTIVE_FLOOR_RPM` | (auto) | Manual override: active compute floor |
| `COOLEDAI_SPIKE_HOLD_RPM` | (auto) | Manual override: spike hold floor |
| `COOLEDAI_SPIKE_TRIGGER_C` | (auto) | Manual override: spike trigger temp |
