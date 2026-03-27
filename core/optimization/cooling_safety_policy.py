"""
Policy-layer cooling rules (under predictive T+10 / FOPDT).

Field datasets showed efficiency optimization driving fans too low under sustained
GPU load, plus silent periods where no fan response fired despite rising power.
These constants implement:
  - Minimum fan RPM while compute is active (power floor)
  - Post-thermal-spike hold (hysteresis) so fans do not immediately return to floor
  - Power-slope pre-ramp hooks for the power-cost optimizer

FOPDT calibration (``topology.calibrate_rack``) fits step responses only; it does **not**
use the RL ``compute_reward`` weights (energy vs thermal). PINN / recurrent checkpoints
are unrelated to that fit — if retraining embeds “prefer lower fan,” policy floors here
and in ``apply_guardrails`` remain the corrective layer; retrain with thermal-safety
objectives if the network feeds control directly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

_log = logging.getLogger("cooledai.cooling_safety_policy")

import numpy as np

if TYPE_CHECKING:
    from core.optimization.thermal_calibrator import CalibrationProfile


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# --- Active compute fan floor (GPU / package power) ---
ACTIVE_COMPUTE_POWER_THRESHOLD_W = _env_float("COOLEDAI_ACTIVE_COMPUTE_POWER_THRESHOLD_W", 40.0)
# Sustained signal: recent mean must also sit above this (reduces single-sample glitches)
SUSTAINED_COMPUTE_MEAN_THRESHOLD_W = _env_float("COOLEDAI_SUSTAINED_COMPUTE_MEAN_THRESHOLD_W", 35.0)
SUSTAINED_COMPUTE_TAIL = 5

# Chassis rated max RPM for policy math when telemetry only shows a stuck-low fan (agent sends this).
DEFAULT_RATED_MAX_FAN_RPM = _env_float("COOLEDAI_RATED_MAX_FAN_RPM", 7000.0)

# Absolute floor when active (clamped vs max RPM). Aligns with ~2400–2600 RPM guidance on typical servers.
ACTIVE_COMPUTE_MIN_FAN_RPM = _env_float("COOLEDAI_ACTIVE_COMPUTE_MIN_FAN_RPM", 2500.0)
ACTIVE_COMPUTE_MAX_FRAC_OF_RATED = 0.92

# --- Post-spike thermal hold (residual heat) ---
# NOTE Phase 6: Raised from 42→75°C. At 42°C (normal operating temp), spike hold
# fired constantly and blocked all optimization. 75°C is a real spike indicator.
SPIKE_HOLD_ENTER_TEMP_C = _env_float("COOLEDAI_SPIKE_HOLD_ENTER_TEMP_C", 75.0)
SPIKE_HOLD_DURATION_S = _env_float("COOLEDAI_SPIKE_HOLD_DURATION_S", 120.0)  # 2 min (reduced from 4)
SPIKE_HOLD_MIN_FAN_RPM = _env_float("COOLEDAI_SPIKE_HOLD_MIN_FAN_RPM", 2800.0)
SPIKE_HOLD_MAX_FRAC_OF_RATED = 0.95

# --- Rate-of-change safety (Phase 6 — replaces static spike hold) ---
RATE_OF_CHANGE_EMERGENCY_C_PER_S = 0.5   # >0.5°C/s → revert to BMC baseline
RATE_OF_CHANGE_CAUTION_C_PER_S = 0.2     # >0.2°C/s → freeze current RPM
RATE_OF_CHANGE_WINDOW_S = 30.0           # Rolling window for dT/dt

# --- Absolute temperature ceiling ---
THERMAL_CEILING_C = _env_float("COOLEDAI_THERMAL_CEILING_C", 85.0)

# --- Power-slope pre-ramp (proactive fan before temp crosses target) ---
# Phase 6.1: Raised thresholds — 1.5 W/s fires constantly during normal GPU
# inference workloads (Llama ~65W fluctuating), blocking ALL reduction candidates.
# New values: only trigger pre-ramp when power is ramping hard AND margin is small.
POWER_SLOPE_PRE_RAMP_MODERATE_W_S = 8.0   # W/s — was 1.5; only block reductions under steep ramp
POWER_SLOPE_PRE_RAMP_STEEP_W_S = 15.0     # W/s — was 4.0; enforce bump only under very steep ramp
POWER_SLOPE_STEEP_MIN_DELTA = 0.08         # was 0.12; less aggressive bump
# Margin threshold: only trigger pre-ramp when headroom is small
POWER_SLOPE_PRE_RAMP_MARGIN_C = 5.0       # Only pre-ramp if within 5°C of target (was 15°C)

# --- Safety net: never “silent” under rising load ---
# Phase 6.1: Raised from 0.25 W/s — normal inference fluctuation exceeds this.
SAFETY_NET_POWER_SLOPE_W_S = 5.0           # was 0.25; only net for significant power ramp
SAFETY_NET_MIN_DELTA = 0.03                # was 0.06; gentler floor


def sustained_active_compute_signal(power: np.ndarray) -> bool:
    """True when recent power history indicates sustained active compute."""
    if power.size < 2:
        return False
    n = min(SUSTAINED_COMPUTE_TAIL, int(power.size))
    tail = power[-n:]
    mean_tail = float(np.mean(tail))
    last = float(tail[-1])
    return (
        mean_tail >= SUSTAINED_COMPUTE_MEAN_THRESHOLD_W
        and last >= (ACTIVE_COMPUTE_POWER_THRESHOLD_W - 2.0)
    )


def policy_capacity_rpm(
    telemetry_max_cooling_rpm: float,
    rated_max_fan_rpm: Optional[float] = None,
) -> float:
    """Capacity basis for %‑of‑rated floors: max(telemetry, rated).

    If fans are stuck at ~1.9k RPM, ``telemetry_max`` alone makes
    ``min(2500, telem×0.92)`` *below* current speed so the active floor never fires.
    ``rated_max_fan_rpm`` (from agent ``max_fan_rpm`` or env) fixes that.
    """
    if rated_max_fan_rpm is not None and rated_max_fan_rpm > 1.0:
        rated = float(rated_max_fan_rpm)
    else:
        rated = DEFAULT_RATED_MAX_FAN_RPM
    return max(float(telemetry_max_cooling_rpm), rated)


def active_compute_fan_floor_rpm(capacity_rpm: float) -> float:
    """Minimum RPM floor while package power is in active range (before spike hold)."""
    if capacity_rpm < 1e-6:
        return ACTIVE_COMPUTE_MIN_FAN_RPM
    return min(ACTIVE_COMPUTE_MIN_FAN_RPM, capacity_rpm * ACTIVE_COMPUTE_MAX_FRAC_OF_RATED)


def spike_hold_fan_floor_rpm(capacity_rpm: float, configured_min: Optional[float] = None) -> float:
    """RPM floor during post-spike hold window."""
    base = configured_min if configured_min is not None else SPIKE_HOLD_MIN_FAN_RPM
    if capacity_rpm < 1e-6:
        return float(base)
    return min(float(base), capacity_rpm * SPIKE_HOLD_MAX_FRAC_OF_RATED)


def check_rate_of_change(
    temp_history: list,
    window_s: float = RATE_OF_CHANGE_WINDOW_S,
) -> str:
    """Check thermal rate-of-change from recent temperature readings.

    Args:
        temp_history: List of (timestamp, temp_c) tuples, newest last.
        window_s: Time window in seconds for dT/dt calculation.

    Returns:
        "emergency" if rising > 0.5°C/s (revert to BMC baseline)
        "caution" if rising > 0.2°C/s (freeze current RPM)
        "safe" if stable or cooling
    """
    import time as _time
    if len(temp_history) < 2:
        return "safe"
    now = _time.time()
    cutoff = now - window_s
    recent = [(ts, t) for ts, t in temp_history if ts >= cutoff]
    if len(recent) < 2:
        return "safe"
    oldest_ts, oldest_t = recent[0]
    newest_ts, newest_t = recent[-1]
    dt = newest_ts - oldest_ts
    if dt < 1.0:
        return "safe"
    rate = (newest_t - oldest_t) / dt
    if rate > RATE_OF_CHANGE_EMERGENCY_C_PER_S:
        _log.warning("[RATE_SAFETY] Emergency: %.2f°C/s (threshold %.2f)", rate, RATE_OF_CHANGE_EMERGENCY_C_PER_S)
        return "emergency"
    if rate > RATE_OF_CHANGE_CAUTION_C_PER_S:
        _log.info("[RATE_SAFETY] Caution: %.2f°C/s (threshold %.2f)", rate, RATE_OF_CHANGE_CAUTION_C_PER_S)
        return "caution"
    return "safe"


def merge_policy_floors(
    telemetry_max_cooling_rpm: float,
    *,
    rated_max_fan_rpm: Optional[float] = None,
    power_draw_w: float,
    sustained_active: bool,
    spike_hold_min_rpm: float = 0.0,
    calibration_profile: Optional["CalibrationProfile"] = None,
    mode: str = "shadow",
) -> float:
    """Single soft floor in RPM from active compute + optional spike hold.

    When *calibration_profile* is provided, floor values and hysteresis come
    from the profile instead of hardcoded module constants.
    """
    cap = policy_capacity_rpm(telemetry_max_cooling_rpm, rated_max_fan_rpm)
    floor = 0.0

    # Phase 6: In supervised/active modes, skip policy floor stacking.
    # Rate-of-change monitoring + absolute ceiling provides safety instead.
    if mode in ("supervised", "active"):
        _log.debug("[POLICY_FLOORS] Skipped in %s mode (rate-of-change safety active)", mode)
        return 0.0

    # Shadow mode: full policy floors (BMC controls fans anyway)
    # Resolve thresholds: profile overrides → module constants
    if calibration_profile is not None:
        active_floor_rpm = calibration_profile.active_compute_fan_floor_rpm
        spike_floor_rpm = calibration_profile.spike_hold_fan_floor_rpm
    else:
        active_floor_rpm = active_compute_fan_floor_rpm(cap)
        spike_floor_rpm = None  # fall through to legacy logic

    if sustained_active or power_draw_w >= ACTIVE_COMPUTE_POWER_THRESHOLD_W:
        if calibration_profile is not None:
            floor = max(floor, active_floor_rpm)
        else:
            floor = max(floor, active_compute_fan_floor_rpm(cap))
        _log.debug(
            "[ACTIVE_COMPUTE_FLOOR] Applied: sustained_active=%s power_draw=%.1fW "
            "floor=%.0f RPM capacity=%.0f RPM",
            sustained_active, power_draw_w, floor, cap,
        )
    if spike_hold_min_rpm > 0:
        if spike_floor_rpm is not None:
            floor = max(floor, min(spike_hold_min_rpm, spike_floor_rpm))
        else:
            floor = max(floor, min(spike_hold_min_rpm, cap * SPIKE_HOLD_MAX_FRAC_OF_RATED))
        _log.debug(
            "[SPIKE_HOLD_FLOOR] Applied: spike_hold_min_rpm=%.0f final_floor=%.0f RPM",
            spike_hold_min_rpm, floor,
        )
    return floor


def parse_policy_context(policy_context: Optional[Dict[str, Any]]) -> float:
    """Return spike-hold minimum RPM from API/brain policy context (0 if inactive)."""
    if not policy_context:
        return 0.0
    if not policy_context.get("spike_hold_active"):
        return 0.0
    v = policy_context.get("spike_hold_min_rpm")
    if v is None:
        return float(SPIKE_HOLD_MIN_FAN_RPM)
    return float(v)
