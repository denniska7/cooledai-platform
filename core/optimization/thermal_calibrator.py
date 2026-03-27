"""
CooledAI Thermal Calibrator — Auto-Discovers Hardware-Specific Thresholds

Observes telemetry during a configurable window (default 30 min) and derives
all cooling thresholds from the actual hardware distribution.  Eliminates
hardcoded RPM/temp/watt constants that break universality.

Continuous recalibration uses EWMA (alpha=0.25) so thresholds drift smoothly
toward changing conditions without sudden jumps.

A drift detector triggers immediate re-observation when the 30-min rolling
mean GPU temperature shifts >5°C from calibrated ``temp_mean_c``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger("cooledai.thermal_calibrator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_opt_float(name: str) -> Optional[float]:
    """Return float from env or None if not set."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CalibrationProfile — output of calibration consumed by all downstream
# ---------------------------------------------------------------------------

CALIBRATION_STATE_CALIBRATING = "CALIBRATING"
CALIBRATION_STATE_CALIBRATED = "CALIBRATED"
CALIBRATION_STATE_RECALIBRATING = "RECALIBRATING"


@dataclass
class CalibrationProfile:
    """All cooling thresholds derived from hardware observation."""

    # --- state ---
    calibration_state: str = CALIBRATION_STATE_CALIBRATING
    calibration_progress_pct: float = 0.0
    sample_count: int = 0

    # --- observed anchors ---
    fan_idle_rpm: float = 0.0
    fan_ceiling_rpm: float = 0.0
    fan_range_rpm: float = 0.0
    gpu_idle_w: float = 0.0
    temp_mean_c: float = 0.0
    temp_stdev_c: float = 0.0
    temp_p90_c: float = 0.0
    temp_p99_c: float = 0.0
    spike_recovery_s: float = 60.0

    # --- CPU thermal anchors ---
    cpu_temp_mean_c: float = 0.0
    cpu_temp_stdev_c: float = 0.0
    cpu_safe_ceiling_c: float = 55.0  # bootstrap default — safe above normal ST550 range

    # --- hardware constants (queried once) ---
    fan_rated_max_rpm: float = 7000.0
    gpu_hw_thermal_limit_c: float = 83.0

    # --- derived thresholds (the 9 key values) ---
    active_compute_fan_floor_rpm: float = 0.0
    active_compute_trigger_w: float = 15.0
    spike_hold_fan_floor_rpm: float = 0.0
    spike_trigger_temp_c: float = 42.0
    hysteresis_rpm: float = 30.0
    slew_rate_up_rpm_per_cycle: float = 40.0
    slew_rate_down_rpm_per_cycle: float = 12.0
    min_response_quantum_rpm: float = 40.0
    spike_hold_duration_s: float = 240.0
    sustained_compute_window_s: float = 12.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_state": self.calibration_state,
            "calibration_progress_pct": round(self.calibration_progress_pct, 1),
            "sample_count": self.sample_count,
            # observed anchors
            "fan_idle_rpm": round(self.fan_idle_rpm, 1),
            "fan_ceiling_rpm": round(self.fan_ceiling_rpm, 1),
            "fan_range_rpm": round(self.fan_range_rpm, 1),
            "gpu_idle_w": round(self.gpu_idle_w, 1),
            "temp_mean_c": round(self.temp_mean_c, 2),
            "temp_stdev_c": round(self.temp_stdev_c, 2),
            "temp_p90_c": round(self.temp_p90_c, 2),
            "temp_p99_c": round(self.temp_p99_c, 2),
            "spike_recovery_s": round(self.spike_recovery_s, 1),
            # CPU thermal anchors
            "cpu_temp_mean_c": round(self.cpu_temp_mean_c, 2),
            "cpu_temp_stdev_c": round(self.cpu_temp_stdev_c, 2),
            "cpu_safe_ceiling_c": round(self.cpu_safe_ceiling_c, 2),
            # hardware
            "fan_rated_max_rpm": round(self.fan_rated_max_rpm, 1),
            "gpu_hw_thermal_limit_c": round(self.gpu_hw_thermal_limit_c, 1),
            # derived thresholds
            "active_compute_fan_floor_rpm": round(self.active_compute_fan_floor_rpm, 1),
            "active_compute_trigger_w": round(self.active_compute_trigger_w, 1),
            "spike_hold_fan_floor_rpm": round(self.spike_hold_fan_floor_rpm, 1),
            "spike_trigger_temp_c": round(self.spike_trigger_temp_c, 2),
            "hysteresis_rpm": round(self.hysteresis_rpm, 1),
            "slew_rate_up_rpm_per_cycle": round(self.slew_rate_up_rpm_per_cycle, 1),
            "slew_rate_down_rpm_per_cycle": round(self.slew_rate_down_rpm_per_cycle, 1),
            "min_response_quantum_rpm": round(self.min_response_quantum_rpm, 1),
            "spike_hold_duration_s": round(self.spike_hold_duration_s, 1),
            "sustained_compute_window_s": round(self.sustained_compute_window_s, 1),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationProfile":
        """Reconstruct a CalibrationProfile from a dict (inverse of to_dict)."""
        p = cls()
        for f in fields(cls):
            if f.name in d:
                try:
                    setattr(p, f.name, f.type(d[f.name]) if f.type in ("float", "int") else d[f.name])
                except (ValueError, TypeError):
                    pass  # keep default
        # Explicit float coercion for numeric fields
        for f in fields(cls):
            if f.type == "float" and f.name in d:
                try:
                    setattr(p, f.name, float(d[f.name]))
                except (ValueError, TypeError):
                    pass
            elif f.type == "int" and f.name in d:
                try:
                    setattr(p, f.name, int(d[f.name]))
                except (ValueError, TypeError):
                    pass
            elif f.type == "str" and f.name in d:
                setattr(p, f.name, str(d[f.name]))
        return p


# ---------------------------------------------------------------------------
# ThermalCalibrator — runs on agent startup, produces CalibrationProfile
# ---------------------------------------------------------------------------

class ThermalCalibrator:
    """Observes telemetry and derives hardware-specific cooling thresholds.

    Usage::

        cal = ThermalCalibrator()
        # each tick:
        cal.update(fan_rpms=[3100, 3200], gpu_power_w=[62.0], gpu_temp_c=[38.5], cpu_temp_c=[44.0])
        profile = cal.profile  # always safe to read — bootstrap defaults until calibrated
    """

    def __init__(
        self,
        fan_rated_max_rpm: float = 7000.0,
        gpu_hw_thermal_limit_c: float = 83.0,
        calibration_window_s: Optional[float] = None,
        recalib_interval_s: Optional[float] = None,
    ):
        # Hardware constants (overridable by caller or env)
        self._fan_rated_max = fan_rated_max_rpm
        self._gpu_hw_limit = gpu_hw_thermal_limit_c

        self._window_s = calibration_window_s or _env_float("COOLEDAI_CALIB_WINDOW_S", 1800.0)
        self._recalib_interval_s = recalib_interval_s or _env_float("COOLEDAI_RECALIB_INTERVAL_S", 21600.0)

        # Manual override env vars
        self._manual_overrides: Dict[str, float] = {}
        self._detect_manual_overrides()

        # Observation buffers
        self._fan_rpms: List[float] = []
        self._gpu_power_w: List[float] = []
        self._gpu_temp_c: List[float] = []
        self._cpu_temp_c: List[float] = []
        self._timestamps: List[float] = []

        # Split idle/active sample buffers (FIX 1)
        # Each entry is (fan_rpm, gpu_power_w) for that tick
        self._idle_fan_rpms: List[float] = []
        self._active_fan_rpms: List[float] = []
        self._MIN_SPLIT_SAMPLES = 20

        # Spike recovery tracking
        self._temp_peaks: List[float] = []  # timestamps of local temp peaks
        self._recovery_durations: List[float] = []

        # Rolling window for drift detection (30-min)
        self._drift_window_temps: List[float] = []
        self._drift_window_times: List[float] = []
        self._DRIFT_WINDOW_S = 1800.0
        self._DRIFT_THRESHOLD_C = 5.0

        # State
        self._start_time: Optional[float] = None  # Set on first update()
        self._last_recalib_time = 0.0
        self._calibrated_once = False

        # Floor sanity check — EWMA blend to prevent >20% upward jumps
        self._floor_sanity_target: Optional[float] = None
        self._floor_sanity_current: Optional[float] = None

        # Warmup guard — first 10 min, floor can only go DOWN from bootstrap
        self._WARMUP_DURATION_S = 600.0  # 10 minutes
        self._bootstrap_floor: Optional[float] = None  # set after bootstrap profile built

        # Track when recalibration happened (for agent to detect & save)
        self._recalibrated_this_tick = False

        # Build bootstrap profile
        self._profile = self._build_bootstrap_profile()
        self._bootstrap_floor = self._profile.active_compute_fan_floor_rpm

    def _detect_manual_overrides(self) -> None:
        """Check for manual override env vars."""
        mapping = {
            "active_compute_fan_floor_rpm": "COOLEDAI_ACTIVE_FLOOR_RPM",
            "spike_hold_fan_floor_rpm": "COOLEDAI_SPIKE_HOLD_RPM",
            "spike_trigger_temp_c": "COOLEDAI_SPIKE_TRIGGER_C",
        }
        overridden: List[str] = []
        for param, env_var in mapping.items():
            val = _env_opt_float(env_var)
            if val is not None:
                self._manual_overrides[param] = val
                overridden.append(env_var)
        if overridden:
            _log.warning(
                "Manual override active — auto-calibration disabled for [%s]",
                ", ".join(overridden),
            )

    def _build_bootstrap_profile(self) -> CalibrationProfile:
        """Build profile with bootstrap defaults (used while calibrating)."""
        rated = self._fan_rated_max
        p = CalibrationProfile(
            calibration_state=CALIBRATION_STATE_CALIBRATING,
            calibration_progress_pct=0.0,
            sample_count=0,
            fan_rated_max_rpm=rated,
            gpu_hw_thermal_limit_c=self._gpu_hw_limit,
            # Bootstrap defaults
            active_compute_fan_floor_rpm=rated * 0.30,
            active_compute_trigger_w=15.0,
            spike_hold_fan_floor_rpm=rated * 0.36,
            spike_trigger_temp_c=42.0,
            hysteresis_rpm=30.0,
            slew_rate_up_rpm_per_cycle=40.0,
            slew_rate_down_rpm_per_cycle=12.0,
            min_response_quantum_rpm=40.0,
            spike_hold_duration_s=240.0,
            sustained_compute_window_s=12.0,
        )
        # Apply any manual overrides
        for param, val in self._manual_overrides.items():
            setattr(p, param, val)
        return p

    @property
    def profile(self) -> CalibrationProfile:
        """Current calibration profile (always safe to read)."""
        return self._profile

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated_once

    def update(
        self,
        fan_rpms: Optional[List[float]] = None,
        gpu_power_w: Optional[List[float]] = None,
        gpu_temp_c: Optional[List[float]] = None,
        cpu_temp_c: Optional[List[float]] = None,
        now: Optional[float] = None,
    ) -> None:
        """Feed one telemetry tick. Call every cycle."""
        self._recalibrated_this_tick = False
        t = now if now is not None else time.monotonic()
        if self._start_time is None:
            self._start_time = t

        if fan_rpms:
            for v in fan_rpms:
                self._fan_rpms.append(float(v))
        if gpu_power_w:
            for v in gpu_power_w:
                self._gpu_power_w.append(float(v))
        if gpu_temp_c:
            for v in gpu_temp_c:
                self._gpu_temp_c.append(float(v))
                self._drift_window_temps.append(float(v))
                self._drift_window_times.append(t)
        if cpu_temp_c:
            for v in cpu_temp_c:
                self._cpu_temp_c.append(float(v))

        # Classify samples into idle/active based on GPU power vs trigger
        trigger_w = self._profile.active_compute_trigger_w
        if fan_rpms and gpu_power_w:
            avg_fan = float(np.mean([float(v) for v in fan_rpms]))
            avg_power = float(np.mean([float(v) for v in gpu_power_w]))
            if avg_power < trigger_w:
                self._idle_fan_rpms.append(avg_fan)
            else:
                self._active_fan_rpms.append(avg_fan)

        self._timestamps.append(t)

        n = len(self._timestamps)
        self._profile.sample_count = n

        elapsed = t - self._start_time
        progress = min(100.0, (elapsed / self._window_s) * 100.0) if self._window_s > 0 else 100.0
        self._profile.calibration_progress_pct = progress

        # Floor sanity EWMA blend (runs every tick when active)
        if self._floor_sanity_target is not None and self._floor_sanity_current is not None:
            alpha = 0.05
            self._floor_sanity_current = (
                alpha * self._floor_sanity_target
                + (1.0 - alpha) * self._floor_sanity_current
            )
            self._profile.active_compute_fan_floor_rpm = self._floor_sanity_current
            # Converged within 2%? Snap and clear.
            if abs(self._floor_sanity_current - self._floor_sanity_target) < self._floor_sanity_target * 0.02:
                self._profile.active_compute_fan_floor_rpm = self._floor_sanity_target
                self._floor_sanity_target = None
                self._floor_sanity_current = None

        # Check if initial calibration window is complete
        if not self._calibrated_once and elapsed >= self._window_s and n >= 10:
            self._calibrate_from_scratch()
            self._calibrated_once = True
            self._last_recalib_time = t
            self._recalibrated_this_tick = True
            return

        # Periodic recalibration
        if self._calibrated_once and (t - self._last_recalib_time) >= self._recalib_interval_s and n >= 10:
            self._recalibrate_ewma()
            self._last_recalib_time = t
            self._recalibrated_this_tick = True
            return

        # Drift detection
        if self._calibrated_once:
            self._check_drift(t)

    def _check_drift(self, now: float) -> None:
        """Trigger re-observation if 30-min rolling mean shifts >5°C."""
        if not self._drift_window_temps:
            return
        # Prune old entries
        cutoff = now - self._DRIFT_WINDOW_S
        while self._drift_window_times and self._drift_window_times[0] < cutoff:
            self._drift_window_times.pop(0)
            self._drift_window_temps.pop(0)
        if len(self._drift_window_temps) < 5:
            return
        rolling_mean = float(np.mean(self._drift_window_temps))
        if abs(rolling_mean - self._profile.temp_mean_c) > self._DRIFT_THRESHOLD_C:
            _log.info(
                "Drift detected: rolling mean %.1f°C vs calibrated %.1f°C (>%.0f°C shift) — "
                "triggering re-observation.",
                rolling_mean, self._profile.temp_mean_c, self._DRIFT_THRESHOLD_C,
            )
            self._recalibrate_ewma()
            self._last_recalib_time = now

    def _compute_thresholds(
        self,
        fan_arr: np.ndarray,
        power_arr: np.ndarray,
        temp_arr: np.ndarray,
        cpu_arr: Optional[np.ndarray] = None,
    ) -> CalibrationProfile:
        """Compute CalibrationProfile from observation arrays."""
        rated = self._fan_rated_max
        hw_limit = self._gpu_hw_limit

        # --- Split-buffer fan anchor computation ---
        n_idle = len(self._idle_fan_rpms)
        n_active = len(self._active_fan_rpms)

        if n_idle >= self._MIN_SPLIT_SAMPLES:
            idle_arr = np.array(self._idle_fan_rpms, dtype=float)
            fan_idle = float(np.percentile(idle_arr, 10))
        else:
            fan_idle = float(np.percentile(fan_arr, 10)) if len(fan_arr) > 0 else 0.0
            if n_idle < self._MIN_SPLIT_SAMPLES and len(fan_arr) > 0:
                _log.warning(
                    "[THERMAL_CAL] WARNING: insufficient idle_samples=%d — using combined buffer",
                    n_idle,
                )

        if n_active >= self._MIN_SPLIT_SAMPLES:
            active_arr = np.array(self._active_fan_rpms, dtype=float)
            fan_ceiling = float(np.percentile(active_arr, 95))
        else:
            fan_ceiling = float(np.percentile(fan_arr, 95)) if len(fan_arr) > 0 else rated * 0.5
            if n_active < self._MIN_SPLIT_SAMPLES and len(fan_arr) > 0:
                _log.warning(
                    "[THERMAL_CAL] WARNING: insufficient active_samples=%d — using combined buffer",
                    n_active,
                )
        fan_range = fan_ceiling - fan_idle
        # Ensure positive range
        if fan_range < 50.0:
            fan_range = max(50.0, rated * 0.3)

        gpu_idle = float(np.percentile(power_arr, 10)) if len(power_arr) > 0 else 10.0
        temp_mean = float(np.mean(temp_arr)) if len(temp_arr) > 0 else 40.0
        temp_std = float(np.std(temp_arr)) if len(temp_arr) > 0 else 3.0
        if temp_std < 0.5:
            temp_std = 0.5  # Prevent degenerate thresholds
        temp_p90 = float(np.percentile(temp_arr, 90)) if len(temp_arr) > 0 else temp_mean + 5
        temp_p99 = float(np.percentile(temp_arr, 99)) if len(temp_arr) > 0 else temp_mean + 10

        # CPU thermal stats
        if cpu_arr is not None and len(cpu_arr) > 0:
            cpu_mean = float(np.mean(cpu_arr))
            cpu_std = float(np.std(cpu_arr))
            if cpu_std < 0.5:
                cpu_std = 0.5
            cpu_ceiling = cpu_mean + 2.0 * cpu_std
        else:
            cpu_mean = 0.0
            cpu_std = 0.0
            cpu_ceiling = 55.0  # bootstrap default

        # Spike recovery: median seconds from temp peak back to (mean + 1σ)
        spike_recovery = self._estimate_spike_recovery(temp_arr, temp_mean, temp_std)

        # --- Derived thresholds ---
        active_floor = fan_idle + fan_range * 0.12
        active_trigger_w = gpu_idle * 1.30
        spike_hold_floor = fan_ceiling * 1.10
        # Spike trigger: derive from HW limit with safety margin, not p99.
        # Old formula (temp_mean + 2.5σ) placed trigger too close to normal ops.
        spike_trigger = min(
            hw_limit - 8.0,                    # 8°C safety margin from HW limit
            max(temp_p99 + 5.0, 75.0),         # At least p99+5°C, floor 75°C
        )
        hysteresis = fan_range * 0.08
        slew_up = fan_range * 0.15
        slew_down = fan_range * 0.04
        min_quantum = fan_range * 0.12
        hold_duration = spike_recovery * 1.50
        sustained_window = 12.0

        # --- Safety clamps ---
        spike_trigger = max(hw_limit - 8.0, min(hw_limit * 0.92, spike_trigger))
        spike_hold_floor = min(spike_hold_floor, rated * 0.95)
        hysteresis = max(20.0, min(60.0, hysteresis))
        slew_up = max(30.0, slew_up)
        min_quantum = max(30.0, min_quantum)
        hold_duration = max(120.0, min(600.0, hold_duration))

        p = CalibrationProfile(
            calibration_state=CALIBRATION_STATE_CALIBRATED,
            calibration_progress_pct=100.0,
            sample_count=len(self._timestamps),
            # anchors
            fan_idle_rpm=fan_idle,
            fan_ceiling_rpm=fan_ceiling,
            fan_range_rpm=fan_range,
            gpu_idle_w=gpu_idle,
            temp_mean_c=temp_mean,
            temp_stdev_c=temp_std,
            temp_p90_c=temp_p90,
            temp_p99_c=temp_p99,
            spike_recovery_s=spike_recovery,
            # CPU thermal
            cpu_temp_mean_c=cpu_mean,
            cpu_temp_stdev_c=cpu_std,
            cpu_safe_ceiling_c=cpu_ceiling,
            # hardware
            fan_rated_max_rpm=rated,
            gpu_hw_thermal_limit_c=hw_limit,
            # derived
            active_compute_fan_floor_rpm=active_floor,
            active_compute_trigger_w=active_trigger_w,
            spike_hold_fan_floor_rpm=spike_hold_floor,
            spike_trigger_temp_c=spike_trigger,
            hysteresis_rpm=hysteresis,
            slew_rate_up_rpm_per_cycle=slew_up,
            slew_rate_down_rpm_per_cycle=slew_down,
            min_response_quantum_rpm=min_quantum,
            spike_hold_duration_s=hold_duration,
            sustained_compute_window_s=sustained_window,
        )

        # Apply manual overrides
        for param, val in self._manual_overrides.items():
            setattr(p, param, val)

        return p

    def _estimate_spike_recovery(
        self, temp_arr: np.ndarray, temp_mean: float, temp_std: float,
    ) -> float:
        """Estimate median seconds from temp peak back to (mean + 1σ)."""
        if len(temp_arr) < 20:
            return 60.0  # default
        threshold = temp_mean + temp_std
        # Find peaks (local max above threshold + 1σ)
        recoveries: List[float] = []
        in_spike = False
        spike_start_idx = 0
        for i in range(1, len(temp_arr)):
            if not in_spike and temp_arr[i] > temp_mean + 2 * temp_std:
                in_spike = True
                spike_start_idx = i
            elif in_spike and temp_arr[i] <= threshold:
                # Recovered — use index distance as proxy for seconds (1 sample ~ 1 tick)
                duration = float(i - spike_start_idx)
                if 0 < duration < 600:
                    recoveries.append(duration)
                in_spike = False
        if recoveries:
            return float(np.median(recoveries))
        return 60.0

    def _apply_floor_guards(self, new_profile: CalibrationProfile) -> None:
        """Apply warmup guard and floor sanity check to a newly computed profile."""
        # Use the last timestamp from the observation buffer (supports injected time in tests)
        t = self._timestamps[-1] if self._timestamps else time.monotonic()
        new_floor = new_profile.active_compute_fan_floor_rpm
        prev_floor = self._profile.active_compute_fan_floor_rpm

        # Warmup guard: first 10 min, floor can only go DOWN from bootstrap
        if (
            self._start_time is not None
            and self._bootstrap_floor is not None
            and (t - self._start_time) < self._WARMUP_DURATION_S
            and new_floor > self._bootstrap_floor
        ):
            new_profile.active_compute_fan_floor_rpm = self._bootstrap_floor
            _log.info(
                "[THERMAL_CAL] warmup_guard_active: floor_clamped_at_bootstrap "
                "(new=%.0f > bootstrap=%.0f, elapsed=%.0fs < %.0fs)",
                new_floor, self._bootstrap_floor,
                t - self._start_time, self._WARMUP_DURATION_S,
            )
            return

        # Floor sanity check: >20% upward jump → EWMA blend over ~10 min
        if prev_floor > 0 and new_floor > prev_floor * 1.20:
            self._floor_sanity_target = new_floor
            self._floor_sanity_current = prev_floor
            new_profile.active_compute_fan_floor_rpm = prev_floor  # hold for now
            _log.info(
                "[THERMAL_CAL] floor_sanity_clamp: previous=%.0f new=%.0f "
                "blending over 10min",
                prev_floor, new_floor,
            )

    def _calibrate_from_scratch(self) -> None:
        """First-time calibration from full observation window."""
        fan_arr = np.array(self._fan_rpms, dtype=float) if self._fan_rpms else np.array([0.0])
        power_arr = np.array(self._gpu_power_w, dtype=float) if self._gpu_power_w else np.array([10.0])
        temp_arr = np.array(self._gpu_temp_c, dtype=float) if self._gpu_temp_c else np.array([40.0])
        cpu_arr = np.array(self._cpu_temp_c, dtype=float) if self._cpu_temp_c else None

        new_profile = self._compute_thresholds(fan_arr, power_arr, temp_arr, cpu_arr=cpu_arr)
        new_profile.calibration_state = CALIBRATION_STATE_CALIBRATED
        self._apply_floor_guards(new_profile)
        self._profile = new_profile

        n_idle = len(self._idle_fan_rpms)
        n_active = len(self._active_fan_rpms)
        _log.info(
            "CooledAI calibration complete — thresholds set from observation: "
            "active_floor=%.0f RPM | spike_hold=%.0f RPM | trigger=%.1f°C | "
            "hysteresis=%.0f RPM | slew_up=%.0f RPM/cycle | slew_down=%.0f RPM/cycle | "
            "min_response=%.0f RPM | hold_duration=%.0fs | "
            "cpu_mean=%.1f°C | cpu_stdev=%.1f°C | cpu_ceiling=%.1f°C | "
            "idle_samples=%d | active_samples=%d",
            self._profile.active_compute_fan_floor_rpm,
            self._profile.spike_hold_fan_floor_rpm,
            self._profile.spike_trigger_temp_c,
            self._profile.hysteresis_rpm,
            self._profile.slew_rate_up_rpm_per_cycle,
            self._profile.slew_rate_down_rpm_per_cycle,
            self._profile.min_response_quantum_rpm,
            self._profile.spike_hold_duration_s,
            self._profile.cpu_temp_mean_c,
            self._profile.cpu_temp_stdev_c,
            self._profile.cpu_safe_ceiling_c,
            n_idle,
            n_active,
        )

    def _recalibrate_ewma(self) -> None:
        """Recalibrate with EWMA blending (alpha=0.25) — drift, never jump."""
        fan_arr = np.array(self._fan_rpms, dtype=float) if self._fan_rpms else np.array([0.0])
        power_arr = np.array(self._gpu_power_w, dtype=float) if self._gpu_power_w else np.array([10.0])
        temp_arr = np.array(self._gpu_temp_c, dtype=float) if self._gpu_temp_c else np.array([40.0])
        cpu_arr = np.array(self._cpu_temp_c, dtype=float) if self._cpu_temp_c else None

        new_profile = self._compute_thresholds(fan_arr, power_arr, temp_arr, cpu_arr=cpu_arr)
        old = self._profile
        alpha = 0.25

        # EWMA blend all numeric threshold fields
        blend_fields = [
            "fan_idle_rpm", "fan_ceiling_rpm", "fan_range_rpm", "gpu_idle_w",
            "temp_mean_c", "temp_stdev_c", "temp_p90_c", "temp_p99_c",
            "spike_recovery_s",
            "cpu_temp_mean_c", "cpu_temp_stdev_c", "cpu_safe_ceiling_c",
            "active_compute_fan_floor_rpm", "active_compute_trigger_w",
            "spike_hold_fan_floor_rpm", "spike_trigger_temp_c",
            "hysteresis_rpm", "slew_rate_up_rpm_per_cycle",
            "slew_rate_down_rpm_per_cycle", "min_response_quantum_rpm",
            "spike_hold_duration_s",
        ]
        for f in blend_fields:
            if f in self._manual_overrides:
                continue  # Manual overrides don't drift
            old_val = getattr(old, f, 0.0)
            new_val = getattr(new_profile, f, 0.0)
            blended = alpha * new_val + (1.0 - alpha) * old_val
            setattr(old, f, blended)

        # Re-apply safety clamps after EWMA blend
        hw_limit = self._gpu_hw_limit
        rated = self._fan_rated_max
        old.spike_trigger_temp_c = max(hw_limit - 8.0, min(hw_limit * 0.92, old.spike_trigger_temp_c))
        old.spike_hold_fan_floor_rpm = min(old.spike_hold_fan_floor_rpm, rated * 0.95)
        old.hysteresis_rpm = max(20.0, min(60.0, old.hysteresis_rpm))
        old.slew_rate_up_rpm_per_cycle = max(30.0, old.slew_rate_up_rpm_per_cycle)
        old.min_response_quantum_rpm = max(30.0, old.min_response_quantum_rpm)
        old.spike_hold_duration_s = max(120.0, min(600.0, old.spike_hold_duration_s))

        # Apply floor guards (warmup + sanity)
        self._apply_floor_guards(old)

        old.calibration_state = CALIBRATION_STATE_CALIBRATED
        old.sample_count = len(self._timestamps)

        _log.info(
            "CooledAI recalibration (EWMA α=0.25): active_floor=%.0f RPM | "
            "spike_hold=%.0f RPM | trigger=%.1f°C | hold_duration=%.0fs | "
            "idle_samples=%d | active_samples=%d",
            old.active_compute_fan_floor_rpm,
            old.spike_hold_fan_floor_rpm,
            old.spike_trigger_temp_c,
            old.spike_hold_duration_s,
            len(self._idle_fan_rpms),
            len(self._active_fan_rpms),
        )

    # --- Persistence (FIX 3) ---

    _SCHEMA_VERSION = 1
    _MAX_PROFILE_AGE_S = 86400.0  # 24 hours

    def save_profile(self, path: str) -> bool:
        """Serialize current CalibrationProfile to JSON (atomic write)."""
        try:
            data = {
                "schema_version": self._SCHEMA_VERSION,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "profile": self._profile.to_dict(),
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
            _log.info("[THERMAL_CAL] Saved calibration profile to %s", path)
            return True
        except Exception as exc:
            _log.warning("[THERMAL_CAL] Failed to save profile to %s: %s", path, exc)
            return False

    def load_profile(self, path: str) -> bool:
        """Load a saved CalibrationProfile from JSON.

        Returns True if successfully loaded; False if file missing, stale, or invalid.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            _log.debug("[THERMAL_CAL] No saved profile at %s", path)
            return False
        except Exception as exc:
            _log.warning("[THERMAL_CAL] Failed to read profile from %s: %s", path, exc)
            return False

        # Validate schema version
        sv = data.get("schema_version")
        if sv != self._SCHEMA_VERSION:
            _log.warning(
                "[THERMAL_CAL] Profile schema_version=%s (expected %d) — discarding",
                sv, self._SCHEMA_VERSION,
            )
            return False

        # Validate age
        saved_at_str = data.get("saved_at")
        if not saved_at_str:
            _log.warning("[THERMAL_CAL] Profile missing saved_at — discarding")
            return False
        try:
            saved_at = datetime.fromisoformat(saved_at_str)
            age_s = (datetime.now(timezone.utc) - saved_at).total_seconds()
        except Exception:
            _log.warning("[THERMAL_CAL] Profile has invalid saved_at — discarding")
            return False
        if age_s > self._MAX_PROFILE_AGE_S:
            _log.warning(
                "[THERMAL_CAL] Profile is %.1f hours old (max 24h) — discarding",
                age_s / 3600.0,
            )
            return False

        # Reconstruct profile
        profile_dict = data.get("profile")
        if not isinstance(profile_dict, dict):
            _log.warning("[THERMAL_CAL] Profile data missing or invalid — discarding")
            return False

        try:
            loaded = CalibrationProfile.from_dict(profile_dict)
        except Exception as exc:
            _log.warning("[THERMAL_CAL] Failed to reconstruct profile: %s", exc)
            return False

        # Sanity check: fan_idle_rpm must NOT be within 10% of fan_ceiling_rpm.
        # This indicates the observation window suffered the circular reference
        # bug (optimizer commanded speed recorded as idle).  A valid system has
        # meaningful separation between idle and ceiling RPM.
        if loaded.fan_ceiling_rpm > 0:
            ratio = loaded.fan_idle_rpm / loaded.fan_ceiling_rpm
            if ratio > 0.90:
                _log.warning(
                    "[THERMAL_CAL] Stale profile rejected: fan_idle_rpm=%.0f is "
                    "within 10%% of fan_ceiling_rpm=%.0f (ratio=%.3f) — "
                    "circular reference detected, re-observing",
                    loaded.fan_idle_rpm, loaded.fan_ceiling_rpm, ratio,
                )
                return False

        self._profile = loaded
        self._profile.calibration_state = CALIBRATION_STATE_CALIBRATED
        self._calibrated_once = True
        _log.info(
            "[THERMAL_CAL] Loaded saved profile from %s saved_at=%s — "
            "skipping observation window",
            path, saved_at_str,
        )
        return True

    def force_recalibrate(self) -> None:
        """Force an immediate recalibration from current buffers."""
        if len(self._timestamps) >= 10:
            self._recalibrate_ewma()
            self._last_recalib_time = time.monotonic()
