"""
Dynamic NVIDIA GPU power cap (nvidia-smi -pl) to reduce hard thermal throttling.

When GPU temperature is comfortable, allow default TDP. As temperature approaches
a soft band, reduce power limit smoothly toward a floor so the driver throttles
gracefully instead of hitting sudden clock drops (poor perf/W).

Phase 2 adds:
  - CalibrationProfile-driven thresholds (hardware-agnostic bands)
  - Auto-calibration observation window (300s before adjusting)
  - Workload parity guard (performance_mode blocks throttle when thermally healthy)
  - [GPU_GOV] structured logging

Temperature bands are configurable via environment variables OR CalibrationProfile.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger("cooledai.gpu_gov")

# ---------------------------------------------------------------------------
# GPU power envelope (from nvidia-smi)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuPowerEnvelope:
    """Per-GPU limits reported by the driver (W)."""

    index: int
    min_w: float
    default_w: float
    max_w: float


def _parse_float_field(s: str) -> Optional[float]:
    s = s.strip().upper()
    if not s or "N/A" in s or "NOT" in s:
        return None
    m = re.search(r"([\d.]+)", s)
    if not m:
        return None
    return float(m.group(1))


def query_nvidia_power_envelopes(timeout_s: float = 8.0) -> List[GpuPowerEnvelope]:
    """
    Query min / default / max power limits per GPU via nvidia-smi.
    Returns empty list if nvidia-smi missing or query fails.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,power.min_limit,power.default_limit,power.max_limit",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    envelopes: List[GpuPowerEnvelope] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(float(parts[0]))
        except ValueError:
            continue
        mn = _parse_float_field(parts[1])
        default = _parse_float_field(parts[2])
        mx = _parse_float_field(parts[3])
        if default is None and mx is not None:
            default = mx
        if mx is None and default is not None:
            mx = default
        if default is None or mx is None:
            continue
        if mn is None:
            mn = max(15.0, default * 0.45)
        mn = max(10.0, min(mn, default))
        mx = max(default, mx)
        envelopes.append(GpuPowerEnvelope(index=idx, min_w=mn, default_w=default, max_w=mx))
    return envelopes


# ---------------------------------------------------------------------------
# Temperature → power mapping (standalone functions, kept for compatibility)
# ---------------------------------------------------------------------------


def compute_target_power_w(
    gpu_temp_c: float,
    env: GpuPowerEnvelope,
    *,
    temp_full_power_c: float,
    temp_soft_start_c: float,
    temp_hard_c: float,
    min_fraction_of_default: float = 0.55,
) -> float:
    """
    Map GPU temperature → target power limit (W).

    T <= temp_full_power_c     → default (full TDP envelope)
    T in (full, soft]          → linear blend default → mid
    T in (soft, hard)         → linear blend mid → floor
    T >= temp_hard_c           → floor (min_w clamped up to min_fraction * default)
    """
    floor_w = max(env.min_w, env.default_w * min_fraction_of_default)
    mid_w = max(floor_w, (env.default_w + floor_w) / 2.0)

    t = float(gpu_temp_c)
    if t <= temp_full_power_c:
        return env.default_w
    if t <= temp_soft_start_c:
        span = max(1e-6, temp_soft_start_c - temp_full_power_c)
        alpha = (t - temp_full_power_c) / span
        return env.default_w + alpha * (mid_w - env.default_w)
    if t < temp_hard_c:
        span = max(1e-6, temp_hard_c - temp_soft_start_c)
        alpha = (t - temp_soft_start_c) / span
        return mid_w + alpha * (floor_w - mid_w)
    return floor_w


def compute_all_targets(
    gpu_temps_c: Sequence[float],
    envelopes: Sequence[GpuPowerEnvelope],
    *,
    temp_full_power_c: float,
    temp_soft_start_c: float,
    temp_hard_c: float,
    min_fraction_of_default: float = 0.55,
) -> List[Tuple[int, float]]:
    """Pair (gpu_index, target_w) for each GPU with known temp + envelope."""
    out: List[Tuple[int, float]] = []
    env_by_idx = {e.index: e for e in envelopes}
    for i, temp in enumerate(gpu_temps_c):
        env = env_by_idx.get(i)
        if env is None:
            continue
        w = compute_target_power_w(
            temp,
            env,
            temp_full_power_c=temp_full_power_c,
            temp_soft_start_c=temp_soft_start_c,
            temp_hard_c=temp_hard_c,
            min_fraction_of_default=min_fraction_of_default,
        )
        w = max(env.min_w, min(env.max_w, round(w)))
        out.append((i, w))
    return out


# ---------------------------------------------------------------------------
# nvidia-smi control
# ---------------------------------------------------------------------------


def set_gpu_power_limit_w(gpu_index: int, watts: int, *, dry_run: bool = False) -> bool:
    """Apply nvidia-smi -pl. Often requires root."""
    if dry_run:
        return True
    try:
        subprocess.check_call(
            ["sudo", "nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))],
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def read_current_power_limits(timeout_s: float = 8.0) -> Dict[int, float]:
    """index -> current enforced limit (W) if available."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,power.limit",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    cur: Dict[int, float] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(float(parts[0]))
            v = _parse_float_field(parts[1])
            if v is not None:
                cur[idx] = v
        except ValueError:
            continue
    return cur


def reset_all_gpus_to_default(envelopes: Sequence[GpuPowerEnvelope], *, dry_run: bool = False) -> None:
    """Restore each GPU to driver default limit (best-effort on shutdown)."""
    for e in envelopes:
        set_gpu_power_limit_w(e.index, int(round(e.default_w)), dry_run=dry_run)


# ---------------------------------------------------------------------------
# GPU clock scaling (PATH B for fixed-TDP GPUs like T4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuClockCaps:
    """Per-GPU max clock speeds (MHz)."""
    index: int
    max_graphics_mhz: int
    max_memory_mhz: int


def query_gpu_clock_caps(timeout_s: float = 8.0) -> List[GpuClockCaps]:
    """Query max graphics and memory clocks per GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.max.graphics,clocks.max.memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    caps: List[GpuClockCaps] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(float(parts[0]))
            gfx = int(float(parts[1]))
            mem = int(float(parts[2]))
            caps.append(GpuClockCaps(index=idx, max_graphics_mhz=gfx, max_memory_mhz=mem))
        except (ValueError, TypeError):
            continue
    return caps


def query_supported_graphics_clocks(gpu_index: int = 0, timeout_s: float = 8.0) -> List[int]:
    """Query supported graphics clock frequencies (MHz) for a GPU, sorted descending."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i", str(gpu_index),
                "--query-supported-clocks=gr",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    clocks: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            clocks.append(int(float(line)))
        except ValueError:
            continue
    return sorted(clocks, reverse=True)


def snap_to_supported_clock(target_mhz: int, supported: List[int]) -> int:
    """Find the closest supported clock <= target_mhz. Falls back to lowest supported."""
    if not supported:
        return target_mhz
    # supported is sorted descending
    for clk in supported:
        if clk <= target_mhz:
            return clk
    return supported[-1]  # lowest available


def set_gpu_clocks(gpu_index: int, mem_mhz: int, gfx_mhz: int, *, dry_run: bool = False) -> bool:
    """Set application clocks via nvidia-smi -ac <mem>,<gfx>. Requires sudo."""
    if dry_run:
        return True
    try:
        subprocess.check_call(
            ["sudo", "nvidia-smi", "-i", str(gpu_index), "-ac", f"{mem_mhz},{gfx_mhz}"],
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def reset_gpu_clocks(gpu_index: int, *, dry_run: bool = False) -> bool:
    """Reset application clocks to default via nvidia-smi -rac. Requires sudo."""
    if dry_run:
        return True
    try:
        subprocess.check_call(
            ["sudo", "nvidia-smi", "-i", str(gpu_index), "-rac"],
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_fixed_tdp(envelopes: Sequence[GpuPowerEnvelope]) -> bool:
    """True if all GPUs have min_w == max_w (no TDP headroom)."""
    return all(abs(e.min_w - e.max_w) < 1.0 for e in envelopes)


def _query_min_memory_clock(gpu_index: int = 0, timeout_s: float = 8.0) -> Optional[int]:
    """Query the lowest supported memory clock for a GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi", "-i", str(gpu_index),
                "--query-supported-clocks=mem",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    clocks: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            clocks.append(int(float(line)))
        except ValueError:
            continue
    return min(clocks) if clocks else None


# ---------------------------------------------------------------------------
# GPU utilization queries
# ---------------------------------------------------------------------------

def query_gpu_utilization(timeout_s: float = 5.0) -> List[Tuple[int, int, int]]:
    """Query per-GPU SM utilization and memory utilization.

    Returns list of (gpu_index, sm_util_pct, mem_util_pct).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    results: List[Tuple[int, int, int]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(float(parts[0]))
            sm = int(float(parts[1]))
            mem = int(float(parts[2]))
            results.append((idx, sm, mem))
        except (ValueError, TypeError):
            continue
    return results


def query_gpu_current_clocks(timeout_s: float = 5.0) -> Dict[int, int]:
    """Query current graphics clock per GPU. Returns {gpu_idx: gfx_mhz}."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    clocks: Dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            clocks[int(float(parts[0]))] = int(float(parts[1]))
        except (ValueError, TypeError):
            continue
    return clocks


def load_governor_config_from_env() -> Tuple[float, float, float, float, float]:
    """
    Returns:
        temp_full_power_c, temp_soft_start_c, temp_hard_c,
        min_change_w (hysteresis), min_interval_s
    """
    full = float(os.environ.get("COOLEDAI_GPU_PL_TEMP_FULL_C", "62"))
    soft = float(os.environ.get("COOLEDAI_GPU_PL_TEMP_SOFT_C", "72"))
    hard = float(os.environ.get("COOLEDAI_GPU_PL_TEMP_HARD_C", "82"))
    min_change = float(os.environ.get("COOLEDAI_GPU_PL_MIN_DELTA_W", "5"))
    interval = float(os.environ.get("COOLEDAI_GPU_PL_MIN_INTERVAL_S", "8"))
    return full, soft, hard, min_change, interval


# ===========================================================================
# GPUPowerGovernor — Phase 2 class with CalibrationProfile integration
# ===========================================================================


class GPUPowerGovernor:
    """Stateful GPU power governor with auto-calibration and parity guard.

    Features over the free functions:
      - CalibrationProfile-driven thresholds (hardware-agnostic)
      - 300s auto-calibration observation window before adjusting
      - Workload parity guard (performance_mode blocks reduction when healthy)
      - [GPU_GOV] structured logging every 10s
      - Per-GPU hysteresis (watts + time)
    """

    CALIBRATION_WINDOW_S = 300.0  # 5 minutes observation before adjusting

    def __init__(
        self,
        envelopes: Sequence[GpuPowerEnvelope],
        *,
        performance_mode: bool = True,
        min_change_w: float = 5.0,
        min_interval_s: float = 8.0,
        min_fraction_of_default: float = 0.55,
        calibration_window_s: float = 300.0,
        dry_run: bool = False,
    ):
        self._envelopes = list(envelopes)
        self._env_by_idx = {e.index: e for e in self._envelopes}
        self._performance_mode = performance_mode
        self._min_change_w = min_change_w
        self._min_interval_s = min_interval_s
        self._min_fraction = min_fraction_of_default
        self._calibration_window_s = calibration_window_s
        self._dry_run = dry_run

        # Temperature bands (initially from env vars, overridden by profile)
        full, soft, hard, _, _ = load_governor_config_from_env()
        self._temp_full_c = full
        self._temp_soft_c = soft
        self._temp_hard_c = hard
        self._spike_trigger_c: Optional[float] = None  # from profile

        # Per-GPU state
        self._last_limits: Dict[int, float] = {}  # gpu_idx → last applied watts
        self._last_tick: float = 0.0  # monotonic time of last adjustment

        # Auto-calibration state
        self._start_time: float = time.monotonic()
        self._calibrated = False
        self._power_samples: List[float] = []  # all power readings during observation
        self._observed_idle_w: float = 0.0  # P10 of power readings
        self._observed_peak_w: float = 0.0  # P90 of power readings
        self._min_allowed_limit_w: Dict[int, float] = {}  # per-GPU floor after calibration

        # FIX 1: Efficiency gate — EWMA power tracking
        self._power_history: Deque[float] = deque(maxlen=30)
        self._active_compute_trigger_w: float = 0.0  # set from profile
        self._ewma_power_w: Optional[float] = None  # EWMA-smoothed power from previous tick
        self._ewma_power_alpha: float = 0.3  # smoothing factor for power EWMA

        # FIX 2: Continuous learning — rolling samples + mini-recalibration
        self._recent_samples: Deque[Tuple[float, float]] = deque(maxlen=500)  # (temp, power)
        self._last_recal_time: float = time.monotonic()
        self._recal_interval_s: float = 1800.0  # 30 minutes
        self._ewma_alpha: float = 0.20
        self._baseline_temp_mean: Optional[float] = None  # for drift detection
        self._drift_threshold_c: float = 4.0

        # Logging state
        self._last_log_time: float = 0.0

        # Clock scaling state (PATH B for fixed-TDP GPUs)
        self._fixed_tdp = is_fixed_tdp(self._envelopes)
        self._clock_caps: Dict[int, GpuClockCaps] = {}
        self._supported_clocks: Dict[int, List[int]] = {}  # gpu_idx → sorted desc
        self._clocks_reduced: Dict[int, bool] = {}  # per-GPU tracking
        if self._fixed_tdp:
            for cap in query_gpu_clock_caps():
                self._clock_caps[cap.index] = cap
                self._clocks_reduced[cap.index] = False
                self._supported_clocks[cap.index] = query_supported_graphics_clocks(cap.index)

        # IMPROVEMENT 1: Idle clock management — 1-tick detection with 2-tick exit hysteresis
        self._idle_ticks: Dict[int, int] = {e.index: 0 for e in self._envelopes}
        self._active_ticks: Dict[int, int] = {e.index: 0 for e in self._envelopes}  # exit hysteresis
        self._gpu_mode: Dict[int, str] = {e.index: "active_compute" for e in self._envelopes}

        # IMPROVEMENT 2: Memory-bound detection — per-GPU consecutive memory-bound counter
        self._membound_ticks: Dict[int, int] = {e.index: 0 for e in self._envelopes}

        # IMPROVEMENT 2b: Pre-idle prediction — power derivative tracking
        self._power_readings: Deque[float] = deque(maxlen=4)  # last 4 power readings for derivative
        self._util_readings: Deque[int] = deque(maxlen=3)  # last 3 utilization readings
        self._pre_idle_applied: Dict[int, bool] = {e.index: False for e in self._envelopes}

        # IMPROVEMENT 3: P-state race tracker — learn hardware response time
        self._idle_transition_start: Dict[int, Optional[float]] = {e.index: None for e in self._envelopes}
        self._idle_transition_power_start: Dict[int, Optional[float]] = {e.index: None for e in self._envelopes}
        self._hw_response_ewma_s: float = 5.0  # initial guess: T4 enters P-state in ~5s
        self._cooledai_response_times: Deque[float] = deque(maxlen=30)
        self._hw_response_times: Deque[float] = deque(maxlen=30)
        self._race_wins: int = 0
        self._race_losses: int = 0
        self._race_total: int = 0

        # IMPROVEMENT 4: Workload cycle timing — EWMA of inter-idle period
        self._last_active_end_time: Dict[int, Optional[float]] = {e.index: None for e in self._envelopes}
        self._avg_request_cycle_s: float = 0.0  # EWMA of time between idle periods
        self._cycle_samples: int = 0

        # IMPROVEMENT 5: Co-idle — lowest memory + core clock pair
        self._min_mem_clock: Dict[int, int] = {}
        self._min_core_clock: Dict[int, int] = {}
        if self._fixed_tdp:
            for cap in self._clock_caps.values():
                # Query lowest supported memory clock
                self._min_mem_clock[cap.index] = cap.max_memory_mhz  # fallback
                self._min_core_clock[cap.index] = (
                    self._supported_clocks[cap.index][-1]
                    if self._supported_clocks.get(cap.index) else int(cap.max_graphics_mhz * 0.10)
                )
                # Try to get min memory clock from supported clocks query
                min_mem = _query_min_memory_clock(cap.index)
                if min_mem is not None:
                    self._min_mem_clock[cap.index] = min_mem

        _log.info(
            "[GPU_GOV] Governor initialized — %d GPUs, performance_mode=%s, "
            "calibration_window=%.0fs, thresholds=%.0f/%.0f/%.0f°C, "
            "fixed_tdp=%s, clock_scaling=%s",
            len(self._envelopes), self._performance_mode,
            self._calibration_window_s,
            self._temp_full_c, self._temp_soft_c, self._temp_hard_c,
            self._fixed_tdp, bool(self._clock_caps),
        )

    @property
    def performance_mode(self) -> bool:
        return self._performance_mode

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def envelopes(self) -> List[GpuPowerEnvelope]:
        return list(self._envelopes)

    # ------------------------------------------------------------------
    # CalibrationProfile integration (Step 2)
    # ------------------------------------------------------------------

    def update_from_profile(self, profile: Any) -> None:
        """Replace hardcoded temp bands with CalibrationProfile-derived values.

        Mapping:
          Full TDP:     temp_mean_c + 0.5 × temp_stdev_c
          Begin blend:  spike_trigger_temp_c
          Floor power:  temp_p90_c
        """
        temp_mean = getattr(profile, "temp_mean_c", 0.0)
        temp_stdev = getattr(profile, "temp_stdev_c", 0.0)
        spike_trigger = getattr(profile, "spike_trigger_temp_c", 0.0)
        temp_p90 = getattr(profile, "temp_p90_c", 0.0)

        if temp_mean <= 0 or temp_p90 <= 0:
            return  # Profile not ready

        new_full = temp_mean + 0.5 * temp_stdev
        new_soft = spike_trigger if spike_trigger > new_full else new_full + 5.0
        new_hard = temp_p90 if temp_p90 > new_soft else new_soft + 5.0

        # Clamp: full must be < soft < hard, all within sane range
        new_full = max(30.0, min(85.0, new_full))
        new_soft = max(new_full + 1.0, min(90.0, new_soft))
        new_hard = max(new_soft + 1.0, min(95.0, new_hard))

        if (abs(new_full - self._temp_full_c) > 0.5 or
                abs(new_soft - self._temp_soft_c) > 0.5 or
                abs(new_hard - self._temp_hard_c) > 0.5):
            _log.info(
                "[GPU_GOV] Profile thresholds updated: full=%.1f→%.1f°C "
                "soft=%.1f→%.1f°C hard=%.1f→%.1f°C (mean=%.1f σ=%.1f p90=%.1f spike=%.1f)",
                self._temp_full_c, new_full,
                self._temp_soft_c, new_soft,
                self._temp_hard_c, new_hard,
                temp_mean, temp_stdev, temp_p90, spike_trigger,
            )
            self._temp_full_c = new_full
            self._temp_soft_c = new_soft
            self._temp_hard_c = new_hard
            self._spike_trigger_c = spike_trigger

        # FIX 1: Capture active_compute_trigger for efficiency gate
        act = getattr(profile, "active_compute_trigger_w", None)
        if isinstance(act, (int, float)) and act > 0:
            self._active_compute_trigger_w = float(act)

        # FIX 2: Set baseline temp for drift detection on first profile
        if self._baseline_temp_mean is None:
            self._baseline_temp_mean = temp_mean

    # ------------------------------------------------------------------
    # Auto-calibration observation window (Step 4)
    # ------------------------------------------------------------------

    def _check_calibration(self, gpu_power_w: Optional[float] = None) -> bool:
        """Feed power sample and check if calibration window has elapsed.

        Returns True if governor is calibrated and may adjust limits.
        """
        if self._calibrated:
            return True

        if gpu_power_w is not None and gpu_power_w > 0:
            self._power_samples.append(gpu_power_w)

        elapsed = time.monotonic() - self._start_time
        if elapsed < self._calibration_window_s:
            # Log progress periodically
            now = time.monotonic()
            if now - self._last_log_time >= 10.0:
                self._last_log_time = now
                _log.info(
                    "[GPU_GOV] calibrating — %.0fs of %.0fs",
                    elapsed, self._calibration_window_s,
                )
            return False

        # Calibration complete — compute observed power stats
        if self._power_samples:
            sorted_samples = sorted(self._power_samples)
            n = len(sorted_samples)
            p10_idx = max(0, int(n * 0.10))
            p90_idx = min(n - 1, int(n * 0.90))
            self._observed_idle_w = sorted_samples[p10_idx]
            self._observed_peak_w = sorted_samples[p90_idx]

            # Compute per-GPU min_allowed_limit: max(nvidia_min, idle × 1.10)
            for env in self._envelopes:
                self._min_allowed_limit_w[env.index] = max(
                    env.min_w,
                    self._observed_idle_w * 1.10,
                )

            _log.info(
                "[GPU_GOV] Calibration complete — %d samples, "
                "observed_idle=%.1fW observed_peak=%.1fW min_allowed=%.1fW",
                n, self._observed_idle_w, self._observed_peak_w,
                min(self._min_allowed_limit_w.values()) if self._min_allowed_limit_w else 0,
            )
        else:
            _log.info("[GPU_GOV] Calibration complete — no power samples collected, using driver limits.")

        self._calibrated = True
        return True

    # ------------------------------------------------------------------
    # Main tick (Step 3 + 5)
    # ------------------------------------------------------------------

    def tick(
        self,
        gpu_temps_c: Sequence[float],
        gpu_power_w: Optional[Sequence[float]] = None,
    ) -> List[Tuple[int, float, str]]:
        """Process one agent cycle.

        Returns list of (gpu_index, applied_watts, reason) for GPUs where
        power limit was changed. Empty list if nothing changed.

        Reasons: "temperature_band", "hysteresis_hold", "idle",
                 "calibrating", "parity_guard"
        """
        now = time.monotonic()
        results: List[Tuple[int, float, str]] = []

        # Feed calibration with max power sample
        max_power = max(gpu_power_w) if gpu_power_w else None
        if not self._check_calibration(max_power):
            return results  # Still calibrating — no adjustments

        # FIX 1: Update rolling power history + EWMA
        if max_power is not None:
            self._power_history.append(max_power)
            if self._ewma_power_w is None:
                self._ewma_power_w = max_power
            else:
                self._ewma_power_w = (self._ewma_power_alpha * max_power +
                                      (1 - self._ewma_power_alpha) * self._ewma_power_w)

        # FIX 2: Feed continuous learning samples
        if gpu_temps_c and gpu_power_w:
            max_temp = max(gpu_temps_c)
            self._recent_samples.append((max_temp, max_power or 0.0))
            self._check_continuous_learning(now)

        # Time hysteresis
        if now - self._last_tick < self._min_interval_s:
            return results

        for i, temp in enumerate(gpu_temps_c):
            env = self._env_by_idx.get(i)
            if env is None:
                continue

            power_w = gpu_power_w[i] if gpu_power_w and i < len(gpu_power_w) else None

            # Compute target from temperature bands
            target_w = compute_target_power_w(
                temp, env,
                temp_full_power_c=self._temp_full_c,
                temp_soft_start_c=self._temp_soft_c,
                temp_hard_c=self._temp_hard_c,
                min_fraction_of_default=self._min_fraction,
            )

            # Apply min_allowed_limit from calibration
            cal_min = self._min_allowed_limit_w.get(i, env.min_w)
            target_w = max(cal_min, min(env.max_w, round(target_w)))

            reason = "temperature_band"

            # --- Workload parity guard (Step 5) ---
            if self._performance_mode:
                spike_temp = self._spike_trigger_c or self._temp_soft_c
                if temp < spike_temp and target_w < env.default_w:
                    # GPU is thermally healthy — don't throttle in perf mode
                    target_w = env.default_w
                    reason = "parity_guard"

            # --- FIX 1: Efficiency gate (efficiency mode only) ---
            if not self._performance_mode and target_w < env.default_w:
                if not self._efficiency_gate_open(temp, power_w):
                    target_w = env.default_w
                    reason = "efficiency_gate_blocked"
                    _log.info(
                        "[GPU_GOV] efficiency_gate=blocked temp=%.1f°C power=%.1fW "
                        "ewma_power=%.1fW spike_trigger=%.1f°C",
                        temp, power_w or 0.0,
                        self._ewma_power_w or 0.0,
                        self._spike_trigger_c or self._temp_soft_c,
                    )

            # --- IMPROVEMENT 1+2: Utilization-based clock management ---
            if self._fixed_tdp and i in self._clock_caps:
                self._tick_utilization_clock_mgmt(i, temp, power_w, env, results)
                continue  # Skip TDP-based apply for fixed-TDP GPUs

            # Idle detection: if GPU drawing < 15% of default, skip
            if power_w is not None and power_w < env.default_w * 0.15:
                target_w = env.default_w
                reason = "idle"

            # Watts hysteresis (TDP-capable GPUs only)
            prev = self._last_limits.get(i)
            if prev is not None and abs(target_w - prev) < self._min_change_w:
                reason = "hysteresis_hold"
                self._maybe_log(i, temp, prev, env.default_w, reason, power_w)
                continue

            # Apply TDP change
            if set_gpu_power_limit_w(i, int(target_w), dry_run=self._dry_run):
                self._last_limits[i] = target_w
                results.append((i, target_w, reason))
                self._maybe_log(i, temp, target_w, env.default_w, reason, power_w)
            else:
                self._maybe_log(i, temp, prev or env.default_w, env.default_w, "apply_failed", power_w)

        if results:
            self._last_tick = now

        return results

    # ------------------------------------------------------------------
    # IMPROVEMENT 1+2: Utilization-based clock management
    # ------------------------------------------------------------------

    def _tick_utilization_clock_mgmt(
        self,
        gpu_idx: int,
        temp_c: float,
        power_w: Optional[float],
        env: GpuPowerEnvelope,
        results: List[Tuple[int, float, str]],
    ) -> None:
        """Handle idle, memory-bound, and active states via clock scaling.

        Called per-GPU on fixed-TDP hardware. Uses utilization queries
        to decide clock state rather than temperature bands.

        Implements:
          IMP1: 1-tick idle detection + 2-tick exit hysteresis
          IMP2: Pre-idle prediction via power derivative
          IMP3: P-state race tracker (learn T4 hardware response)
          IMP4: Workload cycle timing prediction
          IMP5: Co-idle deep clock reduction (mem + core)
        """
        now = time.monotonic()
        cap = self._clock_caps[gpu_idx]
        supported = self._supported_clocks.get(gpu_idx, [])
        currently_reduced = self._clocks_reduced.get(gpu_idx, False)

        # Query utilization (cached per tick via self._util_cache)
        sm_pct, mem_pct = self._get_gpu_util(gpu_idx)

        # Track power + utilization for derivative (IMP2)
        if power_w is not None:
            self._power_readings.append(power_w)
        self._util_readings.append(sm_pct)

        # Thermal headroom tracking (every tick)
        spike_temp = self._spike_trigger_c or self._temp_soft_c
        headroom = spike_temp - temp_c
        current_clocks = query_gpu_current_clocks()
        cur_gfx = current_clocks.get(gpu_idx, 0)
        boost_sustained = (cur_gfx >= cap.max_graphics_mhz - 15)
        _log.info(
            "[GPU_GOV] tick gpu=%d thermal_headroom=%.1f°C boost_sustained=%s "
            "sm_util=%d%% mem_util=%d%% temp=%.1f°C power=%.1fW gfx_clock=%dMHz mode=%s",
            gpu_idx, headroom, boost_sustained,
            sm_pct, mem_pct, temp_c, power_w or 0.0, cur_gfx,
            self._gpu_mode.get(gpu_idx, "unknown"),
        )
        if headroom > 10.0 and boost_sustained:
            _log.info(
                "[GPU_GOV] fan_optimization_benefit=confirmed gpu=%d "
                "headroom=%.1f°C — fan layer enabling GPU boost",
                gpu_idx, headroom,
            )

        # Efficiency gate check — never reduce if gate is blocked
        gate_blocked = not self._efficiency_gate_open(temp_c, power_w)

        # === IMP3: P-state race tracking — detect idle transitions ===
        if sm_pct < 5 and self._idle_transition_start.get(gpu_idx) is None:
            # Transition START: utilization just dropped below 5%
            self._idle_transition_start[gpu_idx] = now
            self._idle_transition_power_start[gpu_idx] = power_w

        if sm_pct >= 15 and self._idle_transition_start.get(gpu_idx) is not None:
            # Transition ENDED: utilization came back before settling
            self._idle_transition_start[gpu_idx] = None
            self._idle_transition_power_start[gpu_idx] = None

        # Detect hardware P-state engagement (power dropped below 20W)
        if (self._idle_transition_start.get(gpu_idx) is not None
                and power_w is not None and power_w < 20.0):
            hw_start = self._idle_transition_start[gpu_idx]
            if hw_start is not None:
                hw_response_s = now - hw_start
                self._hw_response_times.append(hw_response_s)
                # EWMA update
                alpha = 0.3
                self._hw_response_ewma_s = (
                    alpha * hw_response_s + (1 - alpha) * self._hw_response_ewma_s
                )
                # Did CooledAI beat the hardware?
                cooledai_responded = self._clocks_reduced.get(gpu_idx, False)
                self._race_total += 1
                if cooledai_responded:
                    self._race_wins += 1
                    race = "win"
                else:
                    self._race_losses += 1
                    race = "loss"

                win_rate = (self._race_wins / max(1, self._race_total)) * 100.0
                _log.info(
                    "[GPU_GOV] idle_race=%s hardware_response=%.1fs "
                    "hw_ewma=%.1fs win_rate_session=%.0f%% gpu=%d",
                    race, hw_response_s, self._hw_response_ewma_s,
                    win_rate, gpu_idx,
                )
                self._idle_transition_start[gpu_idx] = None
                self._idle_transition_power_start[gpu_idx] = None

        # === IMP4: Track workload cycle timing ===
        prev_mode = self._gpu_mode.get(gpu_idx, "active_compute")
        if prev_mode in ("idle_efficiency", "pre_idle") and sm_pct >= 15:
            # Just exited idle — record cycle timing
            if self._last_active_end_time.get(gpu_idx) is not None:
                cycle_s = now - self._last_active_end_time[gpu_idx]
                if 2.0 < cycle_s < 600.0:  # sane range
                    self._cycle_samples += 1
                    alpha = 0.2
                    if self._avg_request_cycle_s <= 0:
                        self._avg_request_cycle_s = cycle_s
                    else:
                        self._avg_request_cycle_s = (
                            alpha * cycle_s + (1 - alpha) * self._avg_request_cycle_s
                        )

        if sm_pct < 5 and prev_mode == "active_compute":
            self._last_active_end_time[gpu_idx] = now

        # === State machine: idle / pre_idle / memory_bound / active_compute ===

        # IMP1: Idle detection — 1 tick (was 3)
        if sm_pct < 5:
            self._idle_ticks[gpu_idx] = self._idle_ticks.get(gpu_idx, 0) + 1
            self._active_ticks[gpu_idx] = 0
            self._membound_ticks[gpu_idx] = 0
        else:
            # IMP1: 2-tick exit hysteresis — require 2 consecutive active ticks to leave idle
            if sm_pct >= 15:
                self._active_ticks[gpu_idx] = self._active_ticks.get(gpu_idx, 0) + 1
            else:
                self._active_ticks[gpu_idx] = 0
            self._idle_ticks[gpu_idx] = 0

        # IMP2: Memory-bound detection
        if sm_pct < 40 and mem_pct > 80:
            self._membound_ticks[gpu_idx] = self._membound_ticks.get(gpu_idx, 0) + 1
        else:
            self._membound_ticks[gpu_idx] = 0

        # === IMP2: Pre-idle prediction via power derivative ===
        pre_idle_predicted = False
        if (len(self._power_readings) >= 3 and len(self._util_readings) >= 2
                and not gate_blocked):
            readings = list(self._power_readings)
            # Power derivative: average W/tick drop over last 3 readings
            derivs = [readings[j] - readings[j - 1] for j in range(1, len(readings))]
            avg_deriv = sum(derivs) / len(derivs)
            cur_util = self._util_readings[-1]
            prev_util = self._util_readings[-2] if len(self._util_readings) >= 2 else cur_util
            util_falling = cur_util < prev_util

            if (avg_deriv < -15.0  # dropping > 15W/tick
                    and cur_util < 30
                    and util_falling
                    and temp_c < 65.0):
                pre_idle_predicted = True
                if prev_mode not in ("idle_efficiency", "pre_idle"):
                    _log.info(
                        "[GPU_GOV] pre_idle_prediction=True derivative=%.1fW/tick "
                        "utilization=%d%% temp=%.1f°C pre_applying_efficiency_clocks gpu=%d",
                        avg_deriv, cur_util, temp_c, gpu_idx,
                    )

        # === IMP4: Workload cycle prediction ===
        cycle_predicted = False
        if (self._avg_request_cycle_s > 0
                and self._cycle_samples >= 3
                and not gate_blocked
                and prev_mode == "active_compute"):
            last_end = self._last_active_end_time.get(gpu_idx)
            if last_end is not None:
                time_since = now - last_end
                if (time_since > self._avg_request_cycle_s * 0.7
                        and sm_pct < 20):
                    cycle_predicted = True
                    if not self._pre_idle_applied.get(gpu_idx, False):
                        _log.info(
                            "[GPU_GOV] cycle_prediction=True avg_cycle=%.1fs "
                            "time_since_last_active=%.1fs utilization=%d%% gpu=%d",
                            self._avg_request_cycle_s, time_since, sm_pct, gpu_idx,
                        )

        # --- Transition to idle_efficiency (IMP1: 1 tick, IMP5: deep clocks) ---
        should_idle = (self._idle_ticks.get(gpu_idx, 0) >= 1  # IMP1: 1-tick
                       or pre_idle_predicted              # IMP2: pre-idle prediction
                       or cycle_predicted)                # IMP4: cycle prediction

        if should_idle and not gate_blocked:
            if prev_mode not in ("idle_efficiency", "pre_idle"):
                # IMP5: Use minimum BOTH memory and core clocks for deep idle
                min_gfx = self._min_core_clock.get(
                    gpu_idx,
                    supported[-1] if supported else int(cap.max_graphics_mhz * 0.50),
                )
                min_mem = self._min_mem_clock.get(gpu_idx, cap.max_memory_mhz)
                if set_gpu_clocks(gpu_idx, min_mem, min_gfx, dry_run=self._dry_run):
                    self._clocks_reduced[gpu_idx] = True
                    new_mode = "pre_idle" if (pre_idle_predicted or cycle_predicted) else "idle_efficiency"
                    self._gpu_mode[gpu_idx] = new_mode
                    self._pre_idle_applied[gpu_idx] = True
                    _log.info(
                        "[GPU_GOV] mode=%s clocks_reduced=True gpu=%d "
                        "gfx→%dMHz mem→%dMHz utilization=%d%% power_draw=%.1fW",
                        new_mode, gpu_idx, min_gfx, min_mem, sm_pct, power_w or 0.0,
                    )
                    results.append((gpu_idx, env.default_w, new_mode))
            return

        # --- Transition to memory_bound ---
        if self._membound_ticks.get(gpu_idx, 0) >= 5 and not gate_blocked:
            if prev_mode != "memory_bound":
                target_gfx = int(cap.max_graphics_mhz * 0.75)
                reduced_gfx = snap_to_supported_clock(target_gfx, supported) if supported else target_gfx
                if set_gpu_clocks(gpu_idx, cap.max_memory_mhz, reduced_gfx, dry_run=self._dry_run):
                    self._clocks_reduced[gpu_idx] = True
                    self._gpu_mode[gpu_idx] = "memory_bound"
                    _log.info(
                        "[GPU_GOV] mode=memory_bound sm=%d%% mem_util=%d%% gpu=%d "
                        "core_clock_reduced=True gfx→%dMHz",
                        sm_pct, mem_pct, gpu_idx, reduced_gfx,
                    )
                    results.append((gpu_idx, env.default_w, "memory_bound"))
            return

        # --- Transition to active_compute (immediate restore) ---
        # IMP1: 2-tick hysteresis to exit idle — require 2 consecutive active ticks
        need_restore = False
        active_count = self._active_ticks.get(gpu_idx, 0)
        if prev_mode in ("idle_efficiency", "pre_idle") and active_count >= 2:
            need_restore = True
        elif prev_mode == "memory_bound" and sm_pct >= 60:
            need_restore = True
        elif prev_mode in ("idle_efficiency", "pre_idle", "memory_bound") and gate_blocked:
            need_restore = True  # Gate demands full power

        if need_restore and currently_reduced:
            if reset_gpu_clocks(gpu_idx, dry_run=self._dry_run):
                self._clocks_reduced[gpu_idx] = False
                self._gpu_mode[gpu_idx] = "active_compute"
                self._pre_idle_applied[gpu_idx] = False
                _log.info(
                    "[GPU_GOV] mode=active_compute clocks_restored=True gpu=%d "
                    "utilization=%d%% from=%s",
                    gpu_idx, sm_pct, prev_mode,
                )
                results.append((gpu_idx, env.default_w, "active_compute"))
            return

        # No transition needed — hold current state
        self._maybe_log(gpu_idx, temp_c, env.default_w, env.default_w,
                        prev_mode, power_w)

    def _get_gpu_util(self, gpu_idx: int) -> Tuple[int, int]:
        """Get SM and memory utilization for a GPU, caching per tick."""
        if not hasattr(self, "_util_cache_time") or (time.monotonic() - self._util_cache_time) > 1.0:
            self._util_cache: Dict[int, Tuple[int, int]] = {}
            for idx, sm, mem in query_gpu_utilization():
                self._util_cache[idx] = (sm, mem)
            self._util_cache_time = time.monotonic()
        return self._util_cache.get(gpu_idx, (0, 0))

    # ------------------------------------------------------------------
    # FIX 1: Efficiency gate
    # ------------------------------------------------------------------

    def _efficiency_gate_open(self, temp_c: float, power_w: Optional[float]) -> bool:
        """Check if efficiency-mode TDP reduction is allowed.

        Returns True if reduction is permitted, False if blocked.
        BLOCKS reduction when ALL three conditions are simultaneously true:
          1. Current GPU power > 90W
          2. Current GPU temperature is within 3°C of spike_trigger_temp_c
          3. GPU power trend is positive (current power > EWMA from previous tick)
        """
        spike_temp = self._spike_trigger_c or self._temp_soft_c

        # All three must be true to BLOCK
        cond_power_high = (power_w is not None and power_w > 90.0)
        cond_near_spike = (temp_c >= spike_temp - 3.0)
        cond_power_ramping = (
            power_w is not None
            and self._ewma_power_w is not None
            and power_w > self._ewma_power_w
        )

        if cond_power_high and cond_near_spike and cond_power_ramping:
            return False  # BLOCKED — all three danger conditions met

        return True  # OPEN — safe to reduce

    # ------------------------------------------------------------------
    # FIX 2: Continuous learning
    # ------------------------------------------------------------------

    def _check_continuous_learning(self, now: float) -> None:
        """Perform mini-recalibration and drift detection from rolling samples."""
        if len(self._recent_samples) < 30:
            return  # Need minimum samples

        # Drift detection: check if recent temp mean shifted > 4°C from baseline
        if self._baseline_temp_mean is not None:
            recent_temps = [t for t, _ in list(self._recent_samples)[-100:]]
            recent_mean = sum(recent_temps) / len(recent_temps)
            drift = abs(recent_mean - self._baseline_temp_mean)
            if drift >= self._drift_threshold_c:
                _log.info(
                    "[GPU_GOV] Drift detected: baseline=%.1f°C current_mean=%.1f°C "
                    "drift=%.1f°C — triggering immediate recalibration",
                    self._baseline_temp_mean, recent_mean, drift,
                )
                self._apply_mini_recalibration()
                self._baseline_temp_mean = recent_mean
                self._last_recal_time = now
                return

        # Periodic mini-recalibration every _recal_interval_s
        if now - self._last_recal_time >= self._recal_interval_s:
            self._apply_mini_recalibration()
            self._last_recal_time = now

    def _apply_mini_recalibration(self) -> None:
        """EWMA-smooth observed idle/peak power from recent samples."""
        if not self._recent_samples:
            return

        powers = sorted(p for _, p in self._recent_samples if p > 0)
        if len(powers) < 10:
            return

        n = len(powers)
        new_idle = powers[max(0, int(n * 0.10))]
        new_peak = powers[min(n - 1, int(n * 0.90))]

        alpha = self._ewma_alpha
        if self._observed_idle_w > 0:
            self._observed_idle_w = alpha * new_idle + (1 - alpha) * self._observed_idle_w
        else:
            self._observed_idle_w = new_idle

        if self._observed_peak_w > 0:
            self._observed_peak_w = alpha * new_peak + (1 - alpha) * self._observed_peak_w
        else:
            self._observed_peak_w = new_peak

        # Update per-GPU min_allowed_limit
        for env in self._envelopes:
            self._min_allowed_limit_w[env.index] = max(
                env.min_w,
                self._observed_idle_w * 1.10,
            )

        _log.info(
            "[GPU_GOV] Mini-recalibration: idle=%.1fW peak=%.1fW (EWMA α=%.2f, %d samples)",
            self._observed_idle_w, self._observed_peak_w, alpha, len(powers),
        )

    def _maybe_log(
        self,
        gpu_idx: int,
        temp_c: float,
        current_limit_w: float,
        default_w: float,
        reason: str,
        power_w: Optional[float],
    ) -> None:
        """Emit [GPU_GOV] structured log every 10 seconds."""
        now = time.monotonic()
        if now - self._last_log_time < 10.0:
            return
        self._last_log_time = now

        reduction_pct = 0.0
        if default_w > 0 and current_limit_w < default_w:
            reduction_pct = (1.0 - current_limit_w / default_w) * 100.0

        mode = "performance" if self._performance_mode else "efficiency"
        _log.info(
            "[GPU_GOV] gpu=%d gpu_temp=%.1f°C current_limit=%.0fW "
            "default_limit=%.0fW reduction=%.1f%% reason=%s mode=%s "
            "power_draw=%.1fW bands=%.0f/%.0f/%.0f°C",
            gpu_idx, temp_c, current_limit_w,
            default_w, reduction_pct, reason, mode,
            power_w or 0.0,
            self._temp_full_c, self._temp_soft_c, self._temp_hard_c,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def reset_to_defaults(self) -> None:
        """Restore all GPUs to driver default power limits and clocks."""
        reset_all_gpus_to_default(self._envelopes, dry_run=self._dry_run)
        # Reset clocks on fixed-TDP GPUs
        for idx in self._clocks_reduced:
            if self._clocks_reduced[idx]:
                reset_gpu_clocks(idx, dry_run=self._dry_run)
                self._clocks_reduced[idx] = False
        self._last_limits.clear()
        _log.info("[GPU_GOV] All GPUs restored to default power limits and clocks.")

    # ------------------------------------------------------------------
    # FIX 2: State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist governor learning state to JSON alongside calibration profile."""
        state = {
            "observed_idle_w": self._observed_idle_w,
            "observed_peak_w": self._observed_peak_w,
            "baseline_temp_mean": self._baseline_temp_mean,
            "min_allowed_limit_w": self._min_allowed_limit_w,
            "calibrated": self._calibrated,
            "temp_full_c": self._temp_full_c,
            "temp_soft_c": self._temp_soft_c,
            "temp_hard_c": self._temp_hard_c,
            "spike_trigger_c": self._spike_trigger_c,
            "active_compute_trigger_w": self._active_compute_trigger_w,
            "performance_mode": self._performance_mode,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        _log.info("[GPU_GOV] State saved to %s", path)

    def load_state(self, path: str) -> bool:
        """Restore governor learning state from JSON. Returns True on success."""
        try:
            with open(path) as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            _log.warning("[GPU_GOV] Failed to load state from %s: %s", path, e)
            return False

        self._observed_idle_w = state.get("observed_idle_w", self._observed_idle_w)
        self._observed_peak_w = state.get("observed_peak_w", self._observed_peak_w)
        self._baseline_temp_mean = state.get("baseline_temp_mean", self._baseline_temp_mean)
        loaded_min = state.get("min_allowed_limit_w", {})
        # JSON keys are strings — convert back to int
        self._min_allowed_limit_w = {int(k): v for k, v in loaded_min.items()}
        self._calibrated = state.get("calibrated", self._calibrated)
        self._temp_full_c = state.get("temp_full_c", self._temp_full_c)
        self._temp_soft_c = state.get("temp_soft_c", self._temp_soft_c)
        self._temp_hard_c = state.get("temp_hard_c", self._temp_hard_c)
        self._spike_trigger_c = state.get("spike_trigger_c", self._spike_trigger_c)
        self._active_compute_trigger_w = state.get("active_compute_trigger_w", self._active_compute_trigger_w)
        self._performance_mode = state.get("performance_mode", self._performance_mode)

        _log.info(
            "[GPU_GOV] State loaded from %s — idle=%.1fW peak=%.1fW calibrated=%s",
            path, self._observed_idle_w, self._observed_peak_w, self._calibrated,
        )
        return True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        envelopes: Optional[Sequence[GpuPowerEnvelope]] = None,
        *,
        dry_run: bool = False,
    ) -> Optional["GPUPowerGovernor"]:
        """Create governor from environment variables and nvidia-smi probe.

        Returns None if no GPUs found.
        """
        if envelopes is None:
            envelopes = query_nvidia_power_envelopes()
        if not envelopes:
            return None

        perf_mode_str = os.environ.get("COOLEDAI_GPU_PERF_MODE", "true").strip().lower()
        performance_mode = perf_mode_str not in ("0", "false", "no", "off")

        _, _, _, min_change, interval = load_governor_config_from_env()
        cal_window = float(os.environ.get("COOLEDAI_GPU_GOV_CAL_WINDOW_S", "300"))

        return cls(
            envelopes,
            performance_mode=performance_mode,
            min_change_w=min_change,
            min_interval_s=interval,
            calibration_window_s=cal_window,
            dry_run=dry_run,
        )
