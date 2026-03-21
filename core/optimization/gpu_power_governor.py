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

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
            ["nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))],
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

        # Logging state
        self._last_log_time: float = 0.0

        _log.info(
            "[GPU_GOV] Governor initialized — %d GPUs, performance_mode=%s, "
            "calibration_window=%.0fs, thresholds=%.0f/%.0f/%.0f°C",
            len(self._envelopes), self._performance_mode,
            self._calibration_window_s,
            self._temp_full_c, self._temp_soft_c, self._temp_hard_c,
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

            # Idle detection: if GPU drawing < 15% of default, skip
            if power_w is not None and power_w < env.default_w * 0.15:
                target_w = env.default_w
                reason = "idle"

            # Watts hysteresis
            prev = self._last_limits.get(i)
            if prev is not None and abs(target_w - prev) < self._min_change_w:
                reason = "hysteresis_hold"
                # Still log but don't apply
                self._maybe_log(i, temp, prev, env.default_w, reason, power_w)
                continue

            # Apply
            if set_gpu_power_limit_w(i, int(target_w), dry_run=self._dry_run):
                self._last_limits[i] = target_w
                results.append((i, target_w, reason))
                self._maybe_log(i, temp, target_w, env.default_w, reason, power_w)
            else:
                self._maybe_log(i, temp, prev or env.default_w, env.default_w, "apply_failed", power_w)

        if results:
            self._last_tick = now

        return results

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
        """Restore all GPUs to driver default power limits."""
        reset_all_gpus_to_default(self._envelopes, dry_run=self._dry_run)
        self._last_limits.clear()
        _log.info("[GPU_GOV] All GPUs restored to default power limits.")

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
