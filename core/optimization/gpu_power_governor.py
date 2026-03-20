"""
Dynamic NVIDIA GPU power cap (nvidia-smi -pl) to reduce hard thermal throttling.

When GPU temperature is comfortable, allow default TDP. As temperature approaches
a soft band, reduce power limit smoothly toward a floor so the driver throttles
gracefully instead of hitting sudden clock drops (poor perf/W).

All temperature bands are configurable via environment variables.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


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


def read_current_power_limits(timeout_s: float = 8.0) -> dict[int, float]:
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
    cur: dict[int, float] = {}
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
