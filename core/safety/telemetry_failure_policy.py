"""
Telemetry and actuator failure hints for conservative cooling posture.

This does not replace hardware watchdogs on the agent. It nudges cloud
optimization toward safety when data is stale, fans may not be following
commands, or cooling appears saturated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class FailurePosture:
    """Reasons we tightened cooling or flagged escalation."""

    stale_history: bool = False
    stale_age_sec: float = 0.0
    fan_slippage: bool = False
    slippage_detail: str = ""
    saturation_suspected: bool = False
    saturation_detail: str = ""
    duty_floor: int = 0  # minimum duty after adjustments (0 = no floor)
    duty_boost: int = 0  # add to target duty
    reasons: List[str] = field(default_factory=list)


def assess_history_staleness(
    node_timestamps: Sequence[datetime],
    now: Optional[datetime] = None,
    max_age_sec: float = 120.0,
) -> Tuple[bool, float]:
    """
    True if the newest sample in the sliding window is older than max_age_sec.
    Returns (is_stale, age_of_newest_sample_seconds).
    """
    if not node_timestamps:
        return True, float("inf")
    now_dt = now or datetime.now(timezone.utc)
    newest = max(node_timestamps)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    age = (now_dt - newest).total_seconds()
    return age > max_age_sec, max(0.0, age)


def assess_fan_slippage(
    last_commanded_duty: Optional[float],
    fan_rpm: float,
    max_fan_rpm: float,
    *,
    min_commanded_to_check: float = 75.0,
    min_expected_fraction: float = 0.55,
    tach_stuck_below_rpm: float = 400.0,
) -> Tuple[bool, str]:
    """
    Heuristic: high commanded duty but tach suggests much lower delivered airflow.

    Uses implied duty from tach vs max_fan_rpm. Tuned for chassis fans; adjust
    fractions per site if IPMI tach is missing or scaled differently.
    """
    if last_commanded_duty is None or max_fan_rpm < 500:
        return False, ""
    if last_commanded_duty < min_commanded_to_check:
        return False, ""
    implied = max(0.0, min(100.0, (fan_rpm / max_fan_rpm) * 100.0))
    expected_min = last_commanded_duty * min_expected_fraction
    if fan_rpm < tach_stuck_below_rpm and last_commanded_duty >= min_commanded_to_check:
        return True, f"tach_stuck_low rpm={fan_rpm:.0f} duty_cmd={last_commanded_duty:.0f}"
    if implied + 15.0 < expected_min:  # allow 15% slack
        return True, f"slip implied_duty={implied:.0f}% cmd={last_commanded_duty:.0f}% rpm={fan_rpm:.0f}"
    return False, ""


def assess_temp_rise_at_max_cooling(
    temperatures: Sequence[float],
    proposed_duty: int,
    *,
    min_samples: int = 5,
    min_duty: int = 95,
    rise_per_sample_c: float = 0.08,
) -> Tuple[bool, str]:
    """
    If we're already commanding ~max cooling but temperature still drifts up,
    flag saturation (hardware or environmental limit).
    """
    if proposed_duty < min_duty or len(temperatures) < min_samples:
        return False, ""
    recent = list(temperatures)[-min_samples:]
    if recent[-1] > recent[0] + rise_per_sample_c * (len(recent) - 1):
        return True, "temp_rising_at_high_duty"
    return False, ""


def apply_failure_adjustments(
    target_duty: int,
    *,
    stale_history: bool,
    stale_age_sec: float = 0.0,
    fan_slip: bool,
    slip_detail: str = "",
    saturation: bool,
    sat_detail: str = "",
    stale_boost: int = 15,
    slip_boost: int = 12,
    slip_floor: int = 88,
    stale_floor: int = 70,
) -> Tuple[int, FailurePosture]:
    """
    Apply conservative boosts / floors. Saturation does not lower duty (we're
    already maxed); it only adds a reason for operators / downstream alerts.
    """
    posture = FailurePosture()
    d = max(0, min(100, int(target_duty)))

    if stale_history:
        posture.stale_history = True
        posture.stale_age_sec = stale_age_sec
        posture.duty_boost = max(posture.duty_boost, stale_boost)
        posture.duty_floor = max(posture.duty_floor, stale_floor)
        posture.reasons.append("stale_telemetry_window")

    if fan_slip:
        posture.fan_slippage = True
        posture.slippage_detail = slip_detail or "fan_slippage"
        posture.duty_boost = max(posture.duty_boost, slip_boost)
        posture.duty_floor = max(posture.duty_floor, slip_floor)
        posture.reasons.append("fan_slippage_heuristic")

    if saturation:
        posture.saturation_suspected = True
        posture.saturation_detail = sat_detail or "temp_rising_at_high_duty"
        posture.reasons.append("cooling_saturation_suspected")

    out = d + posture.duty_boost
    out = max(out, posture.duty_floor)
    out = max(0, min(100, out))
    return out, posture


def build_posture_from_nodes(
    nodes: Sequence[Any],
    *,
    target_duty: int,
    last_commanded_duty: Optional[float],
    fan_rpm: float,
    max_fan_rpm: float,
    now: Optional[datetime] = None,
) -> Tuple[int, FailurePosture]:
    """Convenience: extract timestamps and temps from BaseNode-like objects."""
    ts_list: List[datetime] = []
    temps: List[float] = []
    for n in nodes:
        tstamp = getattr(n, "timestamp", None)
        if isinstance(tstamp, datetime):
            ts_list.append(tstamp)
        ti = getattr(n, "thermal_input", None)
        if isinstance(ti, (int, float)) and not isinstance(ti, bool):
            temps.append(float(ti))

    stale, age = assess_history_staleness(ts_list, now=now)
    slip, slip_msg = assess_fan_slippage(last_commanded_duty, fan_rpm, max_fan_rpm)
    sat, sat_msg = assess_temp_rise_at_max_cooling(temps, target_duty)

    return apply_failure_adjustments(
        target_duty,
        stale_history=stale,
        stale_age_sec=age if stale else 0.0,
        fan_slip=slip,
        slip_detail=slip_msg,
        saturation=sat,
        sat_detail=sat_msg,
    )
