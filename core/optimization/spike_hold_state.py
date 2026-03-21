"""
Shared post-spike fan hold (hysteresis) state.

Any code path that calls OptimizationBrain with ``thermal_session_key`` gets the same
4-minute elevated minimum RPM after max observed temperature exceeds
``SPIKE_HOLD_ENTER_TEMP_C`` — not only ``/api/v1/optimize/control``.

Callers with extra context (e.g. rolling API history) should also invoke
``record_spike_if_hot`` with the max temperature they know about before ``analyze``,
so spikes visible only in history still extend the hold.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

_lock = threading.Lock()
_until: Dict[str, float] = {}


def record_spike_if_hot(
    session_key: str,
    max_temp_c: float,
    now: Optional[float] = None,
    spike_trigger_temp_c: Optional[float] = None,
    spike_hold_duration_s: Optional[float] = None,
) -> None:
    """If ``max_temp_c`` exceeds the hold threshold, extend hold window for ``session_key``.

    When ``spike_trigger_temp_c`` is provided (from CalibrationProfile), use it
    instead of the hardcoded default.  This prevents the hold from being
    permanently engaged when normal operating temps exceed the 42°C default.
    """
    if not session_key:
        return
    from core.optimization.cooling_safety_policy import (
        SPIKE_HOLD_DURATION_S,
        SPIKE_HOLD_ENTER_TEMP_C,
    )

    threshold = spike_trigger_temp_c if spike_trigger_temp_c is not None else SPIKE_HOLD_ENTER_TEMP_C
    duration = spike_hold_duration_s if spike_hold_duration_s is not None else SPIKE_HOLD_DURATION_S

    t = time.time() if now is None else float(now)
    if max_temp_c > threshold:
        with _lock:
            _until[session_key] = t + duration


def spike_hold_floor_rpm(
    session_key: Optional[str],
    max_fan_rpm: float,
    now: Optional[float] = None,
) -> float:
    """
    Minimum RPM implied by an active spike hold for this session, or 0.0 if inactive.

    Uses the same cap style as the former agent-only policy_context path.
    """
    if not session_key:
        return 0.0
    from core.optimization.cooling_safety_policy import SPIKE_HOLD_MIN_FAN_RPM

    t = time.time() if now is None else float(now)
    with _lock:
        until = _until.get(session_key, 0.0)
    if t >= until:
        return 0.0
    mx = max(max_fan_rpm, 1000.0)
    return float(min(SPIKE_HOLD_MIN_FAN_RPM, mx * 0.92))


def clear_spike_hold(session_key: str) -> None:
    """Testing / admin: drop hold state for a session."""
    if not session_key:
        return
    with _lock:
        _until.pop(session_key, None)
