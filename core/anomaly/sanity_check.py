"""
CooledAI Sanity Check - Sensor Ghost Detection

If a temperature reading jumps by more than 20°C in a single second, flag it as
a 'Sensor Ghost' and ignore it (replace with last valid reading). Prevents
electrical noise and faulty sensor spikes from driving fan jitter.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.hal.base_node import BaseNode

_logger = logging.getLogger("cooledai.sanity_check")

# If |ΔT| > 20°C within MAX_DT_SECONDS, treat as Sensor Ghost
SENSOR_GHOST_THRESHOLD_C = 20.0
SENSOR_GHOST_MAX_DT_SECONDS = 1.5  # Allow slight tolerance for "single second"
# Rate-based: if |ΔT|/dt > 20°C/s for any dt, treat as Sensor Ghost (handles variable poll intervals)
SENSOR_GHOST_RATE_C_PER_S = 20.0


@dataclass
class SensorGhostFlag:
    """Record of a Sensor Ghost event for logging."""

    node_id: str
    raw_temp: float
    used_temp: float
    delta_c: float
    dt_seconds: float


def _get_temperature(node: BaseNode) -> Optional[float]:
    """Extract primary temperature from node."""
    t = getattr(node, "thermal_input", None)
    if t is not None and (isinstance(t, (int, float)) and not (isinstance(t, bool))):
        return float(t)
    return None


def _get_timestamp(node: BaseNode) -> Optional[datetime]:
    """Get node timestamp."""
    ts = getattr(node, "timestamp", None)
    if isinstance(ts, datetime):
        return ts
    return None


def filter_sensor_ghost(
    nodes: List[BaseNode],
    previous_state: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold_c: float = SENSOR_GHOST_THRESHOLD_C,
    max_dt_seconds: float = SENSOR_GHOST_MAX_DT_SECONDS,
    rate_threshold_c_per_s: float = SENSOR_GHOST_RATE_C_PER_S,
) -> Tuple[Dict[str, Dict[str, Any]], List[SensorGhostFlag]]:
    """
    Sanity Check: Replace Sensor Ghost readings with last valid temperature.

    If |ΔT| > 20°C in a single second (or rate > 20°C/s), flag as 'Sensor Ghost'
    and ignore—replace with last valid reading. Prevents electrical noise and
    faulty sensor spikes from driving fan jitter.

    Mutates nodes in place. Updates thermal_input where ghost detected.

    Args:
        nodes: List of BaseNode (mutated).
        previous_state: node_id -> {thermal_input, timestamp} from last call.
        threshold_c: Max plausible °C change within max_dt_seconds (default 20).
        max_dt_seconds: Time window for "single second" check (default 1.5).
        rate_threshold_c_per_s: Max plausible °C/s for any dt (default 20).

    Returns:
        (new_state for next call, list of SensorGhostFlag for logging).
    """
    state: Dict[str, Dict[str, Any]] = (
        {k: dict(v) for k, v in previous_state.items()} if previous_state else {}
    )
    flags: List[SensorGhostFlag] = []

    for node in nodes:
        nid = getattr(node, "node_id", "") or ""
        temp = _get_temperature(node)
        ts = _get_timestamp(node)

        if temp is None:
            continue

        prev = state.get(nid)
        if prev is not None:
            prev_temp = prev.get("thermal_input")
            prev_ts = prev.get("timestamp")
            if prev_temp is not None and prev_ts is not None and ts is not None:
                try:
                    dt = (ts - prev_ts).total_seconds()
                    if dt > 0:
                        delta = abs(float(temp) - float(prev_temp))
                        rate_c_per_s = delta / dt
                        # Sensor Ghost: >20°C in ~1s OR rate > 20°C/s (handles variable poll intervals)
                        is_ghost = (
                            (dt <= max_dt_seconds and delta > threshold_c)
                            or rate_c_per_s > rate_threshold_c_per_s
                        )
                        if is_ghost:
                            node.thermal_input = float(prev_temp)
                            flags.append(
                                SensorGhostFlag(
                                    node_id=nid,
                                    raw_temp=float(temp),
                                    used_temp=float(prev_temp),
                                    delta_c=delta,
                                    dt_seconds=dt,
                                )
                            )
                            _logger.info(
                                "Sensor Ghost: %s |ΔT|=%.1f°C in %.2fs (%.1f°C/s) (raw=%.1f°C, used=%.1f°C)",
                                nid,
                                delta,
                                dt,
                                rate_c_per_s,
                                temp,
                                prev_temp,
                            )
                            temp = float(prev_temp)
                except (TypeError, AttributeError):
                    pass

        if temp is not None and ts is not None:
            state[nid] = {"thermal_input": temp, "timestamp": ts}

    return state, flags


class SanityCheckFilter:
    """
    Stateful filter for streaming telemetry. Replaces Sensor Ghost readings.

    If a temperature jumps by more than 20°C in a single second (or rate > 20°C/s),
    flag as 'Sensor Ghost' and ignore—replace with last valid reading. Prevents
    fan jitter from electrical noise.
    """

    def __init__(
        self,
        threshold_c: float = SENSOR_GHOST_THRESHOLD_C,
        max_dt_seconds: float = SENSOR_GHOST_MAX_DT_SECONDS,
        rate_threshold_c_per_s: float = SENSOR_GHOST_RATE_C_PER_S,
    ):
        self._state: Dict[str, Dict[str, Any]] = {}
        self.threshold_c = threshold_c
        self.max_dt_seconds = max_dt_seconds
        self.rate_threshold_c_per_s = rate_threshold_c_per_s

    def filter(self, nodes: List[BaseNode]) -> List[SensorGhostFlag]:
        """Apply sanity check to nodes. Mutates in place. Returns flags."""
        self._state, flags = filter_sensor_ghost(
            nodes,
            self._state,
            threshold_c=self.threshold_c,
            max_dt_seconds=self.max_dt_seconds,
            rate_threshold_c_per_s=self.rate_threshold_c_per_s,
        )
        return flags

    def reset(self) -> None:
        """Clear state (e.g. after topology change)."""
        self._state.clear()
