"""
CooledAI Thermal First-Derivative (Rate of Change) Enrichment

Computes dT/dt for all temperature sensors so the AI can see thermal momentum,
not just current temperature. Used by the ingestion pipeline and API.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from core.hal.base_node import BaseNode

# Minimum time delta (seconds) to compute rate; avoids division by tiny dt
MIN_DT_SECONDS = 0.1
# Cap magnitude of rate (°C/s) to avoid spikes from bad samples
MAX_RATE_MAGNITUDE = 50.0


def _get_temperature(node: BaseNode) -> Optional[float]:
    """Extract primary temperature from node.

    Checks thermal_input first, then raw_data for tdie, tctl, temp, etc.

    Args:
        node: BaseNode with thermal data.

    Returns:
        Temperature in °C, or None if not found.
    """
    if node.thermal_input is not None and node.thermal_input > 0:
        return float(node.thermal_input)
    raw = getattr(node, "raw_data", None) or {}
    for key in ("thermal_input", "tdie", "tctl", "temp", "temperature", "cpu temp"):
        for k, v in raw.items():
            if key in str(k).lower() and v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None


def _get_timestamp(node: BaseNode) -> Optional[datetime]:
    """Get node timestamp.

    Args:
        node: BaseNode with optional timestamp.

    Returns:
        datetime if present, else None.
    """
    ts = getattr(node, "timestamp", None)
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    return None


def compute_derivatives(
    nodes: List[BaseNode],
    previous_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute first derivative (rate of change) of temperature for each node.
    Updates each node's thermal_input_rate in place and returns new state for next call.

    State is keyed by node_id: { "thermal_input": float, "timestamp": datetime }.
    Pass the returned state as previous_state on the next batch so streaming
    telemetry gets correct dT/dt across batches.

    Args:
        nodes: List of BaseNode (will be mutated: thermal_input_rate set where computable).
        previous_state: Optional state from last call (node_id -> {thermal_input, timestamp}).

    Returns:
        New state dict to pass as previous_state on next invocation.
    """
    state: Dict[str, Dict[str, Any]] = previous_state.copy() if previous_state else {}

    for node in nodes:
        nid = getattr(node, "node_id", "") or ""
        temp = _get_temperature(node)
        ts = _get_timestamp(node)

        rate: Optional[float] = None
        if nid in state and temp is not None and ts is not None:
            prev = state[nid]
            prev_temp = prev.get("thermal_input")
            prev_ts = prev.get("timestamp")
            if prev_temp is not None and prev_ts is not None:
                try:
                    dt = (ts - prev_ts).total_seconds()
                    if dt >= MIN_DT_SECONDS:
                        rate = (temp - prev_temp) / dt
                        if rate is not None and abs(rate) > MAX_RATE_MAGNITUDE:
                            rate = MAX_RATE_MAGNITUDE if rate > 0 else -MAX_RATE_MAGNITUDE
                except (TypeError, AttributeError):
                    pass

        setattr(node, "thermal_input_rate", rate)

        if temp is not None and ts is not None:
            state[nid] = {"thermal_input": temp, "timestamp": ts}

    return state


class TelemetryDerivativeEnricher:
    """Stateful enricher for streaming telemetry.

    Maintains last (temperature, timestamp) per node_id and computes
    thermal_input_rate (dT/dt) on each enrich() call. Use for real-time
    API ingestion where compute_derivatives needs state across batches.
    """

    def __init__(self):
        """Initialize with empty state."""
        self._state: Dict[str, Dict[str, Any]] = {}

    def enrich(self, nodes: List[BaseNode]) -> None:
        """Compute thermal_input_rate for each node and update internal state.

        Mutates nodes in place. Updates self._state for next call.

        Args:
            nodes: List of BaseNode to enrich with thermal_input_rate.
        """
        self._state = compute_derivatives(nodes, self._state)

    def reset(self) -> None:
        """Clear state (e.g. after facility topology change)."""
        self._state.clear()
