"""
CooledAI Telemetry Smoothing - Moving Average & EWMA

Applies smoothing before telemetry hits the OptimizationBrain to reduce fan
jitter from electrical noise. EWMA is O(1) per field (no sum over a window each
tick); moving average keeps a fixed-length history for stronger low-pass filtering.
"""

from collections import deque
from typing import Dict, List

from core.hal.base_node import BaseNode


class MovingAverageFilter:
    """
    Per-node moving average filter for thermal_input, power_draw, cooling_output.

    Maintains last N readings per node_id. On each apply(), replaces raw values
    with the mean of the window. Use for streaming telemetry to smooth noise.
    """

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of samples to average (default 5).
        """
        self.window_size = max(1, window_size)
        self._history: Dict[str, Dict[str, deque]] = {}

    def _ensure_node(self, nid: str) -> None:
        if nid not in self._history:
            self._history[nid] = {
                "thermal_input": deque(maxlen=self.window_size),
                "power_draw": deque(maxlen=self.window_size),
                "cooling_output": deque(maxlen=self.window_size),
            }

    def apply(self, nodes: List[BaseNode]) -> None:
        """
        Apply moving average to nodes. Mutates nodes in place.

        For each node: append current values to per-node history, then set
        thermal_input, power_draw, cooling_output to the mean of the window.

        Args:
            nodes: List of BaseNode to smooth.
        """
        for node in nodes:
            nid = getattr(node, "node_id", "") or ""
            self._ensure_node(nid)

            thermal = float(getattr(node, "thermal_input", 0) or 0)
            power = float(getattr(node, "power_draw", 0) or 0)
            cooling = float(getattr(node, "cooling_output", 0) or 0)

            hist = self._history[nid]
            hist["thermal_input"].append(thermal)
            hist["power_draw"].append(power)
            hist["cooling_output"].append(cooling)

            # Replace with rolling mean
            if hist["thermal_input"]:
                node.thermal_input = sum(hist["thermal_input"]) / len(hist["thermal_input"])
            if hist["power_draw"]:
                node.power_draw = sum(hist["power_draw"]) / len(hist["power_draw"])
            if hist["cooling_output"]:
                node.cooling_output = sum(hist["cooling_output"]) / len(hist["cooling_output"])

    def reset(self) -> None:
        """Clear history (e.g. after topology change)."""
        self._history.clear()


class EwmaTelemetryFilter:
    """
    Per-node exponential weighted moving average (EWMA).

    Same role as MovingAverageFilter but constant time per update—better for
    high-frequency agent/API paths. alpha ≈ 2/(N+1) approximates an N-sample
    SMA (e.g. alpha=0.33 ≈ 5-sample).
    """

    def __init__(self, alpha: float = 0.33):
        self.alpha = max(0.05, min(0.95, float(alpha)))
        self._state: Dict[str, Dict[str, float]] = {}

    def _ensure_node(self, nid: str) -> Dict[str, float]:
        if nid not in self._state:
            self._state[nid] = {}
        return self._state[nid]

    def apply(self, nodes: List[BaseNode]) -> None:
        """Apply EWMA to nodes. Mutates nodes in place."""
        a = self.alpha
        for node in nodes:
            nid = getattr(node, "node_id", "") or ""
            st = self._ensure_node(nid)
            for key in ("thermal_input", "power_draw", "cooling_output"):
                raw = float(getattr(node, key, 0) or 0)
                prev = st.get(key)
                if prev is None:
                    st[key] = raw
                else:
                    st[key] = a * raw + (1.0 - a) * prev
                setattr(node, key, st[key])

    def reset(self) -> None:
        self._state.clear()
