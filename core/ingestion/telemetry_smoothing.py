"""
CooledAI Telemetry Smoothing - Moving Average Filter

Applies a moving average (e.g. last 5 readings) to smooth telemetry before it
hits the OptimizationBrain. Prevents fan jitter from electrical noise.
"""

from collections import deque
from typing import Any, Dict, List

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
