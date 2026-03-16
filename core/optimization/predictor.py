"""
CooledAI Temp Predictor - Look-Ahead Predictive Modeling

Predicts temperature at T+10 seconds based on power trajectory.
Uses FOPDT (First-Order Plus Dead Time) when rack topology and calibration exist;
otherwise falls back to linear extrapolation. Self-calibrates tau, theta, gain
from telemetry per rack to handle variance across hardware and facility layouts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.hal.base_node import BaseNode

# Prediction horizon (seconds)
PREDICTION_HORIZON_S = 10.0

# Thermal inertia: typical time constant for temp decay after load drop (seconds)
# Higher = hardware holds heat longer = over-cool less during idle
DEFAULT_THERMAL_INERTIA_S = 120.0

# Calibration trigger: calibrate rack when buffer gains this many samples
CALIBRATE_EVERY_N_SAMPLES = 50


@dataclass
class PredictionResult:
    """Result of temperature prediction at T+10s.

    Attributes:
        predicted_temp: Current temperature.
        predicted_temp_t10: Predicted temperature at T+10 seconds.
        power_trajectory_slope: Power trend (W/s).
        temp_trajectory_slope: Temperature trend (°C/s).
        thermal_inertia_s: Estimated thermal time constant (seconds).
        confidence: Prediction confidence in [0, 1].
    """
    predicted_temp: float
    predicted_temp_t10: float  # T+10 seconds
    power_trajectory_slope: float  # W/s
    temp_trajectory_slope: float   # °C/s
    thermal_inertia_s: float
    confidence: float  # 0-1


class TempPredictor:
    """Predicts temperature at T+10s from power and temp trajectories.

    When rack topology and FOPDT calibration exist: uses First-Order Plus Dead
    Time model to account for delay between fan speed increase and temperature
    decrease. Self-calibrates tau, theta, gain from telemetry per rack.

    Otherwise: linear extrapolation T_pred = T_now + (dT/dt) * 10. Thermal
    inertia estimated from load-drop events (exponential decay tau).
    """

    def __init__(
        self,
        prediction_horizon_s: float = PREDICTION_HORIZON_S,
        default_thermal_inertia_s: float = DEFAULT_THERMAL_INERTIA_S,
        min_samples: int = 3,
        registry: Optional[object] = None,
        node_to_rack: Optional[Dict[str, str]] = None,
    ):
        self.prediction_horizon_s = prediction_horizon_s
        self.default_thermal_inertia_s = default_thermal_inertia_s
        self.min_samples = min_samples
        self._registry = registry
        self._node_to_rack = node_to_rack

    def _get_registry(self) -> Optional[object]:
        if self._registry is not None:
            return self._registry
        try:
            from core.models.topology import get_default_registry
            return get_default_registry()
        except Exception:
            return None

    def _get_node_to_rack(self) -> Dict[str, str]:
        if self._node_to_rack is not None:
            return self._node_to_rack
        try:
            from core.config import get_node_to_rack_map
            return get_node_to_rack_map()
        except Exception:
            return {}

    def predict(self, nodes: List[BaseNode]) -> Optional[PredictionResult]:
        """Predict temperature at T+10 seconds from power/temp trajectory.

        When rack topology and FOPDT calibration exist: uses First-Order Plus
        Dead Time model with self-calibrated tau, theta, gain per rack.

        Otherwise: linear extrapolation T_pred = T_now + temp_slope * 10.

        Args:
            nodes: List of BaseNode with thermal_input, power_draw, timestamp.

        Returns:
            PredictionResult or None if insufficient data.
        """
        if not nodes or len(nodes) < self.min_samples:
            return None

        # Ingest telemetry for self-calibration
        self._ingest_telemetry(nodes)

        thermal = np.array([n.thermal_input for n in nodes])
        power = np.array([n.power_draw for n in nodes])
        timestamps = [getattr(n, "timestamp", None) for n in nodes]

        # Build time axis (seconds from first sample)
        if all(t is not None for t in timestamps):
            try:
                t0 = timestamps[0]
                time_s = np.array([(t - t0).total_seconds() for t in timestamps])
            except Exception:
                time_s = np.arange(len(nodes), dtype=float)  # Assume 1s per sample
        else:
            time_s = np.arange(len(nodes), dtype=float)

        if len(time_s) < 2 or time_s[-1] - time_s[0] < 1e-6:
            return None

        # Linear regression: temp = a*t + b, power = c*t + d
        n_use = min(10, len(nodes) - 1)
        t_win = time_s[-n_use:]
        T_win = thermal[-n_use:]
        P_win = power[-n_use:]

        dt = t_win[-1] - t_win[0]
        if dt < 1e-6:
            return None
        temp_slope = (T_win[-1] - T_win[0]) / dt
        power_slope = (P_win[-1] - P_win[0]) / dt

        current_temp = float(thermal[-1])
        power_last = float(power[-1]) if len(power) > 0 else 0.0

        # Try FOPDT prediction when rack topology and calibration exist
        predicted_t10, thermal_inertia, fopdt_used = self._predict_fopdt_if_available(
            nodes, current_temp, power_last, time_s, t_win, thermal
        )

        if not fopdt_used:
            predicted_t10 = current_temp + temp_slope * self.prediction_horizon_s
            thermal_inertia = self._estimate_thermal_inertia(thermal, power, time_s)

        confidence = min(1.0, 0.5 + 0.1 * len(nodes) + 0.1 * n_use)
        if fopdt_used:
            confidence = min(1.0, confidence + 0.15)  # Slightly higher for FOPDT

        return PredictionResult(
            predicted_temp=current_temp,
            predicted_temp_t10=predicted_t10,
            power_trajectory_slope=power_slope,
            temp_trajectory_slope=temp_slope,
            thermal_inertia_s=thermal_inertia,
            confidence=confidence,
        )

    def _ingest_telemetry(self, nodes: List[BaseNode]) -> None:
        """Add node telemetry to RackRegistry for self-calibration."""
        registry = self._get_registry()
        if registry is None:
            return
        node_to_rack = self._get_node_to_rack()
        try:
            from core.config import resolve_node_to_rack
        except Exception:
            return

        for n in nodes:
            node_id = getattr(n, "node_id", "")
            rack_id = node_to_rack.get(node_id) or resolve_node_to_rack(node_id, None)
            if not rack_id:
                continue
            ts = getattr(n, "timestamp", None)
            if ts is None:
                continue
            t_sec = ts.timestamp() if hasattr(ts, "timestamp") else 0.0
            registry.add_telemetry(
                rack_id=rack_id,
                timestamp=t_sec,
                temperature=n.thermal_input,
                cooling_rpm=n.cooling_output,
                power_w=n.power_draw,
            )
            buf = getattr(registry, "_telemetry_buffer", {}).get(rack_id, [])
            if len(buf) > 0 and len(buf) % CALIBRATE_EVERY_N_SAMPLES == 0:
                registry.calibrate_rack(rack_id)

    def _predict_fopdt_if_available(
        self,
        nodes: List[BaseNode],
        current_temp: float,
        power_last: float,
        time_s: np.ndarray,
        t_win: np.ndarray,
        thermal: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """Use FOPDT when racks have calibration. Returns (predicted_t10, thermal_inertia, used)."""
        registry = self._get_registry()
        if registry is None:
            return current_temp, self.default_thermal_inertia_s, False

        node_to_rack = self._get_node_to_rack()
        try:
            from core.config import resolve_node_to_rack
        except Exception:
            return current_temp, self.default_thermal_inertia_s, False

        # Group nodes by rack; use last reading per rack
        rack_to_node: Dict[str, BaseNode] = {}
        for n in nodes:
            node_id = getattr(n, "node_id", "")
            rack_id = node_to_rack.get(node_id) or resolve_node_to_rack(node_id, None)
            if rack_id:
                rack_to_node[rack_id] = n  # Last wins

        if not rack_to_node:
            return current_temp, self.default_thermal_inertia_s, False

        timestamps = [getattr(n, "timestamp", None) for n in nodes]
        if not timestamps or timestamps[-1] is None:
            return current_temp, self.default_thermal_inertia_s, False
        t_now = timestamps[-1].timestamp() if hasattr(timestamps[-1], "timestamp") else time_s[-1]
        t_pred = t_now + self.prediction_horizon_s

        predictions = []
        tau_list = []

        for rack_id, n in rack_to_node.items():
            rack = registry.get_rack(rack_id)
            if rack.fopdt.n_samples < 30:  # Not calibrated
                continue
            cooling_history = registry.get_cooling_history(rack_id)
            if not cooling_history:
                continue
            T_pred = registry.predict_fopdt(
                rack_id=rack_id,
                t_now=t_now,
                t_pred=t_pred,
                T_current=n.thermal_input,
                cooling_history=cooling_history,
                power_w=n.power_draw,
            )
            predictions.append(T_pred)
            tau_list.append(rack.tau)

        if not predictions:
            return current_temp, self.default_thermal_inertia_s, False

        # Worst-case T (max) for proactive cooling
        predicted_t10 = float(np.max(predictions))
        thermal_inertia = float(np.median(tau_list)) if tau_list else self.default_thermal_inertia_s
        return predicted_t10, thermal_inertia, True

    def _estimate_thermal_inertia(
        self,
        thermal: np.ndarray,
        power: np.ndarray,
        time_s: np.ndarray,
    ) -> float:
        """
        Estimate thermal inertia: how long hardware holds heat after load drops.
        Uses exponential decay model: dT/dt ~ -T/tau when power drops.
        """
        if len(power) < 4:
            return self.default_thermal_inertia_s

        # Find load drop events: power decreased significantly
        power_diff = np.diff(power)
        drops = np.where(power_diff < -10)[0]  # Drop > 10W
        if len(drops) == 0:
            return self.default_thermal_inertia_s

        # After a drop, how fast does temp fall?
        tau_estimates = []
        for i in drops:
            if i + 3 >= len(thermal):
                continue
            T_after = thermal[i + 1 : i + 4]
            t_after = time_s[i + 1 : i + 4]
            if T_after[0] <= T_after[-1]:
                continue  # Temp didn't drop
            # Rough tau: T(t) = T0 * exp(-t/tau) -> ln(T/T0) = -t/tau
            dt = t_after[-1] - t_after[0]
            if dt < 1e-6:
                continue
            dT = T_after[-1] - T_after[0]
            if T_after[0] < 1e-6:
                continue
            # tau ~ -dt / ln(T_final/T_initial)
            try:
                ratio = T_after[-1] / T_after[0]
                if ratio > 0.1 and ratio < 1.0:
                    tau = -dt / np.log(ratio)
                    tau_estimates.append(max(10, min(300, tau)))
            except Exception:
                pass

        if tau_estimates:
            return float(np.median(tau_estimates))
        return self.default_thermal_inertia_s


def compute_thermal_inertia_seconds(nodes: List[BaseNode]) -> float:
    """Compute thermal inertia: how long hardware holds heat after load drops.

    Uses FOPDT tau when rack calibration exists; otherwise estimates from
    load-drop events. Prevents over-cooling during idle—if load just dropped,
    temp decay is slow; brief over-cooling is acceptable.

    Args:
        nodes: List of BaseNode with thermal and power history.

    Returns:
        Thermal inertia in seconds (default 120 if insufficient data).
    """
    predictor = TempPredictor()
    result = predictor.predict(nodes)
    return result.thermal_inertia_s if result else DEFAULT_THERMAL_INERTIA_S
