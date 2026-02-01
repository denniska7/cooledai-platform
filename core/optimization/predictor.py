"""
CooledAI Temp Predictor - Look-Ahead Predictive Modeling

Predicts temperature at T+10 seconds based on power trajectory.
Uses linear regression / trend analysis (LSTM-style can be added later).
Enables proactive cooling adjustments before spikes occur.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.hal.base_node import BaseNode

# Prediction horizon (seconds)
PREDICTION_HORIZON_S = 10.0

# Thermal inertia: typical time constant for temp decay after load drop (seconds)
# Higher = hardware holds heat longer = over-cool less during idle
DEFAULT_THERMAL_INERTIA_S = 120.0


@dataclass
class PredictionResult:
    """Result of temperature prediction."""
    predicted_temp: float
    predicted_temp_t10: float  # T+10 seconds
    power_trajectory_slope: float  # W/s
    temp_trajectory_slope: float   # °C/s
    thermal_inertia_s: float
    confidence: float  # 0-1


class TempPredictor:
    """
    Predicts temperature at T+10s based on power/temp trajectory.
    Uses linear regression on recent samples (mathematical trend analysis).
    """

    def __init__(
        self,
        prediction_horizon_s: float = PREDICTION_HORIZON_S,
        default_thermal_inertia_s: float = DEFAULT_THERMAL_INERTIA_S,
        min_samples: int = 3,
    ):
        self.prediction_horizon_s = prediction_horizon_s
        self.default_thermal_inertia_s = default_thermal_inertia_s
        self.min_samples = min_samples

    def predict(self, nodes: List[BaseNode]) -> Optional[PredictionResult]:
        """
        Predict temperature at T+10 seconds based on power/temp trajectory.
        Returns None if insufficient data.
        """
        if not nodes or len(nodes) < self.min_samples:
            return None

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
        # Use last N samples for trend
        n_use = min(10, len(nodes) - 1)
        t_win = time_s[-n_use:]
        T_win = thermal[-n_use:]
        P_win = power[-n_use:]

        # Slopes (°C/s, W/s)
        dt = t_win[-1] - t_win[0]
        if dt < 1e-6:
            return None
        temp_slope = (T_win[-1] - T_win[0]) / dt
        power_slope = (P_win[-1] - P_win[0]) / dt

        # Predict T+10s: T_pred = T_now + temp_slope * 10
        # If power is rising, temp slope may increase - simple model: T_pred = T_now + slope * 10
        current_temp = float(thermal[-1])
        predicted_t10 = current_temp + temp_slope * self.prediction_horizon_s

        # Thermal inertia: estimate from load drop events
        thermal_inertia = self._estimate_thermal_inertia(thermal, power, time_s)

        # Confidence: higher with more samples and consistent trend
        confidence = min(1.0, 0.5 + 0.1 * len(nodes) + 0.1 * n_use)

        return PredictionResult(
            predicted_temp=current_temp,
            predicted_temp_t10=predicted_t10,
            power_trajectory_slope=power_slope,
            temp_trajectory_slope=temp_slope,
            thermal_inertia_s=thermal_inertia,
            confidence=confidence,
        )

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
    """
    How long does this hardware hold heat after load drops?
    Used to prevent over-cooling during idle periods.
    """
    predictor = TempPredictor()
    result = predictor.predict(nodes)
    return result.thermal_inertia_s if result else DEFAULT_THERMAL_INERTIA_S
