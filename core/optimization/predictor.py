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

# Max points in sliding window for trend fit (balances noise vs responsiveness)
TREND_WINDOW_MAX = 14

# Recent-sample half-life for weighted regression (seconds). Newer points weigh more.
TREND_WEIGHT_HALF_LIFE_S = 4.0

# Cap |dT/dt| used for short-horizon extrapolation (°C/s). Field logs showed ~0.5–1°C/s
# on worst ramps; 1.8 leaves headroom without letting single-sample spikes dominate.
MAX_EXTRAP_SLOPE_C_PER_S = 1.8


def _weighted_linear_trend(
    t: np.ndarray,
    y: np.ndarray,
    half_life_s: float = TREND_WEIGHT_HALF_LIFE_S,
) -> Tuple[float, float, float, float]:
    """Weighted least-squares line y ≈ slope*t + intercept.

    Weights decay exponentially toward older samples (robust for CSV / agent bursts).

    Returns:
        slope, intercept, r_squared in [0,1], weighted_rmse of residuals.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    if n < 2:
        return 0.0, float(y[-1]) if n else 0.0, 0.0, 0.0

    span = max(float(t[-1] - t[0]), 1e-9)
    hl = max(0.35, min(half_life_s, span * 0.5))
    w = np.exp(np.log(0.5) * (t[-1] - t) / hl)
    w_sum = np.sum(w)
    if w_sum < 1e-18:
        w = np.ones(n) / n
        w_sum = 1.0
    w = w * (n / w_sum)

    wt = np.sum(w)
    tw = np.sum(w * t)
    yw = np.sum(w * y)
    wtt = np.sum(w * t * t)
    wty = np.sum(w * t * y)

    denom = wt * wtt - tw * tw
    if abs(denom) < 1e-18:
        return 0.0, float(y[-1]), 0.0, 0.0

    slope = (wt * wty - tw * yw) / denom
    intercept = (yw - slope * tw) / wt

    y_pred = slope * t + intercept
    resid = y - y_pred
    ss_res = float(np.sum(w * resid ** 2))
    y_mean = yw / wt
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = float(np.nan_to_num(r2, nan=0.0, posinf=1.0, neginf=0.0))
    r2 = max(0.0, min(1.0, r2))
    rmse = float(np.sqrt(ss_res / max(wt, 1e-18)))
    return float(slope), float(intercept), r2, rmse


def _confidence_from_trend_quality(
    n_samples: int,
    span_s: float,
    r2_thermal: float,
    r2_power: float,
    fopdt_used: bool,
) -> float:
    """Map fit quality + window shape to [0,1] confidence (calibrated, not arbitrary)."""
    span_factor = min(1.0, span_s / 9.0)
    n_factor = min(1.0, n_samples / 12.0)
    fit_quality = 0.5 * r2_thermal + 0.5 * r2_power
    conf = 0.12 + 0.40 * fit_quality + 0.28 * n_factor + 0.20 * span_factor
    if fopdt_used:
        conf += 0.08
    return float(min(1.0, max(0.15, conf)))


@dataclass
class PredictionResult:
    """Result of temperature prediction at T+10s.

    Attributes:
        predicted_temp: Current temperature (last sample; not modified by clamps).
        predicted_temp_t10: Predicted temperature at T+10 seconds.
        power_trajectory_slope: Power trend (W/s).
        temp_trajectory_slope: Temperature trend (°C/s) from weighted regression **without**
            ``MAX_EXTRAP_SLOPE_C_PER_S`` — reflects observed local ramp/spike in telemetry.
        extrapolation_temp_slope_c_per_s: Slope actually used for short-horizon linear
            extrapolation and FOPDT residual guard — **clamped** to limit forecast blow-up.
        thermal_inertia_s: Estimated thermal time constant (seconds).
        confidence: Prediction confidence in [0, 1].
    """
    predicted_temp: float
    predicted_temp_t10: float  # T+10 seconds
    power_trajectory_slope: float  # W/s
    temp_trajectory_slope: float   # °C/s (observed / reported trend)
    extrapolation_temp_slope_c_per_s: float  # °C/s (used for T+10 linear path)
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

        Otherwise: linear extrapolation uses a **clamped** slope for the forecast only;
        ``temp_trajectory_slope`` in the result stays the **unclamped** regression slope
        so fast legitimate sensor transitions remain visible in metrics.

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

        # Weighted regression on recent window (noise-resistant vs two-point slope)
        n_use = min(TREND_WINDOW_MAX, len(nodes))
        t_win = time_s[-n_use:]
        T_win = thermal[-n_use:]
        P_win = power[-n_use:]

        dt = float(t_win[-1] - t_win[0])
        if dt < 1e-6:
            return None

        temp_slope_raw, _, r2_t, _ = _weighted_linear_trend(t_win, T_win)
        power_slope, _, r2_p, _ = _weighted_linear_trend(t_win, P_win)

        # Clamp **only** the slope used for short-horizon forecast — keep raw for telemetry truth.
        slope_extrap = float(
            np.clip(
                temp_slope_raw,
                -MAX_EXTRAP_SLOPE_C_PER_S,
                MAX_EXTRAP_SLOPE_C_PER_S,
            )
        )

        current_temp = float(thermal[-1])
        power_last = float(power[-1]) if len(power) > 0 else 0.0

        # Try FOPDT prediction when rack topology and calibration exist
        predicted_t10, thermal_inertia, fopdt_used = self._predict_fopdt_if_available(
            nodes, current_temp, power_last, time_s, t_win, thermal
        )

        if not fopdt_used:
            raw_t10 = current_temp + slope_extrap * self.prediction_horizon_s
            max_delta = max(0.5, MAX_EXTRAP_SLOPE_C_PER_S * self.prediction_horizon_s)
            predicted_t10 = float(np.clip(raw_t10, current_temp - max_delta, current_temp + max_delta))
            thermal_inertia = self._estimate_thermal_inertia(thermal, power, time_s)
        else:
            # Soft guard: FOPDT shouldn't diverge absurdly from local trend
            max_delta = max(2.0, MAX_EXTRAP_SLOPE_C_PER_S * self.prediction_horizon_s * 2.0)
            predicted_t10 = float(
                np.clip(predicted_t10, current_temp - max_delta, current_temp + max_delta)
            )

        confidence = _confidence_from_trend_quality(
            n_samples=n_use,
            span_s=dt,
            r2_thermal=r2_t,
            r2_power=r2_p,
            fopdt_used=fopdt_used,
        )

        return PredictionResult(
            predicted_temp=current_temp,
            predicted_temp_t10=predicted_t10,
            power_trajectory_slope=power_slope,
            temp_trajectory_slope=float(temp_slope_raw),
            extrapolation_temp_slope_c_per_s=slope_extrap,
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

        calibrated_racks: set = set()
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
            if (
                len(buf) > 0
                and len(buf) % CALIBRATE_EVERY_N_SAMPLES == 0
                and rack_id not in calibrated_racks
            ):
                registry.calibrate_rack(rack_id)
                calibrated_racks.add(rack_id)

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
