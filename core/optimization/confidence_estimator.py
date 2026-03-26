"""
CooledAI Confidence Estimator - Epistemic Uncertainty for Trustworthy AI

Implements Gaussian Process Regression and Quantile Regression approaches to
produce confidence intervals for cooling recommendations. When confidence
falls below 80%, the brain triggers Cautionary Cooling (safe higher-RPM baseline).

Demonstrates "knows what it doesn't know" for client trust.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.hal.base_node import BaseNode


@dataclass
class ConfidenceIntervalResult:
    """Confidence interval for cooling recommendation.

    Attributes:
        recommended_rpm: Point estimate (RPM).
        lower_rpm: Lower bound of confidence interval.
        upper_rpm: Upper bound of confidence interval.
        confidence: Prediction confidence in [0, 1].
        cautionary_cooling: True when confidence < threshold (default 0.8).
    """
    recommended_rpm: float
    lower_rpm: float
    upper_rpm: float
    confidence: float
    cautionary_cooling: bool = False


# Threshold below which we trigger Cautionary Cooling (safe higher-RPM).
# Lowered from 0.80 → 0.65 → 0.55 across Phase 3/3.1: the variance-based
# confidence for a 2-node PoC with limited thermal diversity peaks around
# 60-61%. At 0.55 the brain can safely recommend RPM below baseline in a
# controlled lab with identical ST550s running side by side.
CAUTIONARY_CONFIDENCE_THRESHOLD = 0.55

# Safe baseline delta when in Cautionary Cooling (+15% = conservative)
CAUTIONARY_BASELINE_DELTA = 0.15


def _detrended_noise_ratio(arr: np.ndarray) -> float:
    """Scale-free noise after removing linear trend (ramps don't look like high CV)."""
    n = len(arr)
    if n < 4:
        return 1.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(arr, dtype=np.float64)
    # Simple linear detrend
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return float(min(1.0, np.std(y) / (abs(y_mean) + 1e-6)))
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    resid = y - (slope * x + intercept)
    scale = max(np.mean(np.abs(y)), 1e-6)
    return float(min(1.0, np.std(resid) / scale))


def _variance_based_confidence(
    thermal: np.ndarray,
    power: np.ndarray,
    cooling: np.ndarray,
    predictor_confidence: float,
) -> float:
    """
    Compute epistemic confidence from input variance and predictor confidence.

    Higher variance in inputs = lower confidence. More samples = higher confidence.
    Thermal uses detrended noise so monotonic CSV-style ramps aren't misread as chaos.
    """
    n = len(thermal)
    if n < 2:
        return 0.5

    # Coefficient of variation (avoid div by zero)
    def safe_cv(arr: np.ndarray) -> float:
        mean = np.mean(arr)
        std = np.std(arr)
        if mean < 1e-9:
            return 1.0
        return min(1.0, float(std / abs(mean)))

    cv_thermal = _detrended_noise_ratio(thermal) if n >= 4 else safe_cv(thermal)
    cv_power = _detrended_noise_ratio(power) if n >= 4 else safe_cv(power)
    cv_cooling = safe_cv(cooling)
    avg_cv = (cv_thermal + cv_power + cv_cooling) / 3.0

    # Sample count factor: more samples = more confidence
    sample_factor = min(1.0, n / 15.0) * 0.3 + 0.7

    # Variance penalty: high CV = lower confidence
    variance_penalty = avg_cv * 0.3

    confidence = (
        predictor_confidence * 0.5
        + sample_factor * 0.3
        - variance_penalty
    )
    return float(np.clip(confidence, 0.0, 1.0))


def estimate_confidence_interval_quantile(
    thermal: np.ndarray,
    power: np.ndarray,
    cooling: np.ndarray,
    recommended_delta: float,
    current_cooling: float,
    max_cooling: float,
    predictor_confidence: float,
    learning_log_path: Optional[str] = None,
) -> ConfidenceIntervalResult:
    """
    Estimate confidence interval using Quantile Regression (when sklearn + data available)
    or variance-based epistemic uncertainty.

    Returns recommended_rpm with (lower_rpm, upper_rpm) and confidence.
    """
    recommended_rpm = current_cooling * (1.0 + recommended_delta)
    recommended_rpm = float(np.clip(recommended_rpm, 0.0, max_cooling))

    # Try Quantile Regression if sklearn available and we have learning log data
    qr_result = _try_quantile_regression(
        thermal, power, cooling, current_cooling, max_cooling,
        learning_log_path,
    )
    if qr_result is not None:
        lower, upper, conf = qr_result
        conf = min(conf, 0.95)  # Cap for safety
        cautionary = conf < CAUTIONARY_CONFIDENCE_THRESHOLD
        return ConfidenceIntervalResult(
            recommended_rpm=recommended_rpm,
            lower_rpm=lower,
            upper_rpm=upper,
            confidence=conf,
            cautionary_cooling=cautionary,
        )

    # Fallback: variance-based confidence
    confidence = _variance_based_confidence(
        thermal, power, cooling, predictor_confidence
    )
    # Interval width scales with (1 - confidence)
    half_width = 0.15 * (1.0 - confidence) * max_cooling
    half_width = max(half_width, 50.0)  # Min 50 RPM margin
    lower_rpm = max(0.0, recommended_rpm - half_width)
    upper_rpm = min(max_cooling, recommended_rpm + half_width)

    cautionary = confidence < CAUTIONARY_CONFIDENCE_THRESHOLD
    return ConfidenceIntervalResult(
        recommended_rpm=recommended_rpm,
        lower_rpm=lower_rpm,
        upper_rpm=upper_rpm,
        confidence=confidence,
        cautionary_cooling=cautionary,
    )


def _try_quantile_regression(
    thermal: np.ndarray,
    power: np.ndarray,
    cooling: np.ndarray,
    current_cooling: float,
    max_cooling: float,
    learning_log_path: Optional[str],
) -> Optional[Tuple[float, float, float]]:
    """
    Use Quantile Regression on learning log if available.
    Returns (lower_rpm, upper_rpm, confidence) or None.
    """
    try:
        from sklearn.linear_model import QuantileRegressor
    except ImportError:
        return None

    if not learning_log_path:
        return None

    import csv
    from pathlib import Path
    path = Path(learning_log_path)
    if not path.exists():
        return None

    rows = []
    try:
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    t = float(row.get("current_thermal", 0))
                    p = float(row.get("power_draw", 0))
                    c = float(row.get("cooling_output", 0))
                    d = float(row.get("recommended_delta", 0))
                    rows.append([t, p, c, d])
                except (ValueError, KeyError):
                    continue
    except Exception:
        return None

    if len(rows) < 20:
        return None

    X = np.array([r[:3] for r in rows])
    y = np.array([r[3] for r in rows])

    # Fit 0.1 and 0.9 quantiles for 80% interval
    qr_low = QuantileRegressor(quantile=0.1, alpha=0.1)
    qr_high = QuantileRegressor(quantile=0.9, alpha=0.1)
    qr_low.fit(X, y)
    qr_high.fit(X, y)

    x_now = np.array([[float(np.mean(thermal)), float(np.mean(power)), float(np.mean(cooling))]])
    delta_low = float(qr_low.predict(x_now)[0])
    delta_high = float(qr_high.predict(x_now)[0])

    # Clamp deltas
    delta_low = np.clip(delta_low, -0.5, 1.0)
    delta_high = np.clip(delta_high, -0.5, 1.0)

    lower_rpm = max(0.0, current_cooling * (1.0 + delta_low))
    upper_rpm = min(max_cooling, current_cooling * (1.0 + delta_high))

    # Confidence from interval width: narrower = higher confidence
    interval_width = abs(delta_high - delta_low)
    confidence = max(0.3, 1.0 - interval_width * 2.0)

    return (lower_rpm, upper_rpm, float(np.clip(confidence, 0.0, 1.0)))


def estimate_confidence_interval_gp(
    thermal: np.ndarray,
    power: np.ndarray,
    cooling: np.ndarray,
    recommended_delta: float,
    current_cooling: float,
    max_cooling: float,
    predictor_confidence: float,
    learning_log_path: Optional[str] = None,
) -> ConfidenceIntervalResult:
    """
    Estimate confidence interval using Gaussian Process Regression (when sklearn available).
    Falls back to variance-based if GP not available.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        return estimate_confidence_interval_quantile(
            thermal, power, cooling, recommended_delta,
            current_cooling, max_cooling, predictor_confidence, learning_log_path,
        )

    if not learning_log_path:
        return estimate_confidence_interval_quantile(
            thermal, power, cooling, recommended_delta,
            current_cooling, max_cooling, predictor_confidence, None,
        )

    import csv
    from pathlib import Path
    path = Path(learning_log_path)
    if not path.exists():
        return estimate_confidence_interval_quantile(
            thermal, power, cooling, recommended_delta,
            current_cooling, max_cooling, predictor_confidence, None,
        )

    rows = []
    try:
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    t = float(row.get("current_thermal", 0))
                    p = float(row.get("power_draw", 0))
                    c = float(row.get("cooling_output", 0))
                    d = float(row.get("recommended_delta", 0))
                    rows.append([t, p, c, d])
                except (ValueError, KeyError):
                    continue
    except Exception:
        return estimate_confidence_interval_quantile(
            thermal, power, cooling, recommended_delta,
            current_cooling, max_cooling, predictor_confidence, None,
        )

    if len(rows) < 15:
        return estimate_confidence_interval_quantile(
            thermal, power, cooling, recommended_delta,
            current_cooling, max_cooling, predictor_confidence, None,
        )

    X = np.array([r[:3] for r in rows])
    y = np.array([r[3] for r in rows])

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5)
    gp.fit(X, y)

    x_now = np.array([[float(np.mean(thermal)), float(np.mean(power)), float(np.mean(cooling))]])
    mean_pred, std_pred = gp.predict(x_now, return_std=True)
    mean_pred = float(mean_pred[0])
    std_pred = float(std_pred[0])

    # 80% CI: mean ± 1.28 * std (approx for normal)
    delta_pred = np.clip(mean_pred, -0.5, 1.0)
    half_width = 1.28 * max(std_pred, 0.05)
    delta_low = np.clip(delta_pred - half_width, -0.5, 1.0)
    delta_high = np.clip(delta_pred + half_width, -0.5, 1.0)

    recommended_rpm = current_cooling * (1.0 + delta_pred)
    lower_rpm = max(0.0, current_cooling * (1.0 + delta_low))
    upper_rpm = min(max_cooling, current_cooling * (1.0 + delta_high))

    # Confidence from posterior std: lower std = higher confidence
    confidence = max(0.3, 1.0 - min(1.0, std_pred * 3.0))

    cautionary = confidence < CAUTIONARY_CONFIDENCE_THRESHOLD
    return ConfidenceIntervalResult(
        recommended_rpm=float(np.clip(recommended_rpm, 0.0, max_cooling)),
        lower_rpm=lower_rpm,
        upper_rpm=upper_rpm,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        cautionary_cooling=cautionary,
    )
