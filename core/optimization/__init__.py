"""
CooledAI Optimization Engine

Central brain for thermal optimization. Takes normalized BaseNode data
and calculates the Efficiency Gap: thermal lag, over-provisioning, oscillation.

Usage:
    from optimization import OptimizationBrain
    brain = OptimizationBrain()
    gap = brain.analyze(nodes)

Note: Imports are lazy to allow lightweight modules (gpu_power_governor,
thermal_calibrator) to be imported on agent nodes that lack pandas/scipy.
"""


def __getattr__(name: str):
    """Lazy imports — only load heavy modules when actually accessed."""
    _brain_names = {
        "OptimizationBrain", "EfficiencyGap", "apply_guardrails",
        "MAX_SAFE_TEMP", "MIN_FAN_RPM",
    }
    _confidence_names = {
        "estimate_confidence_interval_quantile", "ConfidenceIntervalResult",
        "CAUTIONARY_CONFIDENCE_THRESHOLD", "CAUTIONARY_BASELINE_DELTA",
    }
    _optimizer_names = {
        "PowerCostOptimizer", "power_cost_of_cooling", "fan_power_at_rpm",
        "PowerCostResult", "OptimizationResult",
    }

    if name in _brain_names:
        from core.optimization.optimization_brain import (
            OptimizationBrain,
            EfficiencyGap,
            apply_guardrails,
            MAX_SAFE_TEMP,
            MIN_FAN_RPM,
        )
        return locals()[name]

    if name in _confidence_names:
        from core.optimization.confidence_estimator import (
            estimate_confidence_interval_quantile,
            ConfidenceIntervalResult,
            CAUTIONARY_CONFIDENCE_THRESHOLD,
            CAUTIONARY_BASELINE_DELTA,
        )
        return locals()[name]

    if name in _optimizer_names:
        from core.optimization.optimizer import (
            PowerCostOptimizer,
            power_cost_of_cooling,
            fan_power_at_rpm,
            PowerCostResult,
            OptimizationResult,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OptimizationBrain",
    "EfficiencyGap",
    "apply_guardrails",
    "MAX_SAFE_TEMP",
    "MIN_FAN_RPM",
    "estimate_confidence_interval_quantile",
    "ConfidenceIntervalResult",
    "CAUTIONARY_CONFIDENCE_THRESHOLD",
    "CAUTIONARY_BASELINE_DELTA",
    "PowerCostOptimizer",
    "power_cost_of_cooling",
    "fan_power_at_rpm",
    "PowerCostResult",
    "OptimizationResult",
]
