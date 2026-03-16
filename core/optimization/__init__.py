"""
CooledAI Optimization Engine

Central brain for thermal optimization. Takes normalized BaseNode data
and calculates the Efficiency Gap: thermal lag, over-provisioning, oscillation.

Usage:
    from optimization import OptimizationBrain
    brain = OptimizationBrain()
    gap = brain.analyze(nodes)
"""

from core.optimization.optimization_brain import (
    OptimizationBrain,
    EfficiencyGap,
    apply_guardrails,
    MAX_SAFE_TEMP,
    MIN_FAN_RPM,
)
from core.optimization.confidence_estimator import (
    estimate_confidence_interval_quantile,
    ConfidenceIntervalResult,
    CAUTIONARY_CONFIDENCE_THRESHOLD,
    CAUTIONARY_BASELINE_DELTA,
)
from core.optimization.optimizer import (
    PowerCostOptimizer,
    power_cost_of_cooling,
    fan_power_at_rpm,
    PowerCostResult,
    OptimizationResult,
)

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
