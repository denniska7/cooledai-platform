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

__all__ = [
    "OptimizationBrain",
    "EfficiencyGap",
    "apply_guardrails",
    "MAX_SAFE_TEMP",
    "MIN_FAN_RPM",
]
