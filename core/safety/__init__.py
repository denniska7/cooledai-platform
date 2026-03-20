"""Safety helpers: telemetry health, actuator mismatch hints, saturation flags."""

from core.safety.telemetry_failure_policy import (
    FailurePosture,
    apply_failure_adjustments,
    assess_fan_slippage,
    assess_history_staleness,
    assess_temp_rise_at_max_cooling,
)

__all__ = [
    "FailurePosture",
    "apply_failure_adjustments",
    "assess_fan_slippage",
    "assess_history_staleness",
    "assess_temp_rise_at_max_cooling",
]
