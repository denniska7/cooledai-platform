"""
CooledAI Anomaly Detection - Predictive maintenance layer (core).

Detects failing cooling components: e.g. when a cooling unit's power draw increases
while its ΔT (temperature drop) decreases, flag as 'Failing Component' for diagnostic
reasoning. Shifts value from Cooling Optimization to Predictive Maintenance.
"""

from core.anomaly.cooling_anomaly import (
    detect_failing_cooling_units,
    get_cooling_unit_delta_t,
    format_findings_for_diagnostic,
    severity_score_from_finding,
    CoolingUnitState,
    FailingComponentFinding,
)
from core.anomaly.sanity_check import (
    filter_sensor_ghost,
    SanityCheckFilter,
    SensorGhostFlag,
)

__all__ = [
    "detect_failing_cooling_units",
    "get_cooling_unit_delta_t",
    "format_findings_for_diagnostic",
    "severity_score_from_finding",
    "CoolingUnitState",
    "FailingComponentFinding",
    "filter_sensor_ghost",
    "SanityCheckFilter",
    "SensorGhostFlag",
]
