"""
CooledAI Thermal Forensics - Physical Anomaly Detection

Detects physical anomalies that indicate equipment failure or sensor malfunction:
1. Runaway Detection: dT/dt > 2.0°C/min while power stable (fan failure, intake blockage)
2. Sensor Cross-Validation: 2-out-of-3 voting, ignore outlier >15°C (sensor malfunction)
3. Action: Thermal runaway → GUARD_MODE + cooling 100% (before MAX_SAFE_TEMP)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hal.base_node import BaseNode

from core.optimization.optimization_brain import MAX_SAFE_TEMP, EfficiencyGap

_logger = logging.getLogger("cooledai.thermal_forensics")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)

# Runaway threshold: °C per minute (power stable = anomaly)
RUNAWAY_THRESHOLD_C_PER_MIN = 2.0

# Power stability: max variation (W) or fraction to consider "stable"
POWER_STABILITY_THRESHOLD_W = 15.0  # ±15W
POWER_STABILITY_THRESHOLD_FRAC = 0.05  # ±5%

# Sensor cross-validation: outlier threshold (°C)
SENSOR_OUTLIER_THRESHOLD_C = 15.0

# Sensor drift: Core Avg vs Package Temp - if drift > this, Sensor Integrity Failure
SENSOR_DRIFT_THRESHOLD_C = 15.0

# Keys to search for multi-sensor temps (case-insensitive partial match)
MULTI_SENSOR_KEYS = ["tctl", "tdie", "die", "inlet", "cpu temp", "package temp", "core temp"]

# Keys for Core vs Package categorization (for drift detection)
PACKAGE_TEMP_PATTERNS = ["package", "tdie", "tctl", "cpu (tctl/tdie)", "cpu package"]
CORE_TEMP_PATTERNS = ["core avg", "core max", "core min", "core temp", "core 0", "core 1", "core 2"]


@dataclass
class ForensicsResult:
    """
    Result of thermal forensics analysis.
    """
    # Runaway detection
    runaway_detected: bool = False
    runaway_reason: str = ""
    dT_dt_per_min: float = 0.0

    # Sensor cross-validation
    sensor_malfunction: bool = False
    sensor_malfunction_reason: str = ""
    outlier_sensor: Optional[str] = None
    validated_temp: Optional[float] = None  # Consensus temp after voting

    # Sensor drift (Core Avg vs Package Temp)
    sensor_drift_detected: bool = False
    sensor_drift_reason: str = ""

    # Action: trigger GUARD_MODE and force 100% cooling
    trigger_guard_mode: bool = False
    force_cooling_100: bool = False

    # Alerts to log
    alerts: List[str] = field(default_factory=list)


def _categorize_sensor_type(key: str) -> Optional[str]:
    """Return 'package', 'core', or None for uncategorized."""
    k = str(key).lower()
    for p in PACKAGE_TEMP_PATTERNS:
        if p in k:
            return "package"
    for p in CORE_TEMP_PATTERNS:
        if p in k:
            return "core"
    return None


def _detect_sensor_drift(node: "BaseNode") -> Tuple[bool, str]:
    """
    Monitor delta between Core Avg and Package Temp.
    If they drift apart by >15°C unexpectedly, Sensor Integrity Failure.
    
    Returns: (detected, reason)
    """
    rd = getattr(node, "raw_data", None) or {}
    package_temps: List[Tuple[str, float]] = []
    core_temps: List[Tuple[str, float]] = []
    
    for k, v in rd.items():
        if not isinstance(v, (int, float)):
            continue
        try:
            fv = float(v)
            if not (0 < fv < 150):
                continue
        except (TypeError, ValueError):
            continue
        k_lower = str(k).lower()
        stype = _categorize_sensor_type(k)
        if stype == "package":
            package_temps.append((k, fv))
        elif stype == "core":
            core_temps.append((k, fv))
    
    # Also check thermal_input, cpu_temp as package-type
    t = getattr(node, "thermal_input", 0) or 0
    if t > 0:
        package_temps.append(("thermal_input", t))
    ct = getattr(node, "cpu_temp", None)
    if ct is not None and ct > 0:
        package_temps.append(("cpu_temp", ct))
    
    if not package_temps or not core_temps:
        return False, ""
    
    p_avg = sum(v for _, v in package_temps) / len(package_temps)
    c_avg = sum(v for _, v in core_temps) / len(core_temps)
    drift = abs(p_avg - c_avg)
    
    if drift > SENSOR_DRIFT_THRESHOLD_C:
        drift_reason = (
            f"Core Avg ({c_avg:.1f}°C) vs Package Temp ({p_avg:.1f}°C) "
            f"drift {drift:.1f}°C > {SENSOR_DRIFT_THRESHOLD_C}°C threshold. "
            "Sensor malfunction or calibration drift."
        )
        _logger.warning("Sensor Drift: %s", drift_reason)
        return True, drift_reason
    return False, ""


def _extract_multi_sensor_temps(node: "BaseNode") -> List[Tuple[str, float]]:
    """
    Extract up to 3 temperature readings from a node (Tctl, Die, Inlet, etc.).
    Uses thermal_input, cpu_temp, gpu_temp, and raw_data keys.
    """
    temps: List[Tuple[str, float]] = []
    seen: set = set()

    def _add(name: str, val: float) -> None:
        if val is not None and 0 < val < 150 and name not in seen:
            temps.append((name, float(val)))
            seen.add(name)

    # Primary thermal
    t = getattr(node, "thermal_input", 0) or 0
    if t > 0:
        _add("thermal_input", t)

    # ServerNode optional
    for attr in ("cpu_temp", "gpu_temp"):
        v = getattr(node, attr, None)
        if v is not None and v > 0:
            _add(attr, v)

    # raw_data: fuzzy match for Tctl, Tdie, Inlet, etc.
    rd = getattr(node, "raw_data", None) or {}
    for k, v in rd.items():
        if not isinstance(v, (int, float)):
            continue
        k_lower = str(k).lower()
        for pattern in MULTI_SENSOR_KEYS:
            if pattern in k_lower and k_lower not in seen:
                try:
                    fv = float(v)
                    if 0 < fv < 150:
                        temps.append((k, fv))
                        seen.add(k_lower)
                        break
                except (TypeError, ValueError):
                    pass

    return temps[:3]  # Max 3 for 2-out-of-3 voting


def _sensor_cross_validate(node: "BaseNode") -> Tuple[Optional[float], bool, Optional[str], str]:
    """
    2-out-of-3 voting: if one sensor disagrees by >15°C, ignore outlier.
    Returns: (validated_temp, sensor_malfunction, outlier_name, reason)
    """
    temps = _extract_multi_sensor_temps(node)
    if len(temps) < 2:
        # Single sensor - no cross-validation possible
        if temps:
            return temps[0][1], False, None, ""
        return None, False, None, ""

    # 2 sensors: if they disagree by >15°C, flag malfunction
    if len(temps) == 2:
        a, b = temps[0][1], temps[1][1]
        if abs(a - b) > SENSOR_OUTLIER_THRESHOLD_C:
            median = (a + b) / 2
            _logger.warning(
                "Sensor Malfunction: %s (%.1f°C) vs %s (%.1f°C) disagree by %.1f°C > %d°C. "
                "Using median %.1f°C.",
                temps[0][0], a, temps[1][0], b, abs(a - b), int(SENSOR_OUTLIER_THRESHOLD_C), median,
            )
            return median, True, None, (
                f"Sensor disagreement: {temps[0][0]} ({a:.1f}°C) vs {temps[1][0]} ({b:.1f}°C)"
            )
        return (a + b) / 2, False, None, ""

    # 3 sensors: 2-out-of-3 voting
    vals = [(n, v) for n, v in temps]
    sorted_vals = sorted(vals, key=lambda x: x[1])
    lo, mid, hi = sorted_vals[0][1], sorted_vals[1][1], sorted_vals[2][1]

    # Check if low is outlier
    if mid - lo > SENSOR_OUTLIER_THRESHOLD_C and hi - mid <= SENSOR_OUTLIER_THRESHOLD_C:
        outlier = sorted_vals[0]
        consensus = (mid + hi) / 2
        _logger.warning(
            "Sensor Malfunction: %s (%.1f°C) disagrees with consensus (%.1f°C) by >%d°C. Ignoring outlier.",
            outlier[0], outlier[1], consensus, int(SENSOR_OUTLIER_THRESHOLD_C),
        )
        return consensus, True, outlier[0], f"{outlier[0]} ({outlier[1]:.1f}°C) outlier"

    # Check if high is outlier
    if hi - mid > SENSOR_OUTLIER_THRESHOLD_C and mid - lo <= SENSOR_OUTLIER_THRESHOLD_C:
        outlier = sorted_vals[2]
        consensus = (lo + mid) / 2
        _logger.warning(
            "Sensor Malfunction: %s (%.1f°C) disagrees with consensus (%.1f°C) by >%d°C. Ignoring outlier.",
            outlier[0], outlier[1], consensus, int(SENSOR_OUTLIER_THRESHOLD_C),
        )
        return consensus, True, outlier[0], f"{outlier[0]} ({outlier[1]:.1f}°C) outlier"

    # Check if middle is outlier (disagrees with both)
    if mid - lo > SENSOR_OUTLIER_THRESHOLD_C and hi - mid > SENSOR_OUTLIER_THRESHOLD_C:
        outlier = sorted_vals[1]
        consensus = (lo + hi) / 2
        _logger.warning(
            "Sensor Malfunction: %s (%.1f°C) disagrees with consensus (%.1f°C) by >%d°C. Ignoring outlier.",
            outlier[0], outlier[1], consensus, int(SENSOR_OUTLIER_THRESHOLD_C),
        )
        return consensus, True, outlier[0], f"{outlier[0]} ({outlier[1]:.1f}°C) outlier"

    # No outlier - use median
    return mid, False, None, ""


def _detect_runaway(
    nodes: List["BaseNode"],
    runaway_threshold: float = RUNAWAY_THRESHOLD_C_PER_MIN,
    power_stability_w: float = POWER_STABILITY_THRESHOLD_W,
) -> Tuple[bool, str, float]:
    """
    Monitor dT/dt. If temp rising >2.0°C/min while power stable → Physical Anomaly.
    (Potential fan failure or intake blockage)

    Returns: (detected, reason, dT_dt_per_min)
    """
    if not nodes or len(nodes) < 2:
        return False, "", 0.0

    # Use last N samples for rate
    window = min(5, len(nodes) - 1)
    prev = nodes[-(window + 1)]
    curr = nodes[-1]

    t_prev = getattr(prev, "timestamp", None)
    t_curr = getattr(curr, "timestamp", None)
    if t_prev is None or t_curr is None:
        return False, "", 0.0

    try:
        dt_min = (t_curr - t_prev).total_seconds() / 60.0
    except Exception:
        return False, "", 0.0

    if dt_min <= 0:
        return False, "", 0.0

    dT = curr.thermal_input - prev.thermal_input
    dT_dt_per_min = dT / dt_min

    # Only flag if temp is *rising* (positive rate)
    if dT_dt_per_min < runaway_threshold:
        return False, "", dT_dt_per_min

    # Check power stability
    powers = [n.power_draw for n in nodes[-window - 1 :]]
    if len(powers) < 2:
        return False, "", dT_dt_per_min

    p_mean = sum(powers) / len(powers)
    p_var = max(abs(p - p_mean) for p in powers)
    power_stable = p_var <= power_stability_w or (p_mean > 0 and p_var / p_mean <= POWER_STABILITY_THRESHOLD_FRAC)

    if not power_stable:
        return False, "", dT_dt_per_min

    reason = (
        f"Thermal Runaway: dT/dt = {dT_dt_per_min:.1f}°C/min > {runaway_threshold}°C/min "
        f"while power stable ({p_mean:.0f}W ±{p_var:.0f}W). "
        "Potential fan failure or intake blockage."
    )
    _logger.warning("Physical Anomaly: %s", reason)
    return True, reason, dT_dt_per_min


def analyze_thermal_forensics(
    nodes: List["BaseNode"],
    runaway_threshold: float = RUNAWAY_THRESHOLD_C_PER_MIN,
) -> ForensicsResult:
    """
    Run thermal forensics: runaway detection + sensor cross-validation.
    If thermal runaway detected → trigger GUARD_MODE and force cooling 100%
    (before MAX_SAFE_TEMP is reached).
    """
    result = ForensicsResult()

    if not nodes:
        return result

    # 1. Sensor cross-validation (on latest node)
    latest = nodes[-1]
    validated_temp, sensor_malfunction, outlier, reason = _sensor_cross_validate(latest)
    if sensor_malfunction:
        result.sensor_malfunction = True
        result.sensor_malfunction_reason = reason
        result.outlier_sensor = outlier
        result.validated_temp = validated_temp
        result.alerts.append(f"Sensor Malfunction: {reason}")

    # 1b. Sensor drift detection (Core Avg vs Package Temp)
    drift_detected, drift_reason = _detect_sensor_drift(latest)
    if drift_detected:
        result.sensor_drift_detected = True
        result.sensor_drift_reason = drift_reason
        result.trigger_guard_mode = True
        result.alerts.append(f"Sensor Integrity Failure: {drift_reason}")
        _logger.warning("Sensor Integrity Failure. Triggering GUARD_MODE.")

    # 2. Runaway detection
    runaway, runaway_reason, dT_dt = _detect_runaway(nodes, runaway_threshold)
    result.dT_dt_per_min = dT_dt
    if runaway:
        result.runaway_detected = True
        result.runaway_reason = runaway_reason
        result.alerts.append(runaway_reason)

        # Action: GUARD_MODE + cooling 100% immediately (before MAX_SAFE_TEMP)
        result.trigger_guard_mode = True
        result.force_cooling_100 = True
        _logger.critical(
            "Thermal Runaway detected. Triggering GUARD_MODE and setting cooling to 100%% "
            "(preemptive, before MAX_SAFE_TEMP %.1f°C).",
            MAX_SAFE_TEMP,
        )

    return result


def apply_thermal_forensics_action(
    gap: EfficiencyGap,
    forensics_result: ForensicsResult,
) -> EfficiencyGap:
    """
    Apply thermal forensics action: if runaway detected, force cooling to 100%.
    Call after OptimizationBrain.analyze() and apply_guardrails().
    """
    if not forensics_result.force_cooling_100:
        return gap

    # Override AI: set 100% cooling (preemptive, before MAX_SAFE_TEMP)
    gap.recommended_cooling_delta = 1.0
    gap.emergency_mode = True
    gap.guardrail_triggered = True
    gap.guardrail_reason = (
        f"Thermal Runaway: {forensics_result.runaway_reason}. "
        "Preemptive 100% cooling (before MAX_SAFE_TEMP)."
    )
    gap.recommendations.insert(
        0,
        f"⚠️ THERMAL RUNAWAY: {forensics_result.runaway_reason} Cooling set to 100%.",
    )
    return gap
