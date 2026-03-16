"""
CooledAI Cooling Anomaly Detection - Failing component detection.

If a cooling unit's power draw increases while its ΔT (temperature drop) decreases,
flag it as a 'Failing Component' in the diagnostic reasoning. This supports
Predictive Maintenance as a higher-value use case.

Sanity Check (Sensor Ghost): Temperature readings must pass the Sanity Check
filter before being used here. If a temperature jumps >20°C in a single second
(or rate > 20°C/s), it is flagged as 'Sensor Ghost' and ignored—see
core.anomaly.sanity_check. The telemetry ingestion layer applies this filter
and Moving Average (last 5 readings) to smooth telemetry before it hits the
OptimizationBrain, preventing fan jitter from electrical noise.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

# Minimum ΔT (°C) to consider trend meaningful; avoid noise when delta_T is near zero
MIN_DELTA_T_C = 0.5
# Relative change thresholds: power up by at least this ratio, delta_T down by at least this ratio
POWER_INCREASE_RATIO = 1.05   # 5% increase
DELTA_T_DECREASE_RATIO = 0.95  # 5% decrease


@dataclass
class CoolingUnitState:
    """Snapshot of a cooling unit's state for trend-based anomaly detection.

    Attributes:
        power_draw: Electrical power consumption in Watts.
        delta_t_c: Temperature drop (return - supply) in °C across the unit.
        node_id: Unique identifier for the cooling unit.
    """
    power_draw: float
    delta_t_c: float
    node_id: str = ""


@dataclass
class FailingComponentFinding:
    """A detected failing-component anomaly for predictive maintenance.

    Indicates a cooling unit where power draw increased while ΔT decreased,
    a signature of fouling, compressor degradation, or reduced heat transfer.

    Attributes:
        node_id: Identifier of the affected cooling unit.
        message: Human-readable diagnostic message.
        power_draw_prev: Power draw before the anomaly (W).
        power_draw_now: Power draw at detection (W).
        delta_t_prev: Temperature drop before anomaly (°C).
        delta_t_now: Temperature drop at detection (°C).
    """
    node_id: str
    message: str
    power_draw_prev: float
    power_draw_now: float
    delta_t_prev: float
    delta_t_now: float


def get_cooling_unit_delta_t(node: Any) -> Optional[float]:
    """Compute ΔT (temperature drop) for a cooling unit.

    Thermodynamics: ΔT = T_return - T_supply measures heat transfer effectiveness.
    A healthy unit has high ΔT (heat removed from air); a failing unit shows
    lower ΔT for the same power (fouling, degraded heat transfer).

    Args:
        node: A BaseNode or CRAC-like object with thermal readings.

    Returns:
        Temperature drop in °C, or None if not a cooling unit or data missing.
    """
    # CRAC_Unit: return_temp (or thermal_input) - supply_temp
    if hasattr(node, "supply_temp") and getattr(node, "supply_temp", None) is not None:
        return_t = getattr(node, "return_temp", None) or getattr(node, "thermal_input", None)
        if return_t is not None:
            return float(return_t) - float(node.supply_temp)
    # Generic node: raw_data may have supply_temp, return_temp or thermal_input as return
    raw = getattr(node, "raw_data", None) or {}
    supply = raw.get("supply_temp")
    return_t = raw.get("return_temp") or raw.get("thermal_input") or getattr(node, "thermal_input", None)
    if supply is not None and return_t is not None:
        try:
            return float(return_t) - float(supply)
        except (TypeError, ValueError):
            pass
    return None


def _is_cooling_unit(node: Any) -> bool:
    """True if node has cooling-unit semantics (CRAC, chiller, etc.)."""
    if type(node).__name__ == "CRAC_Unit":
        return True
    raw = getattr(node, "raw_data", None) or {}
    if raw.get("node_type") in ("crac", "CRAC", "cooling_unit", "chiller"):
        return True
    # Must have computable ΔT and meaningful power (exclude passive sensors)
    delta_t = get_cooling_unit_delta_t(node)
    if delta_t is not None and (getattr(node, "power_draw", 0) or 0) > 0:
        return True
    return False


def detect_failing_cooling_units(
    nodes: List[Any],
    previous_state: Optional[Dict[str, CoolingUnitState]] = None,
    power_increase_ratio: float = POWER_INCREASE_RATIO,
    delta_t_decrease_ratio: float = DELTA_T_DECREASE_RATIO,
    min_delta_t_c: float = MIN_DELTA_T_C,
) -> Tuple[List[FailingComponentFinding], Dict[str, CoolingUnitState]]:
    """Detect cooling units exhibiting failing-component signature.

    Physics logic: A healthy cooling unit increases power to increase ΔT.
    A failing unit (fouling, compressor degradation) draws more power but
    delivers less ΔT—heat transfer efficiency declines. We flag when:
    - Power increased by at least power_increase_ratio (e.g. 5%)
    - ΔT decreased by at least delta_t_decrease_ratio (e.g. 5%)
    - Previous ΔT was above min_delta_t_c to avoid noise at near-zero ΔT.

    Args:
        nodes: List of cooling nodes (CRAC, chiller, etc.) with power and thermal data.
        previous_state: Optional state from prior call for trend comparison.
        power_increase_ratio: Minimum ratio for "power increased" (default 1.05).
        delta_t_decrease_ratio: Maximum ratio for "ΔT decreased" (default 0.95).
        min_delta_t_c: Minimum ΔT (°C) to consider trend meaningful (default 0.5).

    Returns:
        Tuple of (list of FailingComponentFinding, updated state dict for next call).
    """
    previous_state = previous_state or {}
    new_state: Dict[str, CoolingUnitState] = {}
    findings: List[FailingComponentFinding] = []

    for node in nodes:
        node_id = getattr(node, "node_id", None) or getattr(node, "node_type", "") or "unknown"
        if not node_id or node_id == "unknown":
            node_id = id(node)  # fallback
        delta_t = get_cooling_unit_delta_t(node)
        if delta_t is None or not _is_cooling_unit(node):
            continue
        power = float(getattr(node, "power_draw", 0) or 0)
        delta_t_c = float(delta_t)
        new_state[node_id] = CoolingUnitState(power_draw=power, delta_t_c=delta_t_c, node_id=str(node_id))

        prev = previous_state.get(node_id)
        if prev is None:
            continue
        if prev.delta_t_c < min_delta_t_c:
            continue
        # Failing signature: power went up, ΔT went down
        power_increased = power >= prev.power_draw * power_increase_ratio
        delta_t_decreased = delta_t_c <= prev.delta_t_c * delta_t_decrease_ratio and delta_t_c < prev.delta_t_c
        if power_increased and delta_t_decreased:
            display_id = node_id if isinstance(node_id, str) else str(node_id)
            findings.append(
                FailingComponentFinding(
                    node_id=display_id,
                    message=(
                        f"Failing Component: {display_id} — power draw increased ({prev.power_draw:.0f} W → {power:.0f} W) "
                        f"while ΔT decreased ({prev.delta_t_c:.1f}°C → {delta_t_c:.1f}°C). "
                        "Consider predictive maintenance (e.g. fouling, compressor degradation)."
                    ),
                    power_draw_prev=prev.power_draw,
                    power_draw_now=power,
                    delta_t_prev=prev.delta_t_c,
                    delta_t_now=delta_t_c,
                )
            )

    return findings, new_state


def format_findings_for_diagnostic(findings: List[FailingComponentFinding]) -> str:
    """Turn findings into a single string for diagnostic_reasoning."""
    if not findings:
        return ""
    return " ".join(f.message for f in findings)


def severity_score_from_finding(f: FailingComponentFinding) -> float:
    """Compute a severity score (0–100) for maintenance prioritization.

    Physics logic: Combines power increase ratio and ΔT decrease ratio.
    Higher power increase + larger ΔT drop = more severe degradation.
    Score = 100 * (power_increase_ratio + delta_t_decrease_ratio) / 2,
    capped to [0, 100]. 10% power up + 10% ΔT down ≈ 10; 100%+100% = 100.

    Args:
        f: A FailingComponentFinding with power and ΔT before/after.

    Returns:
        Severity score in [0, 100], higher = more urgent for maintenance.
    """
    power_prev = f.power_draw_prev or 1e-6
    power_increase_ratio = (f.power_draw_now - f.power_draw_prev) / power_prev
    power_increase_ratio = max(0.0, min(2.0, power_increase_ratio))  # cap 0-200%
    delta_t_prev = f.delta_t_prev or 1e-6
    delta_t_decrease_ratio = (f.delta_t_prev - f.delta_t_now) / delta_t_prev
    delta_t_decrease_ratio = max(0.0, min(1.0, delta_t_decrease_ratio))  # cap 0-100%
    # Combined: 0-100 scale; 10% power up + 10% ΔT down ≈ 10, 100%+100% = 100
    score = 100.0 * (power_increase_ratio + delta_t_decrease_ratio) / 2.0
    return round(min(100.0, max(0.0, score)), 1)
