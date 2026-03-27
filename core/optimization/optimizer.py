"""
CooledAI Power-Cost Optimizer - Fan Affinity Laws & Zone Power Minimization

Uses the Fan Affinity Laws (P₂ = P₁ · (N₂/N₁)³) to calculate the projected
energy cost of every cooling decision. The optimizer solves for the lowest
total power across the zone, not just the lowest temperature.

This ensures the 40% savings target by keeping fans in their Efficiency
Sweet Spot (typically 60–80% of max RPM—best cfm/W, minimal turbulence).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from core.optimization.cooling_safety_policy import (
    ACTIVE_COMPUTE_POWER_THRESHOLD_W,
    POWER_SLOPE_PRE_RAMP_MODERATE_W_S,
    POWER_SLOPE_PRE_RAMP_STEEP_W_S,
    POWER_SLOPE_STEEP_MIN_DELTA,
    SAFETY_NET_MIN_DELTA,
    SAFETY_NET_POWER_SLOPE_W_S,
)

_log = logging.getLogger("cooledai.optimizer")

# Fan Affinity Law: P₂ = P₁ · (N₂/N₁)³
# Power scales with the cube of fan speed.

# Efficiency Sweet Spot: 60–80% of max RPM
# Below 50%: efficiency drops (dead zones, turbulence, poor flow)
# Above 85%: diminishing returns, more wear, cubic power cost
SWEET_SPOT_MIN = 0.60
SWEET_SPOT_MAX = 0.80
SWEET_SPOT_TARGET = 0.70  # Aim for 70% when stable

# Default rated power per CRAC at 100% (Watts) - typical 10–15 kW
DEFAULT_RATED_POWER_W = 12_000.0


@dataclass
class PowerCostResult:
    """Result of power-cost computation for a cooling decision.

    Attributes:
        total_cooling_power_w: Total fan power (W) for proposed state.
        power_delta_w: Change vs baseline (negative = savings).
        power_savings_ratio: Fraction of baseline power saved (0–1).
        in_sweet_spot: True if all units in efficiency sweet spot.
        sweet_spot_penalty: Penalty for being outside sweet spot (0–1).
    """
    total_cooling_power_w: float
    power_delta_w: float
    power_savings_ratio: float
    in_sweet_spot: bool
    sweet_spot_penalty: float
    per_unit_power: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of zone power minimization.

    Attributes:
        recommended_delta: Cooling delta (-1 to 1) for lowest zone power.
        recommended_delta_by_unit: Per-unit deltas when multi-unit.
        total_cooling_power_w: Projected total fan power at recommended.
        power_savings_vs_current_w: Watts saved vs current state.
        in_sweet_spot: True if recommended state is in sweet spot.
        thermal_safe: True if temp constraint satisfied.
        reasoning: Human-readable explanation.
    """
    recommended_delta: float
    recommended_delta_by_unit: Dict[str, float] = field(default_factory=dict)
    total_cooling_power_w: float = 0.0
    power_savings_vs_current_w: float = 0.0
    in_sweet_spot: bool = False
    thermal_safe: bool = True
    reasoning: str = ""


def fan_power_at_rpm(
    cooling_rpm: float,
    rated_rpm: float,
    rated_power_w: float,
) -> float:
    """
    Fan power at given RPM using affinity law: P = P_rated · (N/N_rated)³.

    Does NOT clamp above rated_rpm: if VFD over-speeds the fan or max_rpm is
    mis-configured, the cubic law still applies and the real power cost MUST
    be reflected.  Clamping to 1.0 hid over-speed conditions and made the
    savings calculation optimistic.

    Args:
        cooling_rpm: Actual fan speed (RPM or normalized).
        rated_rpm: Reference speed (e.g. max RPM).
        rated_power_w: Power at rated_rpm (Watts).

    Returns:
        Power in Watts.  May exceed rated_power_w if cooling_rpm > rated_rpm.
    """
    if rated_rpm < 1e-9:
        return 0.0
    ratio = max(0.0, cooling_rpm / rated_rpm)
    if ratio > 1.0:
        _log.warning(
            "Fan over-speed detected: %.0f RPM exceeds rated %.0f RPM "
            "(ratio %.2f). Power = %.0f W (above rated %.0f W).",
            cooling_rpm, rated_rpm, ratio,
            rated_power_w * ratio ** 3, rated_power_w,
        )
    return rated_power_w * (ratio ** 3)


def power_cost_of_cooling(
    current_cooling_by_unit: Dict[str, float],
    proposed_cooling_by_unit: Dict[str, float],
    max_cooling_by_unit: Dict[str, float],
    rated_power_by_unit: Optional[Dict[str, float]] = None,
) -> PowerCostResult:
    """
    Compute total zone cooling power for proposed vs current state.

    Uses Fan Affinity Laws. Returns power delta and sweet-spot metrics.

    Args:
        current_cooling_by_unit: unit_id -> current RPM (or 0).
        proposed_cooling_by_unit: unit_id -> proposed RPM.
        max_cooling_by_unit: unit_id -> max RPM.
        rated_power_by_unit: unit_id -> power at 100% (W). Default 12 kW.

    Returns:
        PowerCostResult with total power, delta, savings ratio, sweet-spot status.
    """
    rated_power_by_unit = rated_power_by_unit or {}
    default_power = DEFAULT_RATED_POWER_W

    current_total = 0.0
    proposed_total = 0.0
    per_unit = {}

    all_units = set(current_cooling_by_unit) | set(proposed_cooling_by_unit)

    for unit_id in all_units:
        max_rpm = max_cooling_by_unit.get(unit_id) or 3000.0
        if max_rpm < 1e-9:
            max_rpm = 3000.0
        rated_w = rated_power_by_unit.get(unit_id, default_power)

        current_rpm = current_cooling_by_unit.get(unit_id, 0.0)
        proposed_rpm = proposed_cooling_by_unit.get(unit_id, current_rpm)

        p_current = fan_power_at_rpm(current_rpm, max_rpm, rated_w)
        p_proposed = fan_power_at_rpm(proposed_rpm, max_rpm, rated_w)

        current_total += p_current
        proposed_total += p_proposed
        per_unit[unit_id] = p_proposed

    power_delta = proposed_total - current_total
    power_savings_ratio = (
        (current_total - proposed_total) / current_total
        if current_total > 1e-6 else 0.0
    )
    # Allow negative ratio (cost increase) — clamping to [0,1] hid cases
    # where proposed cooling was MORE expensive than current.
    power_savings_ratio = max(-1.0, min(1.0, power_savings_ratio))

    # Sweet spot: each unit in 60–80% of max
    in_sweet_spot = True
    penalty_sum = 0.0
    n_units = len(all_units) or 1
    for unit_id in all_units:
        proposed_rpm = proposed_cooling_by_unit.get(unit_id, 0.0)
        max_rpm = max_cooling_by_unit.get(unit_id) or 3000.0
        if max_rpm < 1e-9:
            continue
        frac = proposed_rpm / max_rpm
        if frac < 1e-9:
            in_sweet_spot = False
            penalty_sum += 0.5  # Off is "out of sweet spot" (idle)
        elif frac < SWEET_SPOT_MIN or frac > SWEET_SPOT_MAX:
            in_sweet_spot = False
            # Penalty: distance from sweet spot
            if frac < SWEET_SPOT_MIN:
                penalty_sum += (SWEET_SPOT_MIN - frac) / SWEET_SPOT_MIN
            else:
                penalty_sum += min(1.0, (frac - SWEET_SPOT_MAX) / (1.0 - SWEET_SPOT_MAX))
    sweet_spot_penalty = min(1.0, penalty_sum / n_units)

    return PowerCostResult(
        total_cooling_power_w=proposed_total,
        power_delta_w=power_delta,
        power_savings_ratio=power_savings_ratio,
        in_sweet_spot=in_sweet_spot,
        sweet_spot_penalty=sweet_spot_penalty,
        per_unit_power=per_unit,
    )


def in_sweet_spot(cooling_rpm: float, max_rpm: float) -> bool:
    """True if cooling is in efficiency sweet spot (60–80% of max)."""
    if max_rpm < 1e-9:
        return False
    frac = cooling_rpm / max_rpm
    return SWEET_SPOT_MIN <= frac <= SWEET_SPOT_MAX


def sweet_spot_rpm(max_rpm: float, target_frac: float = SWEET_SPOT_TARGET) -> float:
    """Target RPM for efficiency sweet spot."""
    return max_rpm * target_frac


class PowerCostOptimizer:
    """
    Optimizer that solves for lowest total cooling power across the zone.

    Uses Fan Affinity Laws to project energy cost. Biases toward Efficiency
    Sweet Spot (60–80% of max) to hit 40% savings target.

    When a CalibrationProfile is provided, slew rate limiting uses
    asymmetric rates (faster up, slower down) and the minimum response
    quantum rounds up small changes rather than suppressing them.
    """

    def __init__(
        self,
        sweet_spot_min: float = SWEET_SPOT_MIN,
        sweet_spot_max: float = SWEET_SPOT_MAX,
        default_rated_power_w: float = DEFAULT_RATED_POWER_W,
        calibration_profile: object = None,
    ):
        self.sweet_spot_min = sweet_spot_min
        self.sweet_spot_max = sweet_spot_max
        self.default_rated_power_w = default_rated_power_w
        self.calibration_profile = calibration_profile

    def optimize_for_min_power(
        self,
        current_thermal: float,
        target_temp: float,
        max_safe_temp: float,
        current_cooling: float,
        max_cooling: float,
        over_provisioning: float,
        temp_rising: bool,
        oscillation: bool,
        current_cooling_by_unit: Optional[Dict[str, float]] = None,
        max_cooling_by_unit: Optional[Dict[str, float]] = None,
        cooling_unit_ids: Optional[List[str]] = None,
        predicted_temp_t10: Optional[float] = None,
        power_slope_w_per_s: float = 0.0,
        sustained_active_compute: bool = False,
        current_power_w: float = 0.0,
        spike_bypass: bool = False,
    ) -> OptimizationResult:
        """
        Find cooling delta that minimizes zone power while keeping temp safe.

        Logic:
        - If over-provisioned and temp OK: reduce toward sweet spot (70%).
        - If temp rising: increase minimum needed to stay safe.
        - If stable: nudge toward sweet spot if not already there.
        - Oscillation: hold (delta=0).
        - Thermal adequacy guard: if current_thermal is already within 5°C
          of max_safe_temp, refuse to reduce cooling (even if predictor
          says temp is stable). This prevents the optimizer from chasing
          min-power when there is no thermal headroom.

        Args:
            current_thermal: Current temperature (°C).
            target_temp: Target temperature (°C).
            max_safe_temp: Maximum safe temperature (°C).
            current_cooling: Current aggregate cooling (RPM).
            max_cooling: Maximum cooling capacity (RPM).
            over_provisioning: Over-provisioning ratio (0–1).
            temp_rising: True if predicted temp > target.
            oscillation: True if oscillation detected.
            current_cooling_by_unit: Per-unit current RPM (optional).
            max_cooling_by_unit: Per-unit max RPM (optional).
            cooling_unit_ids: List of unit IDs (optional).
            predicted_temp_t10: Predicted temp at T+10s (optional, for adequacy).

        Returns:
            OptimizationResult with recommended_delta and power metrics.
        """
        if oscillation:
            return OptimizationResult(
                recommended_delta=0.0,
                total_cooling_power_w=self._zone_power(current_cooling, max_cooling),
                power_savings_vs_current_w=0.0,
                in_sweet_spot=in_sweet_spot(current_cooling, max_cooling),
                thermal_safe=current_thermal < max_safe_temp,
                reasoning="Holding cooling: oscillation detected (power-aware optimizer).",
            )

        # Power-slope pre-ramp: rising package power ⇒ treat as warming even if T+10 is still calm.
        # Phase 6.2: Added margin check — only pre-ramp when close to target temp.
        # Without this, ANY inference workload (>8 W/s) blocks all fan reductions
        # even with 25°C of thermal headroom.
        _opt_margin = target_temp - current_thermal
        if (
            power_slope_w_per_s >= POWER_SLOPE_PRE_RAMP_MODERATE_W_S
            and _opt_margin <= 5.0
        ):
            temp_rising = True

        # Thermal-adequacy guard: refuse to reduce cooling when headroom is thin.
        # If current temp is within 5°C of max_safe, or predicted T+10 exceeds
        # target+5, treat as "temp_rising" regardless of predictor output.
        THERMAL_HEADROOM_C = 5.0
        if current_thermal >= max_safe_temp - THERMAL_HEADROOM_C:
            temp_rising = True
            _log.info(
                "Thermal-adequacy guard: T_current=%.1f°C within %.0f°C of "
                "max_safe=%.1f°C — forcing temp_rising=True.",
                current_thermal, THERMAL_HEADROOM_C, max_safe_temp,
            )
        if predicted_temp_t10 is not None and predicted_temp_t10 > target_temp + THERMAL_HEADROOM_C:
            temp_rising = True

        # Build per-unit maps if we have aggregate only
        if current_cooling_by_unit is None or max_cooling_by_unit is None:
            current_cooling_by_unit = {"default": current_cooling}
            max_cooling_by_unit = {"default": max_cooling}
            cooling_unit_ids = ["default"]

        cooling_unit_ids = cooling_unit_ids or list(current_cooling_by_unit.keys())

        # Candidate deltas to evaluate
        candidates: List[Tuple[float, float, str]] = []

        # 1. Current state
        p_current = self._zone_power_from_units(
            current_cooling_by_unit, max_cooling_by_unit
        )
        candidates.append((0.0, p_current, "Hold current"))

        # 2. Sweet spot (70% of max) - for over-provisioned or stable
        if not temp_rising:
            proposed_sweet = {
                uid: (max_cooling_by_unit.get(uid, max_cooling) or max_cooling)
                * SWEET_SPOT_TARGET
                for uid in cooling_unit_ids
            }
            p_sweet = self._zone_power_from_units(
                proposed_sweet, max_cooling_by_unit
            )
            delta_sweet = self._delta_to_sweet_spot(
                current_cooling_by_unit, max_cooling_by_unit
            )
            candidates.append((delta_sweet, p_sweet, "Sweet spot (70%)"))

        # 3. Reduce (for over-provisioned) - e.g. -10%
        if over_provisioning > 0.15 and not temp_rising:
            delta_reduce = -0.10
            proposed = {
                uid: max(0, current_cooling_by_unit.get(uid, 0) * (1 + delta_reduce))
                for uid in cooling_unit_ids
            }
            p_reduce = self._zone_power_from_units(proposed, max_cooling_by_unit)
            candidates.append((delta_reduce, p_reduce, "Reduce 10%"))

        # 3b. Thermal-margin proportional reduction: when temp is well below
        # target (>10°C margin), propose a deeper reduction proportional to
        # the headroom.  At 30°C margin → target ~30% of max; at 10°C → ~60%.
        # This lets the optimizer match or beat the agent's local fallback
        # curve instead of being stuck at the "sweet spot" ceiling.
        thermal_margin = target_temp - current_thermal
        if thermal_margin > 5 and not temp_rising:
            margin_frac = min(1.0, thermal_margin / 30.0)  # 0..1
            target_frac = max(0.25, SWEET_SPOT_TARGET - margin_frac * 0.45)
            proposed_margin = {
                uid: max(0, max_cooling_by_unit.get(uid, max_cooling) * target_frac)
                for uid in cooling_unit_ids
            }
            agg_current = sum(current_cooling_by_unit.values()) or 1e-6
            agg_proposed = sum(proposed_margin.values())
            delta_margin = (agg_proposed / agg_current) - 1.0
            if delta_margin < -0.05:  # Only if it's a meaningful reduction
                p_margin = self._zone_power_from_units(proposed_margin, max_cooling_by_unit)
                candidates.append((
                    delta_margin, p_margin,
                    f"Thermal margin ({thermal_margin:.0f}°C headroom, target {target_frac:.0%} of max)",
                ))

        # 4. Increase (for temp rising) - minimum to stay safe
        if temp_rising:
            delta_increase = 0.12
            proposed = {
                uid: min(
                    max_cooling_by_unit.get(uid, max_cooling),
                    current_cooling_by_unit.get(uid, 0) * (1 + delta_increase),
                )
                for uid in cooling_unit_ids
            }
            p_increase = self._zone_power_from_units(proposed, max_cooling_by_unit)
            candidates.append((delta_increase, p_increase, "Increase 12% (temp rising)"))

        # Pick lowest power that satisfies thermal constraint
        if temp_rising:
            # Must increase - pick smallest increase
            increase_candidates = [(d, p, r) for d, p, r in candidates if d > 0]
            if increase_candidates:
                best = min(increase_candidates, key=lambda x: (x[1], -x[0]))
            else:
                best = (0.12, p_current, "Increase (fallback)")
        else:
            # Prefer lower power
            best = min(candidates, key=lambda x: x[1])

        delta, power_w, reason = best

        _log.warning(
            "[OPTIMIZER_DEBUG] temp_rising=%s thermal=%.1f target=%.1f margin=%.1f "
            "power_slope=%.2f n_candidates=%d best=(%s, %.4f) delta=%.4f",
            temp_rising, current_thermal, target_temp, target_temp - current_thermal,
            power_slope_w_per_s, len(candidates), reason, power_w, delta,
        )

        savings = p_current - power_w

        # Steep power ramp: enforce a stronger minimum positive delta (proactive cooling).
        if power_slope_w_per_s >= POWER_SLOPE_PRE_RAMP_STEEP_W_S:
            delta = max(delta, POWER_SLOPE_STEEP_MIN_DELTA)
            reason = f"{reason} + steep power-slope pre-ramp"

        # Safety net: under sustained / active compute with rising power, never return a silent hold/reduce.
        active_trigger = ACTIVE_COMPUTE_POWER_THRESHOLD_W
        if self.calibration_profile is not None:
            active_trigger = getattr(self.calibration_profile, "active_compute_trigger_w", active_trigger)
        if (
            (sustained_active_compute or current_power_w >= active_trigger)
            and power_slope_w_per_s > SAFETY_NET_POWER_SLOPE_W_S
            and delta < SAFETY_NET_MIN_DELTA
        ):
            delta = max(delta, SAFETY_NET_MIN_DELTA)
            reason = f"{reason} + active-load safety-net (rising power)"

        # Asymmetric slew rate limiting (calibration profile)
        # Skip slew cap on upward moves when spike_bypass — fans need to jump immediately
        cp = self.calibration_profile
        if spike_bypass and cp is not None:
            target_rpm = current_cooling * (1.0 + delta) if current_cooling > 1e-6 else max_cooling * delta
            _log.info(
                "[OPTIMIZER] spike_bypass_applied: target_rpm=%.0f delta=%.4f "
                "slew_cap_skipped=True",
                target_rpm, delta,
            )
        if cp is not None and max_cooling > 1e-6 and not spike_bypass:
            slew_up = getattr(cp, "slew_rate_up_rpm_per_cycle", 0)
            slew_down = getattr(cp, "slew_rate_down_rpm_per_cycle", 0)
            min_quantum = getattr(cp, "min_response_quantum_rpm", 0)

            if slew_up > 0 or slew_down > 0:
                delta_rpm = delta * current_cooling if current_cooling > 1e-6 else delta * max_cooling
                if delta_rpm > 0 and slew_up > 0:
                    delta_rpm = min(delta_rpm, slew_up)
                elif delta_rpm < 0 and slew_down > 0:
                    delta_rpm = max(delta_rpm, -slew_down)
                if current_cooling > 1e-6:
                    delta = delta_rpm / current_cooling
                else:
                    delta = delta_rpm / max_cooling

            # Minimum response quantum: round up small changes rather than suppressing
            if min_quantum > 0 and current_cooling > 1e-6:
                delta_rpm_abs = abs(delta * current_cooling)
                if 0 < delta_rpm_abs < min_quantum:
                    sign = 1.0 if delta >= 0 else -1.0
                    delta = sign * min_quantum / current_cooling
                    reason = f"{reason} + min_response_quantum rounded up"

        # Recompute zone power after post-selection delta adjustments
        proposed_cooling = min(
            max_cooling,
            max(0.0, current_cooling * (1.0 + delta)),
        )
        power_w = self._zone_power(proposed_cooling, max_cooling)
        savings = p_current - power_w

        sweet = in_sweet_spot(proposed_cooling, max_cooling)

        return OptimizationResult(
            recommended_delta=delta,
            total_cooling_power_w=power_w,
            power_savings_vs_current_w=max(0, savings),
            in_sweet_spot=sweet,
            thermal_safe=current_thermal < max_safe_temp,
            reasoning=f"Power-optimized: {reason} (zone power {power_w/1000:.1f} kW).",
        )

    def _zone_power(self, cooling: float, max_cooling: float) -> float:
        """Aggregate zone power for single cooling value."""
        return fan_power_at_rpm(
            cooling, max_cooling or 3000.0, self.default_rated_power_w
        )

    def _zone_power_from_units(
        self,
        cooling_by_unit: Dict[str, float],
        max_by_unit: Dict[str, float],
    ) -> float:
        """Total zone power from per-unit cooling."""
        total = 0.0
        for uid, rpm in cooling_by_unit.items():
            max_rpm = max_by_unit.get(uid, 3000.0) or 3000.0
            total += fan_power_at_rpm(rpm, max_rpm, self.default_rated_power_w)
        return total

    def _delta_to_sweet_spot(
        self,
        current_by_unit: Dict[str, float],
        max_by_unit: Dict[str, float],
    ) -> float:
        """Average delta to move all units to sweet spot (70%)."""
        if not current_by_unit or not max_by_unit:
            return 0.0
        deltas = []
        for uid in current_by_unit:
            c = current_by_unit.get(uid, 0)
            m = max_by_unit.get(uid, 3000.0) or 3000.0
            if m < 1e-9:
                continue
            target = m * SWEET_SPOT_TARGET
            if c < 1e-9:
                deltas.append(0.5)  # Turn on to 50%? Use 0.5 as proxy
            else:
                deltas.append((target - c) / c)
        return float(np.mean(deltas)) if deltas else 0.0

    def compute_kwh_saved(
        self,
        power_savings_w: float,
        hours: float = 1.0,
    ) -> float:
        """Convert power savings (W) to kWh saved."""
        return (power_savings_w / 1000.0) * hours
