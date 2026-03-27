"""
CooledAI Optimization Brain - Advanced Predictive Optimization

Takes normalized BaseNode data and calculates the Efficiency Gap:
1. Thermal Lag - Reaction time between power spike and cooling response
2. Over-provisioning - Cooling more than necessary (wasted energy)
3. Oscillation - Inefficient fan/pump hunting (rapid up/down cycling)

Enhancements:
- Multi-variate: CPU Utilization vs Power vs Ambient Inlet Temp
- Thermal Inertia: How long hardware holds heat after load drop (prevent over-cooling)
- Predictive: T+10s temp prediction based on power trajectory
- Reward Function: Energy_Saved*0.4 + Thermal_Stability*0.4 - Mechanical_Wear*0.2
- Dynamic Slew Rate: Gentle when far from limit, aggressive when approaching spike
- Learning Log: Predicted vs Actual for tune_model()
"""

import numpy as np
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from core.hal.base_node import BaseNode
from core.optimization.predictor import TempPredictor, PredictionResult, compute_thermal_inertia_seconds
from core.ingestion.influence_map import InfluenceMap
from core.optimization.reward import (
    compute_reward,
    thermal_stability_from_temp,
    mechanical_wear_penalty,
    RewardResult,
)
from core.optimization.learning_log import log_prediction, tune_model, DEFAULT_LEARNING_LOG
from core.optimization.optimizer import PowerCostOptimizer
from core.optimization.confidence_estimator import (
    estimate_confidence_interval_quantile,
    ConfidenceIntervalResult,
    CAUTIONARY_CONFIDENCE_THRESHOLD,
    CAUTIONARY_BASELINE_DELTA,
)
from core.optimization.cooling_safety_policy import (
    sustained_active_compute_signal,
    parse_policy_context,
    merge_policy_floors,
    policy_capacity_rpm,
    POWER_SLOPE_PRE_RAMP_MODERATE_W_S,
    POWER_SLOPE_PRE_RAMP_STEEP_W_S,
    SUSTAINED_COMPUTE_MEAN_THRESHOLD_W,
)
from core.optimization.spike_hold_state import record_spike_if_hot, spike_hold_floor_rpm

try:
    from core.optimization.thermal_calibrator import CalibrationProfile
except ImportError:
    CalibrationProfile = None  # type: ignore[misc, assignment]

try:
    from backend.safety.guardrails import snap_rpm_to_safe_boundary
except ImportError:
    def snap_rpm_to_safe_boundary(rpm: float, resonance_zones=None):  # noqa: D103
        return rpm  # no-op if backend not available

try:
    from backend.safety.equipment_safety import EquipmentSafety
except ImportError:
    EquipmentSafety = None  # type: ignore[misc, assignment]

# --- ASHRAE Safety Constants (Data Center Thermal Guidelines) ---
MAX_SAFE_TEMP = 80.0   # Celsius - ASHRAE Class A2/A3 inlet limit
MIN_FAN_RPM = 400      # Minimum fan speed - prevents thermal runaway when under load

# Load threshold (Watts) - "under load" = CPU is doing work
LOAD_THRESHOLD_W = 10.0

# --- Mechanical Longevity Constants (CRAC/Chiller Protection) ---
MIN_OFF_TIME = 180     # seconds - min time unit must stay off before restarting
MIN_RUN_TIME = 300     # seconds - min time unit must run before shutting off
MAX_COOLING_CHANGE_RATE = 0.1   # 10% per cycle - slew rate limit (thermal shock prevention)
STAGGER_START_DELAY = 2        # seconds between unit starts (inrush current protection)

# Configure safety event logging
_logger = logging.getLogger("cooledai.guardrails")
_precool_logger = logging.getLogger("cooledai.precool")
_reasoning_logger = logging.getLogger("cooledai.optimization_brain")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - CRITICAL - %(message)s"
    ))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.WARNING)


@dataclass
class EfficiencyGap:
    """
    Result of optimization analysis.
    
    Contains identified inefficiencies and recommended actions.
    """
    # Thermal Lag (seconds) - delay between power spike and cooling response
    thermal_lag_seconds: float = 0.0
    
    # Over-provisioning ratio (0-1) - how much excess cooling
    over_provisioning_ratio: float = 0.0
    
    # Oscillation ratio (0-1) - fraction of time in hunting behavior
    oscillation_ratio: float = 0.0
    
    # Overall efficiency score (0-100, higher = better)
    efficiency_score: float = 100.0
    
    # Recommended cooling output adjustment (-1 to 1, negative = reduce)
    recommended_cooling_delta: float = 0.0
    
    # Guardrail overrides
    emergency_mode: bool = False       # True when guardrail triggered - use 100% cooling
    guardrail_triggered: bool = False   # True if any guardrail intercepted AI
    guardrail_reason: str = ""          # Reason for override (for logging)

    # Mechanical longevity overrides
    anti_short_cycle_hold: bool = False   # True when MIN_OFF/MIN_RUN forces hold
    slew_rate_limited: bool = False       # True when delta was clamped for thermal shock
    hysteresis_hold: bool = False         # True when delta < 5% (chatter prevention)
    spike_bypass: bool = False            # True when spike bypass jumped past slew rate
    stagger_delays: List[float] = field(default_factory=list)  # Per-unit start delays (s)

    # Predictive & reward
    predicted_temp_t10: Optional[float] = None
    thermal_inertia_s: float = 120.0
    reward: Optional[float] = None
    reward_result: Optional[Any] = None

    # Human-readable recommendations
    recommendations: List[str] = field(default_factory=list)

    # Predictive diagnostic: explain reasoning using thermal momentum and influence map
    diagnostic_reasoning: str = ""

    # Reasoning Log: ordered "why" for each action (Glass Box audit)
    reasoning_log: List[str] = field(default_factory=list)
    reasoning_string: str = ""  # Final concatenated string for logging/API

    # LLM prompt context: rate of change + influence map for explainability
    llm_prompt_context: dict = field(default_factory=dict)

    # Energy/financial: estimated kWh saved (per hour) from this recommendation
    kwh_saved: float = 0.0

    # Power-cost (Fan Affinity Laws): cooling power and savings
    cooling_power_w: float = 0.0
    power_savings_w: float = 0.0
    in_sweet_spot: bool = False

    # Per-unit cooling deltas when failing components exist (unit_id -> delta); control layer applies these
    recommended_cooling_delta_by_unit: Dict[str, float] = field(default_factory=dict)

    # Confidence interval: AI "knows what it doesn't know" (trust)
    recommended_rpm_confidence_interval: Optional[tuple] = None  # (lower_rpm, upper_rpm)
    prediction_confidence: float = 1.0   # 0-1; below 0.8 triggers Cautionary Cooling
    cautionary_cooling_state: bool = False   # True when confidence < 80%: safe higher-RPM baseline

    # Raw metrics for debugging
    raw_metrics: dict = field(default_factory=dict)


def apply_guardrails(
    gap: EfficiencyGap,
    current_thermal: float,
    current_cooling: float,
    power_draw: float,
    max_cooling: float = 3000.0,
    *,
    sustained_active_compute: bool = False,
    spike_hold_min_rpm: float = 0.0,
    rated_max_fan_rpm: Optional[float] = None,
    calibration_profile: Optional[Any] = None,
) -> EfficiencyGap:
    """Apply ASHRAE-based safety guardrails to AI recommendations.

    Physics logic: Prevents thermal runaway and equipment damage. Overrides
    when: (1) temperature ≥ MAX_SAFE_TEMP (80°C) or approaching limit while
    AI suggests reducing cooling; (2) AI would set cooling below MIN_FAN_RPM
    (400) while system is under load—temp would rise with no cooling. Also
    applies mechanical resonance guard (snap RPM out of vibration zones).
    When triggered: emergency_mode=True, recommended_cooling_delta=1.0 (100%).

    Args:
        gap: EfficiencyGap from AI analysis.
        current_thermal: Current temperature (°C).
        current_cooling: Current cooling output (RPM or flow rate).
        power_draw: Current power draw (W); used to detect "under load".
        max_cooling: Maximum cooling capacity (for 100% = Emergency Mode).
        rated_max_fan_rpm: Chassis rated max RPM for policy floor caps (see merge_policy_floors).
        calibration_profile: Optional CalibrationProfile for auto-calibrated floor values.

    Returns:
        EfficiencyGap with guardrails applied (may be modified).
    """
    # Compute what the AI's recommendation would result in
    # recommended_cooling_delta: -0.1 = reduce 10%, 0 = no change, 1.0 = max
    proposed_cooling = current_cooling * (1.0 + gap.recommended_cooling_delta)
    
    # Mechanical Resonance Guard: snap RPM if in danger zone (vibration protection)
    proposed_safe = snap_rpm_to_safe_boundary(proposed_cooling)
    if abs(proposed_safe - proposed_cooling) > 0.5 and current_cooling > 0:
        gap.recommended_cooling_delta = (proposed_safe / current_cooling) - 1.0
        msg = f"Resonance Guard: Snapped RPM from {proposed_cooling:.0f} to {proposed_safe:.0f} (avoid vibration zone)."
        gap.recommendations.append(msg)
        gap.reasoning_log.append(msg)
        proposed_cooling = proposed_safe

    # Active compute + post-spike hold: soft RPM floor (not emergency 100%).
    # Prevents efficiency optimizer from locking fans near idle under GPU load.
    # rated_max_fan_rpm: chassis max so floors don't collapse when telemetry max == stuck-low RPM.
    soft_floor_rpm = merge_policy_floors(
        max_cooling,
        rated_max_fan_rpm=rated_max_fan_rpm,
        power_draw_w=power_draw,
        sustained_active=sustained_active_compute,
        spike_hold_min_rpm=spike_hold_min_rpm,
        calibration_profile=calibration_profile,
    )
    gap.raw_metrics["policy_soft_floor_rpm_target"] = soft_floor_rpm
    if soft_floor_rpm > 0 and current_cooling > 1e-6 and proposed_cooling + 1e-6 < soft_floor_rpm:
        new_delta = (soft_floor_rpm / current_cooling) - 1.0
        gap.recommended_cooling_delta = max(gap.recommended_cooling_delta, new_delta)
        proposed_cooling = min(max_cooling, current_cooling * (1.0 + gap.recommended_cooling_delta))
        msg = (
            f"Policy fan floor: raised cooling toward minimum {soft_floor_rpm:.0f} RPM "
            f"(active compute / thermal spike hold)."
        )
        gap.reasoning_log.append(msg)
        gap.recommendations.append(msg)
    
    guardrail_hit = False
    reason = ""
    
    # Guardrail 1: Temperature would exceed ASHRAE limit
    # If current temp is at/above limit, or AI suggests reducing cooling when hot
    if current_thermal >= MAX_SAFE_TEMP:
        guardrail_hit = True
        reason = (
            f"Temperature {current_thermal:.1f}°C >= MAX_SAFE_TEMP ({MAX_SAFE_TEMP}°C). "
            "Override: Emergency Mode (100% cooling)."
        )
    elif current_thermal >= MAX_SAFE_TEMP - 3.0 and gap.recommended_cooling_delta < 0:
        # Approaching limit and AI wants to reduce cooling - dangerous
        guardrail_hit = True
        reason = (
            f"Temperature {current_thermal:.1f}°C approaching limit. "
            "AI suggested reducing cooling - override: Emergency Mode (100% cooling)."
        )
    
    # Guardrail 2: AI would turn fan off (or below MIN_FAN_RPM) while under load
    if not guardrail_hit and power_draw > LOAD_THRESHOLD_W:
        if proposed_cooling < MIN_FAN_RPM:
            guardrail_hit = True
            reason = (
                f"AI would set cooling to {proposed_cooling:.0f} RPM (< MIN_FAN_RPM {MIN_FAN_RPM}) "
                f"while CPU under load ({power_draw:.0f}W). Override: Emergency Mode (100% cooling)."
            )
    
    if guardrail_hit:
        # Log Critical Safety Event
        _logger.critical(
            "Critical Safety Event: %s",
            reason,
            extra={"event": "guardrail_triggered", "reason": reason},
        )
        # Reasoning: why we overrode
        gap.reasoning_log.append(
            f"EMERGENCY: Setting all units to 100% RPM because {reason}"
        )
        # Override AI: set to 100% cooling (Emergency Mode)
        gap.emergency_mode = True
        gap.guardrail_triggered = True
        gap.guardrail_reason = reason
        gap.recommended_cooling_delta = 1.0  # 100% = max cooling
        gap.recommendations.insert(
            0,
            f"⚠️ GUARDRAIL: {reason} Cooling set to 100% (Emergency Mode).",
        )
    
    return gap


def enforce_policy_floor_after_all_layers(
    gap: EfficiencyGap,
    current_cooling: float,
) -> EfficiencyGap:
    """Re-apply active-compute / spike-hold RPM floor after slew & hysteresis.

    Mechanical longevity (often ±5%/cycle when cool) and equipment safety can shrink
    ``recommended_cooling_delta`` so the command never reaches the guardrail floor.
    This step restores the **minimum RPM** implied by ``policy_soft_floor_rpm_target``.

    When ``spike_bypass`` is set, uses ``spike_hold_fan_floor_rpm`` from the
    calibration profile as the effective floor immediately — removes the one-cycle
    delay between spike detection and floor enforcement.
    """
    if gap.emergency_mode:
        return gap

    # When spike_bypass is active, use spike_hold_floor directly from profile
    if gap.spike_bypass:
        cp_dict = gap.raw_metrics.get("calibration_profile")
        if cp_dict and isinstance(cp_dict, dict):
            spike_floor = cp_dict.get("spike_hold_fan_floor_rpm", 0)
            if spike_floor > 0 and current_cooling > 1e-6:
                proposed = current_cooling * (1.0 + gap.recommended_cooling_delta)
                if proposed + 0.5 < spike_floor:
                    gap.recommended_cooling_delta = (spike_floor / current_cooling) - 1.0
                    gap.hysteresis_hold = False
                    gap.raw_metrics["policy_floor_forced_after_layers"] = True
                    gap.reasoning_log.append(
                        f"Spike bypass floor enforcement: raised command to ≥ {spike_floor:.0f} RPM "
                        f"immediately (telemetry RPM {current_cooling:.0f})."
                    )
                return gap

    raw = gap.raw_metrics.get("policy_soft_floor_rpm_target")
    if raw is None:
        return gap
    try:
        target_f = float(raw)
    except (TypeError, ValueError):
        return gap
    if target_f <= 0 or current_cooling < 1e-6:
        return gap
    proposed = current_cooling * (1.0 + gap.recommended_cooling_delta)
    if proposed + 0.5 >= target_f:
        return gap
    gap.recommended_cooling_delta = (target_f / current_cooling) - 1.0
    gap.hysteresis_hold = False
    gap.raw_metrics["policy_floor_forced_after_layers"] = True
    gap.reasoning_log.append(
        f"Policy RPM floor re-applied after mechanical/safety layers: "
        f"raised command to ≥ {target_f:.0f} RPM (telemetry RPM {current_cooling:.0f})."
    )
    return gap


# --- Mechanical Longevity Layer ---

# Threshold: cooling below this fraction of max = "off"
_COOLING_OFF_THRESHOLD = 0.05  # 5% of max

# Hysteresis: only issue RPM command if change > this fraction of current speed (prevents chattering)
HYSTERESIS_DEADBAND = 0.05  # 5% - protects fan lifespan (Enterprise requirement)


class MechanicalLongevityLayer:
    """
    Protects CRAC/chiller equipment: anti-short-cycle, slew rate limiting,
    and staggered start scheduling.
    """

    def __init__(
        self,
        min_off_time: float = MIN_OFF_TIME,
        min_run_time: float = MIN_RUN_TIME,
        max_change_rate: float = MAX_COOLING_CHANGE_RATE,
        stagger_delay: float = STAGGER_START_DELAY,
    ):
        self.min_off_time = min_off_time
        self.min_run_time = min_run_time
        self.max_change_rate = max_change_rate
        self.stagger_delay = stagger_delay
        # Per-unit state: unit_id -> {"state": "on"|"off", "since": monotonic_time}
        self._unit_state: Dict[str, Dict[str, Any]] = {}
        self._last_cooling_per_unit: Dict[str, float] = {}

    def _is_unit_on(self, cooling: float, max_cooling: float) -> bool:
        """Unit is 'on' if cooling exceeds 5% of max capacity.

        Args:
            cooling: Current cooling output.
            max_cooling: Maximum cooling capacity.

        Returns:
            True if unit is considered on.
        """
        if max_cooling < 1e-6:
            return False
        return cooling / max_cooling > _COOLING_OFF_THRESHOLD

    def _update_unit_state(self, unit_id: str, cooling: float, max_cooling: float) -> None:
        """Track unit on/off state and transition time."""
        now = time.monotonic()
        is_on = self._is_unit_on(cooling, max_cooling)

        if unit_id not in self._unit_state:
            # First call: assume unit has been in current state long enough (no history)
            self._unit_state[unit_id] = {
                "state": "on" if is_on else "off",
                "since": now - max(self.min_run_time, self.min_off_time) - 1,
            }
            return

        prev = self._unit_state[unit_id]
        if (is_on and prev["state"] == "on") or (not is_on and prev["state"] == "off"):
            return  # No change
        # State transition detected
        self._unit_state[unit_id] = {"state": "on" if is_on else "off", "since": now}

    def apply_anti_short_cycle(
        self,
        gap: EfficiencyGap,
        current_cooling: float,
        max_cooling: float,
        unit_id: str = "default",
    ) -> EfficiencyGap:
        """Apply anti-short-cycle protection.

        If AI tries to turn unit off before MIN_RUN_TIME or on before
        MIN_OFF_TIME, holds current state (recommended_cooling_delta=0).

        Args:
            gap: EfficiencyGap to modify.
            current_cooling: Current cooling output.
            max_cooling: Maximum cooling capacity.
            unit_id: Identifier of the cooling unit.

        Returns:
            EfficiencyGap with anti-short-cycle applied if hold triggered.
        """
        if gap.emergency_mode:
            return gap  # Emergency bypasses anti-short-cycle
        if gap.spike_bypass:
            return gap  # Spike bypass: jump to target in one cycle

        self._update_unit_state(unit_id, current_cooling, max_cooling)
        now = time.monotonic()
        state_info = self._unit_state.get(unit_id, {"state": "off", "since": now})
        elapsed = now - state_info["since"]
        is_on = state_info["state"] == "on"

        # Proposed new cooling from AI
        proposed = current_cooling * (1.0 + gap.recommended_cooling_delta)
        proposed_on = self._is_unit_on(proposed, max_cooling)
        # When current=0, proposed=0 always; treat delta > 0.5 as "AI wants to turn on"
        if not proposed_on and not is_on and gap.recommended_cooling_delta > 0.5:
            proposed_on = True  # AI wants significant cooling from off state

        hold = False
        reason = ""

        if is_on and not proposed_on:
            # AI wants to turn OFF - check MIN_RUN_TIME
            if elapsed < self.min_run_time:
                hold = True
                reason = (
                    f"Anti-short-cycle: Unit has run {elapsed:.0f}s < MIN_RUN_TIME ({self.min_run_time}s). "
                    "Holding current state."
                )
        elif not is_on and proposed_on:
            # AI wants to turn ON - check MIN_OFF_TIME
            if elapsed < self.min_off_time:
                hold = True
                reason = (
                    f"Anti-short-cycle: Unit has been off {elapsed:.0f}s < MIN_OFF_TIME ({self.min_off_time}s). "
                    "Holding current state."
                )

        if hold:
            gap.anti_short_cycle_hold = True
            gap.recommended_cooling_delta = 0.0  # No change
            gap.recommendations.insert(0, f"⚙️ {reason}")
            gap.reasoning_log.append(reason)

        return gap

    def get_dynamic_slew_rate(
        self,
        current_thermal: float,
        max_safe_temp: float = 80.0,
    ) -> float:
        """
        Dynamic slew rate: gentle when far from limit, aggressive when approaching spike.
        Far from limit -> save motor wear. Approaching spike -> prioritize safety.

        Very large margins (>30°C) allow faster convergence to efficient setpoints
        without repeated 5% baby-steps that keep fans high for minutes.
        """
        margin = max_safe_temp - current_thermal
        if margin <= 3.0:
            return 0.25  # Aggressive when close to limit
        if margin <= 10.0:
            return 0.15  # Moderate
        if margin > 30.0:
            return 0.15  # Large margin: faster convergence to efficient setpoint
        return 0.05  # Normal: gentle (save wear)

    def apply_slew_rate_limit(
        self,
        gap: EfficiencyGap,
        current_cooling: float,
        max_cooling: float,
        current_thermal: Optional[float] = None,
        max_safe_temp: float = 80.0,
    ) -> EfficiencyGap:
        """Limit cooling change rate to prevent thermal shock.

        Clamps recommended_cooling_delta to ±slew_rate. Uses dynamic slew
        rate when current_thermal provided.

        Args:
            gap: EfficiencyGap to modify.
            current_cooling: Current cooling output.
            max_cooling: Maximum cooling capacity.
            current_thermal: Optional current temp for dynamic slew rate.
            max_safe_temp: Maximum safe temperature.

        Returns:
            EfficiencyGap with slew rate clamped if applicable.
        """
        if gap.emergency_mode:
            return gap  # Emergency bypasses slew limit
        if gap.spike_bypass:
            return gap  # Spike bypass: jump to target in one cycle

        rate = self.max_change_rate
        if current_thermal is not None:
            rate = self.get_dynamic_slew_rate(current_thermal, max_safe_temp)

        delta = gap.recommended_cooling_delta
        clamped = max(-rate, min(rate, delta))
        if abs(clamped - delta) > 1e-9:
            gap.slew_rate_limited = True
            gap.recommended_cooling_delta = clamped
            msg = (
                f"Slew rate limit: Delta clamped from {delta:.2f} to ±{rate:.2f} "
                "(gentle when far from limit, aggressive when approaching)."
            )
            gap.recommendations.insert(0, f"⚙️ {msg}")
            gap.reasoning_log.append(msg)
        return gap

    def get_staggered_start_delays(self, n_units: int) -> List[float]:
        """
        Return staggered start delays (seconds) for n units to prevent inrush spikes.
        Unit 0: 0s, Unit 1: 2s, Unit 2: 4s, ...
        """
        return [i * self.stagger_delay for i in range(n_units)]

    def apply_hysteresis(
        self,
        gap: EfficiencyGap,
        deadband: float = HYSTERESIS_DEADBAND,
    ) -> EfficiencyGap:
        """
        Hysteresis: only issue RPM command if change > deadband (default 5%).

        Prevents chattering—tiny, frequent adjustments that wear fans.
        Protects physical lifespan (Enterprise requirement).
        If |delta| < deadband, set to 0 (no command).

        FIX C: Spike/floor commands bypass hysteresis — when a policy floor
        is driving the delta upward, suppressing it would defeat the floor.
        """
        if gap.emergency_mode:
            return gap  # Emergency bypasses hysteresis
        if gap.spike_bypass:
            return gap  # Spike bypass: jump to target in one cycle

        # FIX C: Bypass hysteresis when a policy floor is actively raising cooling
        floor_target = gap.raw_metrics.get("policy_soft_floor_rpm_target")
        if floor_target is not None and float(floor_target) > 0 and gap.recommended_cooling_delta > 0:
            return gap  # Policy floor driving delta up — do not suppress

        # Aggregate delta
        delta = gap.recommended_cooling_delta
        if abs(delta) < deadband and abs(delta) > 1e-9:
            gap.recommended_cooling_delta = 0.0
            gap.hysteresis_hold = True
            msg = (
                f"Hysteresis: Delta {delta:.2%} < {deadband:.0%} deadband—no command issued "
                "(prevents fan chattering, protects lifespan)."
            )
            gap.recommendations.insert(0, f"⚙️ {msg}")
            gap.reasoning_log.append(msg)

        # Per-unit deltas (failing-component redistribution, pre-cooling)
        for unit_id, unit_delta in list(gap.recommended_cooling_delta_by_unit.items()):
            if abs(unit_delta) < deadband and abs(unit_delta) > 1e-9:
                del gap.recommended_cooling_delta_by_unit[unit_id]
                if not gap.hysteresis_hold:
                    gap.hysteresis_hold = True
                    msg = f"Hysteresis: Per-unit delta for {unit_id} below {deadband:.0%} deadband—skipped."
                    gap.recommendations.insert(0, f"⚙️ {msg}")
                    gap.reasoning_log.append(msg)

        return gap


def apply_mechanical_longevity(
    gap: EfficiencyGap,
    current_cooling: float,
    max_cooling: float = 3000.0,
    layer: Optional[MechanicalLongevityLayer] = None,
    unit_id: str = "default",
    current_thermal: Optional[float] = None,
    max_safe_temp: float = MAX_SAFE_TEMP,
) -> EfficiencyGap:
    """Apply Mechanical Longevity Layer to an EfficiencyGap.

    Applies: (1) Slew rate limit (dynamic when current_thermal provided);
    (2) Anti-short-cycle; (3) Hysteresis (no command if change < 5%).

    Args:
        gap: EfficiencyGap to modify.
        current_cooling: Current cooling output.
        max_cooling: Maximum cooling capacity.
        layer: Optional MechanicalLongevityLayer; creates default if None.
        unit_id: Identifier of the cooling unit.
        current_thermal: Optional current temp for dynamic slew rate.
        max_safe_temp: Maximum safe temperature.

    Returns:
        EfficiencyGap with mechanical longevity applied.
    """
    if layer is None:
        layer = MechanicalLongevityLayer()

    # 1. Slew rate limit (dynamic: gentle when far, aggressive when approaching limit)
    gap = layer.apply_slew_rate_limit(
        gap, current_cooling, max_cooling,
        current_thermal=current_thermal,
        max_safe_temp=max_safe_temp,
    )

    # 2. Anti-short-cycle (CRAC/chiller protection)
    gap = layer.apply_anti_short_cycle(gap, current_cooling, max_cooling, unit_id)

    # 3. Hysteresis: only issue command if change > 5% (prevents chattering, protects fan lifespan)
    gap = layer.apply_hysteresis(gap)

    return gap


def get_staggered_start_delays(n_units_starting: int) -> List[float]:
    """
    Power surge protection: 2-second stagger between unit starts.
    Returns [0, 2, 4, ...] seconds delay for each unit.
    """
    return [i * STAGGER_START_DELAY for i in range(n_units_starting)]


class OptimizationBrain:
    """Advanced predictive optimization engine for data center thermal control.

    Computes Efficiency Gap from: thermal lag (power vs cooling response),
    over-provisioning (excess cooling), oscillation (hunting behavior).
    Uses T+10s prediction, thermal inertia, reward function, guardrails,
    and mechanical longevity. Supports failing-component redistribution
    via InfluenceMap when anomaly detection flags degraded units.
    """

    def __init__(
        self,
        target_temp: float = 65.0,
        lag_threshold_seconds: float = 5.0,
        over_provision_threshold: float = 0.15,
        oscillation_threshold: float = 0.2,
        mechanical_longevity_layer: Optional[MechanicalLongevityLayer] = None,
        equipment_safety: Optional[Any] = None,
        predictor: Optional[TempPredictor] = None,
        learning_log_path: str = DEFAULT_LEARNING_LOG,
        shadow_mode: bool = False,
        power_optimizer: Optional[PowerCostOptimizer] = None,
        calibration_profile: Optional[Any] = None,
    ):
        self.target_temp = target_temp
        self.lag_threshold = lag_threshold_seconds
        self.over_provision_threshold = over_provision_threshold
        self.oscillation_threshold = oscillation_threshold
        self.mechanical_layer = mechanical_longevity_layer or MechanicalLongevityLayer()
        self.equipment_safety = equipment_safety
        self.predictor = predictor or TempPredictor()
        self.learning_log_path = learning_log_path
        self.shadow_mode = shadow_mode
        self.power_optimizer = power_optimizer or PowerCostOptimizer()
        self.calibration_profile = calibration_profile

        # Sustained compute power history for absolute trigger (FIX B: 12s rolling mean)
        self._sustained_power_history: List[float] = []
        self._sustained_power_times: List[float] = []

        # FIX A: Peak-hold power buffer (3-second max for spike detection)
        self._peak_power_buffer: List[float] = []  # power values
        self._peak_power_times: List[float] = []    # timestamps
        self._PEAK_HOLD_WINDOW_S = 3.0

        # FIX G: No-response event counter (cycles where no fan command fired)
        self._no_response_count: int = 0
    
    def _nodes_to_arrays(self, nodes: List[BaseNode]) -> tuple:
        """
        Extract arrays from BaseNode list for analysis.
        Returns (thermal, power, cooling, utilization).
        """
        thermal = np.array([n.thermal_input for n in nodes])
        power = np.array([n.power_draw for n in nodes])
        cooling = np.array([n.cooling_output for n in nodes])
        utilization = np.array([n.utilization for n in nodes])
        return thermal, power, cooling, utilization
    
    def _compute_thermal_lag(self, power: np.ndarray, cooling: np.ndarray) -> float:
        """Compute thermal lag: time between power peak and cooling peak.

        Physics: In reactive control, cooling responds after power spike.
        Positive lag = cooling lags power; proactive control reduces lag.
        Measured in samples (or seconds if timestamps are uniform).

        Args:
            power: Power draw array.
            cooling: Cooling output array.

        Returns:
            Lag in samples (idx_max_cooling - idx_max_power).
        """
        if len(power) < 2 or len(cooling) < 2:
            return 0.0
        idx_max_power = np.argmax(power)
        idx_max_cooling = np.argmax(cooling)
        return float(idx_max_cooling - idx_max_power)
    
    def _compute_over_provisioning(
        self,
        thermal: np.ndarray,
        cooling: np.ndarray,
        power: np.ndarray,
        thermal_inertia_s: float = 120.0,
    ) -> float:
        """Compute over-provisioning ratio.

        Physics: Cooling above need when temp < target. Thermal inertia:
        after load drop, hardware holds heat longer—brief over-cooling
        is expected; inertia_factor reduces penalty in that window.

        Args:
            thermal: Temperature array.
            cooling: Cooling output array.
            power: Power draw array.
            thermal_inertia_s: Thermal inertia constant (seconds).

        Returns:
            Ratio in [0, 1] of time over-provisioned.
        """
        if len(thermal) < 2 or np.max(cooling) < 1e-6:
            return 0.0
        below_target = thermal < self.target_temp
        high_cooling = cooling > np.percentile(cooling, 75)
        # Thermal inertia: if power dropped recently, reduce over-provision penalty
        power_diff = np.diff(power, prepend=power[0]) if len(power) > 1 else np.zeros(1)
        recent_drop = len(power_diff) > 0 and power_diff[-1] < -20
        inertia_factor = 0.5 if recent_drop else 1.0  # Less penalty right after load drop
        over_provisioned = np.sum(below_target & high_cooling) / max(len(thermal), 1)
        return float(over_provisioned * inertia_factor)
    
    def _compute_oscillation(self, cooling: np.ndarray) -> float:
        """
        Oscillation: Fan/pump hunting - rapid up/down cycling.
        Ratio of time with high rate-of-change in cooling output.
        """
        if len(cooling) < 3:
            return 0.0
        cooling_diff = np.abs(np.diff(cooling, prepend=cooling[0]))
        valid_diff = cooling_diff[np.isfinite(cooling_diff)]
        if len(valid_diff) < 1:
            return 0.0
        threshold = np.percentile(valid_diff, 85)
        hunting = cooling_diff > max(threshold, 1e-9)
        return float(np.mean(hunting))
    
    def _compute_efficiency_score(self, gap: EfficiencyGap) -> float:
        """Compute overall efficiency score (0–100).

        Physics: Start at 100; subtract penalties for thermal lag (max -20),
        over-provisioning (max -25), oscillation (max -25).

        Args:
            gap: EfficiencyGap with thermal_lag, over_provisioning, oscillation.

        Returns:
            Score in [0, 100], higher = better.
        """
        score = 100.0
        # Penalty for thermal lag (max -20)
        if gap.thermal_lag_seconds > self.lag_threshold:
            score -= min(20, gap.thermal_lag_seconds * 2)
        # Penalty for over-provisioning (max -25)
        score -= gap.over_provisioning_ratio * 25
        # Penalty for oscillation (max -25)
        score -= gap.oscillation_ratio * 25
        return max(0, min(100, score))
    
    def _generate_recommendations(self, gap: EfficiencyGap) -> List[str]:
        """Generate human-readable optimization recommendations."""
        recs = []
        if gap.thermal_lag_seconds > self.lag_threshold:
            recs.append(
                f"Thermal Lag: {gap.thermal_lag_seconds:.1f}s delay. "
                "Use predictive control to ramp cooling before heat spike."
            )
        if gap.over_provisioning_ratio > self.over_provision_threshold:
            recs.append(
                f"Over-provisioning: {gap.over_provisioning_ratio*100:.1f}% excess cooling. "
                "Reduce cooling output when temp is below target."
            )
        if gap.oscillation_ratio > self.oscillation_threshold:
            recs.append(
                f"Oscillation: {gap.oscillation_ratio*100:.1f}% hunting behavior. "
                "Smooth fan/pump curves to reduce mechanical wear."
            )
        if not recs:
            recs.append("Efficiency within target. Maintain current settings.")
        return recs

    def _generate_diagnostic_reasoning(
        self,
        gap: EfficiencyGap,
        nodes: List[BaseNode],
        current_thermal: float,
        current_cooling: float,
    ) -> str:
        """Generate predictive diagnostic explanation.

        Uses thermal momentum (dT/dt) and InfluenceMap for explainability.
        Example: "I am increasing Fan A by 10% because Rack 4 has thermal
        momentum of +1.5°C/min, and Fan A has 0.8 influence on that zone."

        Args:
            gap: EfficiencyGap with llm_prompt_context and raw_metrics.
            nodes: List of nodes (for zone naming).
            current_thermal: Current temperature (°C).
            current_cooling: Current cooling output.

        Returns:
            Human-readable diagnostic string.
        """
        delta = gap.recommended_cooling_delta
        delta_pct = round(delta * 100.0)
        action = "increasing" if delta > 0.05 else ("reducing" if delta < -0.05 else "holding")
        cooling_label = "cooling"

        # Rate of change: prefer per-node for zone naming, else aggregate
        rate_of_change = gap.llm_prompt_context.get("rate_of_change") or {}
        rate_c_per_min = rate_of_change.get("max_rate_c_per_min")
        if rate_c_per_min is None and rate_of_change.get("max_rate_c_per_s") is not None:
            rate_c_per_min = rate_of_change["max_rate_c_per_s"] * 60.0
        thermal_rate_by_node = gap.raw_metrics.get("thermal_rate_by_node") or []
        hot_zone_label = "the monitored zone"
        if thermal_rate_by_node:
            # Node with highest (positive) rate = hottest-climbing zone
            by_rate = [(nid, r) for nid, r in thermal_rate_by_node if r is not None]
            if by_rate:
                node_id, rate_c_per_s = max(by_rate, key=lambda x: x[1] or 0.0)
                rate_c_per_min = (rate_c_per_s or 0.0) * 60.0
                hot_zone_label = node_id.replace("_", " ").strip() or "the hottest-climbing zone"
        if rate_c_per_min is None:
            rate_c_per_min = 0.0

        influence_map = gap.llm_prompt_context.get("influence_map") or gap.raw_metrics.get("influence_map") or {}
        cooling_units = influence_map.get("cooling_units") or []
        weights = influence_map.get("influence_coefficients") or influence_map.get("weights") or {}
        unit_label = cooling_label
        influence_coef = None
        zone_for_influence = hot_zone_label

        if cooling_units and weights:
            unit_label = cooling_units[0]
            zone_weights = weights.get(unit_label) or {}
            rack_zones = influence_map.get("rack_zones") or list(zone_weights.keys())
            if rack_zones:
                zone_for_influence = rack_zones[0]
                influence_coef = zone_weights.get(zone_for_influence)
            if influence_coef is None and zone_weights:
                zone_for_influence = next(iter(zone_weights), "")
                influence_coef = zone_weights.get(zone_for_influence)

        sign = "+" if rate_c_per_min >= 0 else ""
        momentum_str = f"{sign}{rate_c_per_min:.1f}°C/min"

        if influence_coef is not None:
            return (
                f"I am {action} {unit_label} by {abs(delta_pct)}% because {hot_zone_label} has a thermal "
                f"momentum of {momentum_str}, and {unit_label} has a {influence_coef:.2f} influence "
                f"coefficient on that zone."
            )
        if rate_c_per_min != 0 or abs(delta) > 0.05:
            return (
                f"I am {action} {cooling_label} by {abs(delta_pct)}% because {hot_zone_label} has a thermal "
                f"momentum of {momentum_str} (predictive adjustment)."
            )
        return (
            f"I am {action} current settings. Predicted temperature and efficiency metrics "
            "are within target; no change needed."
        )
    
    def _apply_failing_component_redistribution(
        self,
        gap: EfficiencyGap,
        nodes: List[BaseNode],
        influence_map: Optional[InfluenceMap],
        failing_findings: List[Any],
    ) -> None:
        """
        When failing components exist: reduce load on those units and redistribute
        to healthy neighboring units (from InfluenceMap). Updates gap with
        recommended_cooling_delta_by_unit and diagnostic reasoning.
        """
        if not failing_findings or not nodes:
            return
        failing_ids = {f.node_id if hasattr(f, "node_id") else f.get("node_id", "") for f in failing_findings}
        failing_ids = {x for x in failing_ids if x}
        if not failing_ids:
            return

        # Resolve healthy neighboring units from InfluenceMap (units that share zones with failing ones)
        all_units: List[str] = []
        healthy_for_redistribution: List[str] = []
        if influence_map is not None:
            all_units = list(influence_map.cooling_units)
            zones_served_by_failing = set()
            for fid in failing_ids:
                for zone_id, _ in influence_map.get_zones_for_cooling_unit(fid):
                    zones_served_by_failing.add(zone_id)
            for zone_id in zones_served_by_failing:
                for cu_id, _ in influence_map.get_cooling_units_for_zone(zone_id):
                    if cu_id not in failing_ids and cu_id not in healthy_for_redistribution:
                        healthy_for_redistribution.append(cu_id)
            if not healthy_for_redistribution and all_units:
                healthy_for_redistribution = [u for u in all_units if u not in failing_ids]

        # Fallback: any node_id in nodes that is not failing (for when there is no influence map)
        if not healthy_for_redistribution:
            for n in nodes:
                nid = getattr(n, "node_id", None) or ""
                if nid and nid not in failing_ids:
                    healthy_for_redistribution.append(nid)

        # Per-unit deltas: reduce failing, increase healthy
        delta_reduce = -0.15  # reduce load on failing unit
        delta_increase = 0.10  # increase on healthy to compensate (can scale by number of healthy)
        if healthy_for_redistribution:
            delta_increase = min(0.15, 0.10 * max(1, len(failing_ids)))  # slightly more if multiple failing
        for fid in failing_ids:
            gap.recommended_cooling_delta_by_unit[fid] = delta_reduce
        for hid in healthy_for_redistribution:
            gap.recommended_cooling_delta_by_unit[hid] = delta_increase

        # Aggregate delta: neutral or slight reduce when redistributing
        gap.recommended_cooling_delta = 0.0

        # Diagnostic: "Reducing load on [Unit Name] due to detected inefficiency; redistributing cooling to [Other Units]."
        failing_names = ", ".join(sorted(failing_ids))
        other_units = ", ".join(sorted(healthy_for_redistribution)[:10]) if healthy_for_redistribution else "other available units"
        if len(healthy_for_redistribution) > 10:
            other_units += ", ..."
        gap.diagnostic_reasoning = (
            f"Reducing load on {failing_names} due to detected inefficiency; "
            f"redistributing cooling to {other_units}."
        )
        gap.reasoning_log.append(
            f"Reducing {failing_names} RPM by 15% and increasing {other_units} by 10% because "
            f"failing component detected (power up, ΔT down)—predictive maintenance redistribution."
        )
        gap.recommendations.insert(
            0,
            f"Predictive maintenance: Reduce setpoint/load on {failing_names} (failing component); "
            f"redistribute to {other_units}.",
        )
        gap.raw_metrics["failing_components"] = [
            {"node_id": f.node_id if hasattr(f, "node_id") else f.get("node_id"), "message": f.message if hasattr(f, "message") else f.get("message", "")}
            for f in failing_findings
        ]
        gap.raw_metrics["redistribution_healthy_units"] = healthy_for_redistribution

    def _apply_precooling(
        self,
        gap: EfficiencyGap,
        upcoming_jobs: List[Any],
        influence_map: Optional[InfluenceMap],
        precool_delta: float = 0.20,
        max_seconds: float = 60.0,
    ) -> None:
        """
        When a rack is about to get a job in < max_seconds, set Pre-Cooling state
        for fans serving that rack (increase RPM by precool_delta, e.g. 20%).

        Uses InfluenceMap if available; falls back to topology_map.yaml (rack_to_cooling).
        Adds log entry: 'Pre-emptive cooling triggered for Rack X due to upcoming Job ID Y'.
        """
        if not upcoming_jobs:
            return

        for job in upcoming_jobs:
            if isinstance(job, dict):
                rack_id = job.get("rack_id", "")
                job_id = job.get("job_id", "")
                seconds = job.get("seconds_until_start", 0)
            else:
                rack_id = getattr(job, "rack_id", None) or ""
                job_id = getattr(job, "job_id", None) or ""
                seconds = getattr(job, "seconds_until_start", None) or 0

            if not rack_id or not job_id or seconds >= max_seconds:
                continue

            # Get cooling units: InfluenceMap first, then topology_map.yaml fallback
            cooling_units: List[tuple] = []
            if influence_map is not None:
                cooling_units = list(influence_map.get_cooling_units_for_zone(rack_id))
                if not cooling_units:
                    for zone in influence_map.rack_zones:
                        if rack_id in zone or zone in rack_id:
                            cooling_units = influence_map.get_cooling_units_for_zone(zone)
                            break
            if not cooling_units:
                try:
                    from core.config import get_rack_to_cooling_map
                    rack_to_cooling = get_rack_to_cooling_map()
                    unit_ids = rack_to_cooling.get(rack_id, [])
                    cooling_units = [(uid, 1.0) for uid in unit_ids]
                except ImportError:
                    pass

            for unit_id, _ in cooling_units:
                gap.recommended_cooling_delta_by_unit[unit_id] = precool_delta

            if not cooling_units:
                continue

            _precool_logger.info(
                "Pre-emptive cooling triggered for Rack %s due to upcoming Job ID %s",
                rack_id,
                job_id,
                extra={"rack_id": rack_id, "job_id": job_id, "seconds_until_start": seconds},
            )
            unit_list = ", ".join(u for u, _ in cooling_units)
            gap.reasoning_log.append(
                f"Increasing {unit_list} RPM by 20% because Rack {rack_id} thermal momentum is positive "
                f"and Job_ID {job_id} is starting in {seconds:.0f}s (pre-emptive cooling)."
            )
            gap.recommendations.insert(
                0,
                f"Pre-emptive cooling triggered for Rack {rack_id} due to upcoming Job ID {job_id}.",
            )

    def analyze(
        self,
        nodes: List[BaseNode],
        influence_map: Optional[InfluenceMap] = None,
        failing_component_findings: Optional[List[Any]] = None,
        upcoming_jobs: Optional[List[Any]] = None,
        policy_context: Optional[Dict[str, Any]] = None,
        thermal_session_key: Optional[str] = None,
        current_cpu_temp_c: Optional[float] = None,
    ) -> EfficiencyGap:
        """Analyze normalized node data and compute Efficiency Gap.

        Physics pipeline:
        1. Extract thermal, power, cooling arrays; compute thermal momentum (dT/dt).
        2. Thermal lag: power peak vs cooling peak lag.
        3. Over-provisioning: cooling when temp < target; thermal inertia reduces penalty after load drop.
        4. Oscillation: cooling hunting (high |d(cooling)/dt|).
        5. T+10s prediction: temp_for_decision uses predicted temp when confidence > 0.6.
        6. Recommendation: reduce cooling if over-provisioned; increase if temp rising.
        7. Guardrails (ASHRAE, MIN_FAN_RPM), mechanical longevity (slew, anti-short-cycle).
        8. Failing-component redistribution: reduce load on flagged units, increase healthy neighbors.
        9. Pre-cooling: if upcoming_jobs has racks with jobs < 60s, boost fan RPM by 20% for those zones.

        Args:
            nodes: List of BaseNode (may include thermal_input_rate).
            influence_map: Optional InfluenceMap for zone-to-unit mapping.
            failing_component_findings: Optional list of FailingComponentFinding.
            upcoming_jobs: Optional list of UpcomingJob (or dicts with rack_id, job_id, seconds_until_start).
            policy_context: Optional dict from API layer. Keys:
                ``rated_max_fan_rpm`` (required for correct active-compute floor when telemetry
                max RPM is stuck low), ``spike_hold_active`` + ``spike_hold_min_rpm`` (legacy).
            thermal_session_key: Tenant/session id for shared post-spike RPM hold (see spike_hold_state).
                When set, ``analyze`` records max temperature from this batch and applies the same
                hysteresis as the agent path — pass e.g. ``owner_id`` from all server routes that
                should behave consistently.

        Returns:
            EfficiencyGap with metrics, recommendations, and per-unit deltas.
        """
        if not nodes:
            return EfficiencyGap(
                recommendations=["No data to analyze. Check data source."],
            )

        thermal, power, cooling, utilization = self._nodes_to_arrays(nodes)
        current_thermal = float(thermal[-1]) if len(thermal) > 0 else 0.0
        current_cooling = float(cooling[-1]) if len(cooling) > 0 else 0.0
        current_power = float(power[-1]) if len(power) > 0 else 0.0
        observed_max_cooling = float(np.max(cooling)) if len(cooling) > 0 else 3000.0
        observed_max_cooling = max(observed_max_cooling, 1000.0)

        rated_max_fan_rpm: Optional[float] = None
        if policy_context is not None and policy_context.get("rated_max_fan_rpm") is not None:
            try:
                rated_max_fan_rpm = float(policy_context["rated_max_fan_rpm"])
            except (TypeError, ValueError):
                rated_max_fan_rpm = None

        # Use rated capacity (from agent hardware config) as max_cooling when available,
        # so the optimizer works with the full hardware range rather than the observed
        # range (which may be artificially low when fans have been running at low duty).
        max_cooling = max(observed_max_cooling, rated_max_fan_rpm or observed_max_cooling)

        capacity_for_policy = policy_capacity_rpm(max_cooling, rated_max_fan_rpm)

        sustained_active = sustained_active_compute_signal(np.asarray(power, dtype=float))

        # --- FIX A: Peak-hold power buffer (3-second max for spike detection) ---
        cp = self.calibration_profile
        now_mono = time.monotonic()

        # Maintain 3-second peak-hold buffer
        self._peak_power_buffer.append(current_power)
        self._peak_power_times.append(now_mono)
        peak_cutoff = now_mono - self._PEAK_HOLD_WINDOW_S
        while self._peak_power_times and self._peak_power_times[0] < peak_cutoff:
            self._peak_power_times.pop(0)
            self._peak_power_buffer.pop(0)
        peak_power_3s = max(self._peak_power_buffer) if self._peak_power_buffer else current_power

        # --- FIX B: Rolling 12-second mean replaces consecutive counter ---
        active_trigger_w = cp.active_compute_trigger_w if cp is not None else SUSTAINED_COMPUTE_MEAN_THRESHOLD_W
        sustained_window_s = cp.sustained_compute_window_s if cp is not None else 12.0
        self._sustained_power_history.append(current_power)
        self._sustained_power_times.append(now_mono)
        # Prune history older than sustained_window_s
        cutoff = now_mono - sustained_window_s
        while self._sustained_power_times and self._sustained_power_times[0] < cutoff:
            self._sustained_power_times.pop(0)
            self._sustained_power_history.pop(0)

        # FIX F: Require minimum 15 samples before sustained compute decisions
        _MIN_SUSTAINED_SAMPLES = 15

        # --- IDLE GATE: if current power is clearly below trigger, force sustained_active=False ---
        # This prevents stale high-power samples in the rolling window from keeping the
        # active-compute floor engaged during genuinely idle periods.
        # Use a generous multiplier (0.85) so we only suppress when clearly idle.
        _idle_gate_w = active_trigger_w * 0.85
        _current_is_idle = current_power < _idle_gate_w

        if not _current_is_idle:
            if len(self._sustained_power_history) >= _MIN_SUSTAINED_SAMPLES:
                rolling_mean = float(np.mean(self._sustained_power_history))
                if rolling_mean > active_trigger_w:
                    sustained_active = True
            # FIX D: Peak power independently triggers active compute (no temp/confidence gate)
            if peak_power_3s > active_trigger_w * 1.5:
                sustained_active = True
        else:
            # Current power is below idle gate — suppress sustained_active even if
            # legacy tail or old rolling mean would have triggered it.
            sustained_active = False
            _reasoning_logger.debug(
                "[IDLE_GATE] sustained_active forced False: current_power=%.1fW < idle_gate=%.1fW "
                "(active_trigger=%.1fW)",
                current_power, _idle_gate_w, active_trigger_w,
            )

        # FIX G: Track trigger source for diagnostics
        active_trigger_source = "none"
        if sustained_active:
            if peak_power_3s > active_trigger_w * 1.5:
                active_trigger_source = "peak_power_3s"
            elif len(self._sustained_power_history) >= _MIN_SUSTAINED_SAMPLES and float(np.mean(self._sustained_power_history)) > active_trigger_w:
                active_trigger_source = "rolling_12s_mean"
            else:
                active_trigger_source = "legacy_tail"
        elif _current_is_idle:
            active_trigger_source = "idle_gate_suppressed"

        if thermal_session_key:
            _spike_trigger_c = cp.spike_trigger_temp_c if cp is not None else None
            _spike_hold_dur = cp.spike_hold_duration_s if cp is not None else None
            record_spike_if_hot(
                thermal_session_key,
                float(np.max(thermal)),
                time.time(),
                spike_trigger_temp_c=_spike_trigger_c,
                spike_hold_duration_s=_spike_hold_dur,
            )
        spike_floor_rpm = max(
            parse_policy_context(policy_context),
            spike_hold_floor_rpm(thermal_session_key, capacity_for_policy, time.time()),
        )

        # Thermal momentum: first derivative (rate of change) for all temp sensors
        rates = [
            getattr(n, "thermal_input_rate", None)
            for n in nodes
            if getattr(n, "thermal_input_rate", None) is not None
        ]
        thermal_momentum = {}
        if rates:
            thermal_momentum["mean_rate_c_per_s"] = float(np.mean(rates))
            thermal_momentum["max_rate_c_per_s"] = float(np.max(rates))
            thermal_momentum["min_rate_c_per_s"] = float(np.min(rates))
            thermal_momentum["sensors_with_rate"] = len(rates)

        # Thermal inertia: how long hardware holds heat after load drop
        thermal_inertia_s = compute_thermal_inertia_seconds(nodes)
        gap = EfficiencyGap(thermal_inertia_s=thermal_inertia_s)

        # Compute metrics (with thermal inertia in over-provisioning)
        thermal_lag = self._compute_thermal_lag(power, cooling)
        over_provisioning = self._compute_over_provisioning(thermal, cooling, power)
        oscillation = self._compute_oscillation(cooling)

        gap.thermal_lag_seconds = thermal_lag
        gap.over_provisioning_ratio = over_provisioning
        gap.oscillation_ratio = oscillation
        # Per-node rate of change for diagnostic reasoning (zone/rack naming)
        thermal_rate_by_node = [
            (getattr(n, "node_id", f"node_{i}"), getattr(n, "thermal_input_rate", None))
            for i, n in enumerate(nodes)
            if getattr(n, "thermal_input_rate", None) is not None
        ]
        gap.raw_metrics = {
            "thermal_mean": float(np.mean(thermal)),
            "power_mean": float(np.mean(power)),
            "cooling_mean": float(np.mean(cooling)),
            "thermal_momentum": thermal_momentum,
            "thermal_rate_by_node": thermal_rate_by_node,
            "sustained_active_compute": sustained_active,
            # FIX A: peak power over 3-second window
            "peak_power_w": peak_power_3s,
            # FIX G: trigger diagnostics
            "active_trigger_source": active_trigger_source,
            "thermal_session_key": thermal_session_key,
            "spike_hold_min_rpm_request": spike_floor_rpm,
            "merged_policy_soft_floor_rpm": merge_policy_floors(
                max_cooling,
                rated_max_fan_rpm=rated_max_fan_rpm,
                power_draw_w=current_power,
                sustained_active=sustained_active,
                spike_hold_min_rpm=spike_floor_rpm,
                calibration_profile=cp,
            ),
            "policy_capacity_rpm": capacity_for_policy,
            "rated_max_fan_rpm": rated_max_fan_rpm,
        }
        if cp is not None:
            gap.raw_metrics["calibration_profile"] = cp.to_dict()
        if influence_map is not None:
            gap.raw_metrics["influence_map"] = influence_map.to_dict()

        # LLM prompt context: rate of change (incl. °C/min) + influence map for explainability
        rate_c_per_s_max = thermal_momentum.get("max_rate_c_per_s")
        rate_c_per_s_mean = thermal_momentum.get("mean_rate_c_per_s")
        gap.llm_prompt_context = {
            "rate_of_change": {
                "mean_rate_c_per_s": rate_c_per_s_mean,
                "max_rate_c_per_s": rate_c_per_s_max,
                "min_rate_c_per_s": thermal_momentum.get("min_rate_c_per_s"),
                "max_rate_c_per_min": (rate_c_per_s_max * 60.0) if rate_c_per_s_max is not None else None,
                "mean_rate_c_per_min": (rate_c_per_s_mean * 60.0) if rate_c_per_s_mean is not None else None,
                "sensors_with_rate": thermal_momentum.get("sensors_with_rate"),
            },
            "influence_map": influence_map.to_dict() if influence_map is not None else {},
        }

        # Multi-variate correlations (Utilization vs Power vs Inlet)
        try:
            from core.ingestion.data_ingestor import DataIngestor
            ingestor = DataIngestor()
            gap.raw_metrics["correlations"] = ingestor.compute_correlations(nodes)
        except Exception:
            pass

        # Predictive: T+10s temp based on power trajectory
        pred = self.predictor.predict(nodes)
        predicted_t10 = pred.predicted_temp_t10 if pred else current_thermal
        gap.predicted_temp_t10 = predicted_t10
        power_slope = pred.power_trajectory_slope if pred else 0.0
        if pred is not None:
            gap.raw_metrics["predictor_confidence"] = pred.confidence
            gap.raw_metrics["predicted_temp_slope_c_per_s"] = pred.temp_trajectory_slope
            gap.raw_metrics["extrapolation_temp_slope_c_per_s"] = (
                pred.extrapolation_temp_slope_c_per_s
            )
            gap.raw_metrics["predicted_power_slope_w_per_s"] = power_slope
            gap.raw_metrics["power_slope_pre_ramp_moderate"] = (
                power_slope >= POWER_SLOPE_PRE_RAMP_MODERATE_W_S
            )
            gap.raw_metrics["power_slope_pre_ramp_steep"] = (
                power_slope >= POWER_SLOPE_PRE_RAMP_STEEP_W_S
            )

        # Recommended delta based on PREDICTED state (look-ahead), not just current
        # Use predicted T+10 to be proactive
        temp_for_decision = predicted_t10 if pred and pred.confidence > 0.6 else current_thermal
        temp_rising = temp_for_decision > self.target_temp + 5
        # Power-slope pre-ramp: rising package power ⇒ cooling headroom before temp moves.
        # Skip pre-ramp when thermal margin is very large (>15°C below target) — a moderate
        # power ramp with 30°C headroom has minutes of thermal inertia before any risk.
        _pre_ramp_margin = self.target_temp - temp_for_decision
        if power_slope >= POWER_SLOPE_PRE_RAMP_MODERATE_W_S and _pre_ramp_margin <= 5:
            temp_rising = True

        # Power-cost optimizer: solve for lowest zone power (Fan Affinity Laws)
        # Keeps fans in Efficiency Sweet Spot (60-80%) for 40% savings target
        current_cooling_by_unit: Dict[str, float] = {"default": current_cooling}
        max_cooling_by_unit: Dict[str, float] = {"default": max_cooling}
        cooling_unit_ids: List[str] = ["default"]
        if influence_map is not None and influence_map.cooling_units:
            cooling_unit_ids = list(influence_map.cooling_units)
            # Distribute aggregate roughly equally when we don't have per-unit telemetry
            per_unit = current_cooling / max(len(cooling_unit_ids), 1)
            max_per_unit = max_cooling / max(len(cooling_unit_ids), 1)
            current_cooling_by_unit = {uid: per_unit for uid in cooling_unit_ids}
            max_cooling_by_unit = {uid: max_per_unit for uid in cooling_unit_ids}

        # Pre-compute spike bypass flag so the optimizer can skip its slew cap
        _is_spike_bypass = (
            cp is not None
            and peak_power_3s > active_trigger_w * 2.0
            and not _current_is_idle
            and cp.spike_hold_fan_floor_rpm > 0
            and current_cooling < cp.spike_hold_fan_floor_rpm
        )

        opt_result = self.power_optimizer.optimize_for_min_power(
            current_thermal=current_thermal,
            target_temp=self.target_temp,
            max_safe_temp=MAX_SAFE_TEMP,
            current_cooling=current_cooling,
            max_cooling=max_cooling,
            over_provisioning=over_provisioning,
            temp_rising=temp_rising,
            oscillation=oscillation > self.oscillation_threshold,
            current_cooling_by_unit=current_cooling_by_unit,
            max_cooling_by_unit=max_cooling_by_unit,
            cooling_unit_ids=cooling_unit_ids,
            predicted_temp_t10=predicted_t10,
            power_slope_w_per_s=power_slope,
            sustained_active_compute=sustained_active,
            current_power_w=current_power,
            spike_bypass=_is_spike_bypass,
        )

        _reasoning_logger.warning(
            "[BRAIN_DEBUG] target_temp=%.1f temp_for_decision=%.1f temp_rising=%s "
            "over_prov=%.3f power_slope=%.2f pre_ramp_margin=%.1f "
            "pred_t10=%.1f pred_conf=%.2f "
            "opt_delta=%.4f current_cooling=%.0f max_cooling=%.0f",
            self.target_temp, temp_for_decision, temp_rising,
            over_provisioning, power_slope, _pre_ramp_margin,
            predicted_t10 if pred else -1, pred.confidence if pred else -1,
            opt_result.recommended_delta, current_cooling, max_cooling,
        )

        gap.recommended_cooling_delta = opt_result.recommended_delta
        gap.cooling_power_w = opt_result.total_cooling_power_w
        gap.power_savings_w = opt_result.power_savings_vs_current_w
        gap.in_sweet_spot = opt_result.in_sweet_spot

        if over_provisioning > self.over_provision_threshold and not temp_rising:
            gap.reasoning_log.append(
                f"Reducing cooling toward sweet spot (power-optimized): over-provisioning {over_provisioning:.2f}. "
                f"Projected zone power {opt_result.total_cooling_power_w/1000:.1f} kW, savings {opt_result.power_savings_vs_current_w:.0f} W."
            )
        elif temp_rising:
            rate_str = ""
            if thermal_momentum.get("max_rate_c_per_s"):
                rate = thermal_momentum["max_rate_c_per_s"] * 60
                rate_str = f" and thermal momentum +{rate:.1f}°C/min"
            gap.reasoning_log.append(
                f"Increasing cooling (temp rising): predicted T+10 ({predicted_t10:.1f}°C) > target+5{rate_str}. "
                f"Min power increase to stay safe."
            )
        elif oscillation > self.oscillation_threshold:
            # Under sustained compute, do not force a full silent hold — safety net / floors apply next.
            if not sustained_active:
                gap.recommended_cooling_delta = 0.0
                gap.reasoning_log.append(
                    f"Holding cooling because oscillation ratio {oscillation:.2f} exceeds threshold "
                    f"{self.oscillation_threshold} (prevents fan hunting)."
                )
            else:
                gap.reasoning_log.append(
                    f"Oscillation ratio {oscillation:.2f} high but sustained compute active — "
                    f"skipping full hold; policy floors and power-slope rules still apply."
                )
        else:
            gap.reasoning_log.append(
                opt_result.reasoning or "Holding cooling: efficiency within target; power-optimized."
            )

        gap.efficiency_score = self._compute_efficiency_score(gap)
        gap.recommendations = self._generate_recommendations(gap)

        # Estimated kWh saved per hour: Power-Cost Optimizer (Fan Affinity Laws)
        # P₂ = P₁·(N₂/N₁)³ — savings from reducing cooling toward sweet spot
        if gap.power_savings_w > 0:
            gap.kwh_saved = (gap.power_savings_w / 1000.0) * 1.0  # kWh per hour
        else:
            # Fallback when no power savings: over-provisioning × cooling share
            it_kw = current_power / 1000.0
            cooling_share = 0.35
            gap.kwh_saved = max(0.0, over_provisioning * it_kw * cooling_share) if it_kw > 0 else 0.0
        gap.raw_metrics["kwh_saved_per_hour"] = gap.kwh_saved
        gap.raw_metrics["cooling_power_w"] = gap.cooling_power_w
        gap.raw_metrics["power_savings_w"] = gap.power_savings_w

        # Confidence interval: Quantile Regression or variance-based epistemic uncertainty
        # "Knows what it doesn't know" — critical for client trust
        predictor_confidence = pred.confidence if pred else 0.6
        ci_result = estimate_confidence_interval_quantile(
            thermal=thermal,
            power=power,
            cooling=cooling,
            recommended_delta=gap.recommended_cooling_delta,
            current_cooling=current_cooling,
            max_cooling=max_cooling,
            predictor_confidence=predictor_confidence,
            learning_log_path=self.learning_log_path,
        )
        gap.recommended_rpm_confidence_interval = (
            ci_result.lower_rpm,
            ci_result.upper_rpm,
        )
        gap.prediction_confidence = ci_result.confidence
        gap.cautionary_cooling_state = ci_result.cautionary_cooling

        if ci_result.cautionary_cooling:
            # Override to safe higher-RPM baseline (mechanical longevity may clamp to slew rate)
            # BUT respect large thermal margins: when temp is far below target, the optimizer's
            # reduction is already well-justified — overriding it wastes energy.
            thermal_margin_for_caution = self.target_temp - current_thermal
            if thermal_margin_for_caution > 15:
                # Large margin: skip cautionary override entirely — let the optimizer's
                # recommendation (likely a reduction) flow through guardrails/longevity as normal.
                # The 15+ °C thermal buffer IS the safety net; cautionary +15% would only
                # fight the optimizer and prevent the brain from learning efficient operation.
                gap.cautionary_cooling_state = False
                gap.reasoning_log.append(
                    f"Cautionary Cooling skipped: confidence {ci_result.confidence:.1%} < "
                    f"{CAUTIONARY_CONFIDENCE_THRESHOLD:.0%} but {thermal_margin_for_caution:.0f}°C margin "
                    f"provides ample safety — allowing optimizer recommendation ({gap.recommended_cooling_delta:+.1%})."
                )
                gap.raw_metrics["cautionary_cooling_reason"] = "skipped_large_margin"
            else:
                gap.recommended_cooling_delta = CAUTIONARY_BASELINE_DELTA
                gap.reasoning_log.append(
                    f"Cautionary Cooling: Prediction confidence {ci_result.confidence:.1%} < {CAUTIONARY_CONFIDENCE_THRESHOLD:.0%} threshold. "
                    f"Defaulting to safe higher-RPM baseline (+{CAUTIONARY_BASELINE_DELTA*100:.0f}%) until confident."
                )
                gap.recommendations.insert(
                    0,
                    f"⚠️ Cautionary Cooling: Low confidence ({ci_result.confidence:.1%}). Using safe baseline (+{CAUTIONARY_BASELINE_DELTA*100:.0f}% RPM).",
                )
                gap.raw_metrics["cautionary_cooling_reason"] = "confidence_below_threshold"
            gap.raw_metrics["cautionary_thermal_margin"] = thermal_margin_for_caution

        # Apply guardrails
        gap = apply_guardrails(
            gap,
            current_thermal=current_thermal,
            current_cooling=current_cooling,
            power_draw=current_power,
            max_cooling=max_cooling,
            sustained_active_compute=sustained_active,
            spike_hold_min_rpm=spike_floor_rpm,
            rated_max_fan_rpm=rated_max_fan_rpm,
            calibration_profile=cp,
        )

        # Reward function (for RL / grading)
        # energy_saved_ratio: fraction of cooling power saved vs. baseline.
        # Use power_savings from Fan Affinity optimizer when available;
        # otherwise derive from over-provisioning (higher OP = more room to save).
        if gap.cooling_power_w > 0 and gap.power_savings_w > 0:
            energy_saved = min(1.0, gap.power_savings_w / max(gap.cooling_power_w, 1.0))
        else:
            # Over-provisioning IS the savings opportunity: higher OP = more we
            # can save.  Previous formula (1 - OP) rewarded the status quo.
            energy_saved = min(1.0, over_provisioning)
        thermal_stab = thermal_stability_from_temp(current_thermal, self.target_temp)
        wear_penalty = mechanical_wear_penalty(
            oscillation, gap.slew_rate_limited, gap.anti_short_cycle_hold
        )
        reward_result = compute_reward(
            energy_saved, thermal_stab, wear_penalty, gap.guardrail_triggered
        )
        gap.reward = reward_result.total_reward
        gap.reward_result = reward_result

        # --- SPIKE BYPASS: when peak power is a true spike (>2× trigger), bypass slew rate ---
        # The slew rate limit exists to prevent oscillation during gradual ramps — it should
        # not apply to sudden spike events where fans need to jump immediately.
        if (
            cp is not None
            and peak_power_3s > active_trigger_w * 2.0
            and not _current_is_idle
            and cp.spike_hold_fan_floor_rpm > 0
        ):
            # Compute delta to jump current_cooling directly to spike_hold_floor
            spike_floor = cp.spike_hold_fan_floor_rpm
            if max_cooling > 0 and current_cooling < spike_floor:
                spike_delta = (spike_floor - current_cooling) / max_cooling
                if spike_delta > gap.recommended_cooling_delta:
                    gap.recommended_cooling_delta = spike_delta
                    gap.spike_bypass = True
                    _reasoning_logger.info(
                        "[BRAIN] spike_bypass: peak=%.0fW target_rpm=%.0f",
                        peak_power_3s, spike_floor,
                    )
                    gap.reasoning_log.append(
                        f"Spike bypass: peak power {peak_power_3s:.0f}W > 2× trigger "
                        f"{active_trigger_w:.0f}W — jumping to spike_hold_floor "
                        f"{spike_floor:.0f} RPM (bypassing slew rate)."
                    )

        # --- CPU THERMAL GUARD: raise fan floor when CPU exceeds safe ceiling ---
        if (
            current_cpu_temp_c is not None
            and cp is not None
            and hasattr(cp, "cpu_safe_ceiling_c")
            and cp.cpu_safe_ceiling_c > 0
            and current_cpu_temp_c > cp.cpu_safe_ceiling_c
        ):
            cpu_floor = cp.active_compute_fan_floor_rpm
            current_floor = gap.raw_metrics.get("policy_soft_floor_rpm_target", 0)
            new_floor = max(float(current_floor), cpu_floor)
            gap.raw_metrics["policy_soft_floor_rpm_target"] = new_floor
            if current_cooling > 1e-6 and current_cooling < new_floor:
                gap.recommended_cooling_delta = max(
                    gap.recommended_cooling_delta,
                    (new_floor - current_cooling) / max_cooling,
                )
            _reasoning_logger.info(
                "[BRAIN] cpu_thermal_guard: cpu=%.1f°C ceiling=%.1f°C floor_raised_to=%.0f",
                current_cpu_temp_c, cp.cpu_safe_ceiling_c, new_floor,
            )
            gap.reasoning_log.append(
                f"CPU thermal guard: CPU {current_cpu_temp_c:.1f}°C > ceiling "
                f"{cp.cpu_safe_ceiling_c:.1f}°C — fan floor raised to {new_floor:.0f} RPM."
            )

        # Mechanical Longevity with DYNAMIC slew rate (current_thermal for adaptive aggression)
        gap = apply_mechanical_longevity(
            gap,
            current_cooling=current_cooling,
            max_cooling=max_cooling,
            layer=self.mechanical_layer,
            unit_id="default",
            current_thermal=current_thermal,
            max_safe_temp=MAX_SAFE_TEMP,
        )

        # Equipment Safety middleware: min runtime, 5-min delta cap, hysteresis (before hardware)
        if self.equipment_safety is not None:
            gap = self.equipment_safety.apply_to_gap(
                gap,
                current_cooling=current_cooling,
                max_cooling=max_cooling,
                unit_id="default",
            )

        # Diagnostic reasoning: predictive explanation using rate of change and influence map
        gap.diagnostic_reasoning = self._generate_diagnostic_reasoning(
            gap, nodes, current_thermal, current_cooling
        )

        # Failing components: reduce load on flagged units, redistribute to healthy (InfluenceMap)
        # When cautionary_cooling_state, use slightly higher delta for healthy units (more conservative)
        if failing_component_findings:
            self._apply_failing_component_redistribution(
                gap, nodes, influence_map, failing_component_findings
            )
        # If cautionary and we have per-unit deltas, boost healthy units slightly (more conservative)
        if gap.cautionary_cooling_state and gap.recommended_cooling_delta_by_unit:
            for uid in gap.recommended_cooling_delta_by_unit:
                d = gap.recommended_cooling_delta_by_unit[uid]
                if d > 0:  # Healthy units getting increase
                    gap.recommended_cooling_delta_by_unit[uid] = min(0.25, d + 0.05)

        # Pre-cooling: if a rack is about to get a job in < 60s, boost fan RPM by 20% for those zones
        if upcoming_jobs:
            self._apply_precooling(
                gap,
                upcoming_jobs=upcoming_jobs,
                influence_map=influence_map,
                precool_delta=0.20,
                max_seconds=60.0,
            )

        # Policy floor must survive slew / hysteresis / equipment caps (see enforce_policy_floor_after_all_layers).
        gap = enforce_policy_floor_after_all_layers(gap, current_cooling)

        gap.stagger_delays = get_staggered_start_delays(len(nodes) if nodes else 1)

        # FIX G: Track no-response events (cycles where delta ≈ 0 despite active compute)
        no_response_event = (
            abs(gap.recommended_cooling_delta) < 1e-6
            and sustained_active
            and not gap.emergency_mode
        )
        if no_response_event:
            self._no_response_count += 1
        else:
            self._no_response_count = 0
        gap.raw_metrics["no_response_event"] = no_response_event
        gap.raw_metrics["no_response_streak"] = self._no_response_count

        # Build and log Reasoning String (Glass Box audit)
        gap.reasoning_string = " | ".join(gap.reasoning_log) if gap.reasoning_log else gap.diagnostic_reasoning
        if not gap.reasoning_string:
            gap.reasoning_string = gap.diagnostic_reasoning or "No action; efficiency within target."
        gap.raw_metrics["reasoning_string"] = gap.reasoning_string
        _reasoning_logger.info(
            "Reasoning: %s",
            gap.reasoning_string,
            extra={"recommended_delta": gap.recommended_cooling_delta},
        )

        return gap

    def record_for_learning(
        self,
        predicted_temp: float,
        actual_temp: float,
        gap: EfficiencyGap,
        current_thermal: float,
        power_draw: float,
        cooling_output: float,
    ) -> None:
        """Record predicted vs actual temp for feedback loop.

        Call after applying command and measuring actual result.
        Used by tune_model() to adjust aggression coefficients.

        Args:
            predicted_temp: Predicted temperature before action.
            actual_temp: Measured temperature after action.
            gap: EfficiencyGap with recommended_cooling_delta.
            current_thermal: Thermal before action.
            power_draw: Power at time of action.
            cooling_output: Cooling at time of action.
        """
        log_prediction(
            predicted_temp=predicted_temp,
            actual_temp=actual_temp,
            recommended_delta=gap.recommended_cooling_delta,
            current_thermal=current_thermal,
            power_draw=power_draw,
            cooling_output=cooling_output,
            log_path=self.learning_log_path,
        )

    def tune_model(self, min_entries: int = 10) -> dict:
        """Analyze learning log and tune aggression coefficients.

        Compares predicted vs actual temp; if mean_error > 0.5 we under-predicted
        (actual hotter) → increase aggression; if < -0.5 we over-predicted →
        reduce aggression.

        Args:
            min_entries: Minimum log entries required for tuning.

        Returns:
            Dict with aggression_base, aggression_near_limit, mae, tuned, etc.
        """
        return tune_model(self.learning_log_path, min_entries)
