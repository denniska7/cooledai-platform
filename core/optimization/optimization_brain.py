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
from core.optimization.reward import (
    compute_reward,
    thermal_stability_from_temp,
    mechanical_wear_penalty,
    RewardResult,
)
from core.optimization.learning_log import log_prediction, tune_model, DEFAULT_LEARNING_LOG

try:
    from backend.safety.guardrails import snap_rpm_to_safe_boundary
except ImportError:
    def snap_rpm_to_safe_boundary(rpm: float, resonance_zones=None):  # noqa: D103
        return rpm  # no-op if backend not available

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
    stagger_delays: List[float] = field(default_factory=list)  # Per-unit start delays (s)

    # Predictive & reward
    predicted_temp_t10: Optional[float] = None
    thermal_inertia_s: float = 120.0
    reward: Optional[float] = None
    reward_result: Optional[Any] = None

    # Human-readable recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Raw metrics for debugging
    raw_metrics: dict = field(default_factory=dict)


def apply_guardrails(
    gap: EfficiencyGap,
    current_thermal: float,
    current_cooling: float,
    power_draw: float,
    max_cooling: float = 3000.0,
) -> EfficiencyGap:
    """
    Guardrail wrapper: intercepts AI recommendations that would violate safety.
    
    ASHRAE-based safety constants:
    - MAX_SAFE_TEMP: 80°C - override if AI would allow temp above limit
    - MIN_FAN_RPM: 400 - override if AI would turn fan off while under load
    
    When a guardrail is hit: sets emergency_mode=True, recommended_cooling_delta=1.0
    (100% cooling), and logs a Critical Safety Event.
    
    Args:
        gap: EfficiencyGap from AI analysis
        current_thermal: Current temperature (°C)
        current_cooling: Current cooling output (RPM or flow rate)
        power_draw: Current power draw (W) - used to detect "under load"
        max_cooling: Maximum cooling capacity (for 100% = Emergency Mode)
        
    Returns:
        EfficiencyGap with guardrails applied (may be modified)
    """
    # Compute what the AI's recommendation would result in
    # recommended_cooling_delta: -0.1 = reduce 10%, 0 = no change, 1.0 = max
    proposed_cooling = current_cooling * (1.0 + gap.recommended_cooling_delta)
    
    # Mechanical Resonance Guard: snap RPM if in danger zone (vibration protection)
    proposed_safe = snap_rpm_to_safe_boundary(proposed_cooling)
    if abs(proposed_safe - proposed_cooling) > 0.5 and current_cooling > 0:
        gap.recommended_cooling_delta = (proposed_safe / current_cooling) - 1.0
        gap.recommendations.append(
            f"Resonance Guard: Snapped RPM from {proposed_cooling:.0f} to {proposed_safe:.0f} (avoid vibration zone)."
        )
        proposed_cooling = proposed_safe
    
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


# --- Mechanical Longevity Layer ---

# Threshold: cooling below this fraction of max = "off"
_COOLING_OFF_THRESHOLD = 0.05  # 5% of max


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
        """Unit is 'on' if cooling > 5% of max."""
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
        """
        If AI tries to toggle unit state too quickly, force hold until timer expires.
        """
        if gap.emergency_mode:
            return gap  # Emergency bypasses anti-short-cycle

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

        return gap

    def get_dynamic_slew_rate(
        self,
        current_thermal: float,
        max_safe_temp: float = 80.0,
    ) -> float:
        """
        Dynamic slew rate: gentle when far from limit, aggressive when approaching spike.
        Far from limit -> save motor wear (0.05). Approaching spike -> prioritize safety (0.25).
        """
        margin = max_safe_temp - current_thermal
        if margin <= 3.0:
            return 0.25  # Aggressive when close to limit
        if margin <= 10.0:
            return 0.15  # Moderate
        return 0.05  # Gentle when far (save wear)

    def apply_slew_rate_limit(
        self,
        gap: EfficiencyGap,
        current_cooling: float,
        max_cooling: float,
        current_thermal: Optional[float] = None,
        max_safe_temp: float = 80.0,
    ) -> EfficiencyGap:
        """
        Limit cooling change. Uses dynamic slew rate when current_thermal provided.
        Gentle when far from MAX_SAFE_TEMP, aggressive when approaching spike.
        """
        if gap.emergency_mode:
            return gap  # Emergency bypasses slew limit

        rate = self.max_change_rate
        if current_thermal is not None:
            rate = self.get_dynamic_slew_rate(current_thermal, max_safe_temp)

        delta = gap.recommended_cooling_delta
        clamped = max(-rate, min(rate, delta))
        if abs(clamped - delta) > 1e-9:
            gap.slew_rate_limited = True
            gap.recommended_cooling_delta = clamped
            gap.recommendations.insert(
                0,
                f"⚙️ Slew rate limit: Delta clamped from {delta:.2f} to ±{rate:.2f} "
                "(dynamic: gentle when far, aggressive when approaching limit).",
            )
        return gap

    def get_staggered_start_delays(self, n_units: int) -> List[float]:
        """
        Return staggered start delays (seconds) for n units to prevent inrush spikes.
        Unit 0: 0s, Unit 1: 2s, Unit 2: 4s, ...
        """
        return [i * self.stagger_delay for i in range(n_units)]


def apply_mechanical_longevity(
    gap: EfficiencyGap,
    current_cooling: float,
    max_cooling: float = 3000.0,
    layer: Optional[MechanicalLongevityLayer] = None,
    unit_id: str = "default",
    current_thermal: Optional[float] = None,
    max_safe_temp: float = MAX_SAFE_TEMP,
) -> EfficiencyGap:
    """
    Apply Mechanical Longevity Layer: anti-short-cycle, slew rate limit, stagger.
    Dynamic slew rate when current_thermal provided.
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

    return gap


def get_staggered_start_delays(n_units_starting: int) -> List[float]:
    """
    Power surge protection: 2-second stagger between unit starts.
    Returns [0, 2, 4, ...] seconds delay for each unit.
    """
    return [i * STAGGER_START_DELAY for i in range(n_units_starting)]


class OptimizationBrain:
    """
    Advanced predictive optimization engine. Multi-variate analysis,
    thermal inertia, T+10s prediction, reward function, dynamic slew rate.
    """

    def __init__(
        self,
        target_temp: float = 65.0,
        lag_threshold_seconds: float = 5.0,
        over_provision_threshold: float = 0.15,
        oscillation_threshold: float = 0.2,
        mechanical_longevity_layer: Optional[MechanicalLongevityLayer] = None,
        predictor: Optional[TempPredictor] = None,
        learning_log_path: str = DEFAULT_LEARNING_LOG,
    ):
        self.target_temp = target_temp
        self.lag_threshold = lag_threshold_seconds
        self.over_provision_threshold = over_provision_threshold
        self.oscillation_threshold = oscillation_threshold
        self.mechanical_layer = mechanical_longevity_layer or MechanicalLongevityLayer()
        self.predictor = predictor or TempPredictor()
        self.learning_log_path = learning_log_path
    
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
        """
        Thermal Lag: Time (samples) between power peak and cooling peak.
        Positive = cooling lags power (reactive, bad).
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
        """
        Over-provisioning: Cooling more than needed for current load.
        Thermal inertia: if load just dropped, allow brief over-cooling (hardware holds heat).
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
        """
        Overall efficiency score (0-100).
        Penalizes lag, over-provisioning, and oscillation.
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
    
    def analyze(self, nodes: List[BaseNode]) -> EfficiencyGap:
        """
        Analyze normalized node data and compute Efficiency Gap.
        Uses predictive T+10s, thermal inertia, reward function, dynamic slew rate.
        """
        if not nodes:
            return EfficiencyGap(
                recommendations=["No data to analyze. Check data source."],
            )

        thermal, power, cooling, utilization = self._nodes_to_arrays(nodes)
        current_thermal = float(thermal[-1]) if len(thermal) > 0 else 0.0
        current_cooling = float(cooling[-1]) if len(cooling) > 0 else 0.0
        current_power = float(power[-1]) if len(power) > 0 else 0.0
        max_cooling = float(np.max(cooling)) if len(cooling) > 0 else 3000.0
        max_cooling = max(max_cooling, 1000.0)

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
        gap.raw_metrics = {
            "thermal_mean": float(np.mean(thermal)),
            "power_mean": float(np.mean(power)),
            "cooling_mean": float(np.mean(cooling)),
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

        # Recommended delta based on PREDICTED state (look-ahead), not just current
        # Use predicted T+10 to be proactive
        temp_for_decision = predicted_t10 if pred and pred.confidence > 0.6 else current_thermal

        if over_provisioning > self.over_provision_threshold:
            # Reduce cooling - but consider thermal inertia (don't over-cool during idle)
            gap.recommended_cooling_delta = -0.08  # Slightly less aggressive (was -0.1)
        elif temp_for_decision > self.target_temp + 5:
            # Predicted/current temp rising - ramp cooling proactively
            gap.recommended_cooling_delta = 0.12  # Proactive increase (was 0)
        elif oscillation > self.oscillation_threshold:
            gap.recommended_cooling_delta = 0.0
        else:
            gap.recommended_cooling_delta = 0.0

        gap.efficiency_score = self._compute_efficiency_score(gap)
        gap.recommendations = self._generate_recommendations(gap)

        # Apply guardrails
        gap = apply_guardrails(
            gap,
            current_thermal=current_thermal,
            current_cooling=current_cooling,
            power_draw=current_power,
            max_cooling=max_cooling,
        )

        # Reward function (for RL / grading)
        energy_saved = 1.0 - (over_provisioning if over_provisioning < 1 else 1.0)
        thermal_stab = thermal_stability_from_temp(current_thermal, self.target_temp)
        wear_penalty = mechanical_wear_penalty(
            oscillation, gap.slew_rate_limited, gap.anti_short_cycle_hold
        )
        reward_result = compute_reward(
            energy_saved, thermal_stab, wear_penalty, gap.guardrail_triggered
        )
        gap.reward = reward_result.total_reward
        gap.reward_result = reward_result

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

        gap.stagger_delays = get_staggered_start_delays(len(nodes) if nodes else 1)

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
        """
        Record predicted vs actual temp for feedback loop.
        Call after applying command and measuring actual result.
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
        """
        Analyze learning log and adjust Aggression Coefficients.
        Returns tuned coefficients for use in future runs.
        """
        return tune_model(self.learning_log_path, min_entries)
