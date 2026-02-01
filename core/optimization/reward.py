"""
CooledAI Reward Function - Multi-Objective RL Logic

Formal reward function for optimization grading:
Reward = (Energy_Saved * 0.4) + (Thermal_Stability * 0.4) - (Mechanical_Wear_Penalty * 0.2)

Massive negative reward when Guardrail triggered - learn to stay away from safety limits.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardResult:
    """Result of reward computation."""
    total_reward: float
    energy_saved_component: float
    thermal_stability_component: float
    mechanical_wear_penalty: float
    guardrail_penalty: float
    raw_components: dict = field(default_factory=dict)


# Reward weights (sum to 1.0)
WEIGHT_ENERGY_SAVED = 0.4
WEIGHT_THERMAL_STABILITY = 0.4
WEIGHT_MECHANICAL_WEAR = 0.2

# Massive penalty when guardrail triggered - dominates other terms
GUARDRAIL_PENALTY = -1000.0

# Thermal stability: 0 = poor (temp far from target), 1 = perfect (at target)
# Penalize distance from target_temp
TEMP_STABILITY_SCALE = 0.1  # Per °C from target

# Mechanical wear: oscillation, slew rate changes, anti-short-cycle holds
WEAR_OSCILLATION_WEIGHT = 50.0
WEAR_SLEW_WEIGHT = 10.0


def compute_reward(
    energy_saved_ratio: float,
    thermal_stability: float,
    mechanical_wear_penalty: float,
    guardrail_triggered: bool = False,
) -> RewardResult:
    """
    Multi-objective reward function.
    
    Args:
        energy_saved_ratio: 0-1, fraction of energy saved vs baseline
        thermal_stability: 0-1, how stable temp is (1 = perfect)
        mechanical_wear_penalty: 0-1, penalty for oscillation/slew
        guardrail_triggered: If True, apply massive negative reward
    """
    guardrail_penalty = GUARDRAIL_PENALTY if guardrail_triggered else 0.0

    energy_component = energy_saved_ratio * WEIGHT_ENERGY_SAVED
    thermal_component = thermal_stability * WEIGHT_THERMAL_STABILITY
    wear_component = -mechanical_wear_penalty * WEIGHT_MECHANICAL_WEAR

    total = energy_component + thermal_component + wear_component + guardrail_penalty

    return RewardResult(
        total_reward=total,
        energy_saved_component=energy_component,
        thermal_stability_component=thermal_component,
        mechanical_wear_penalty=wear_component,
        guardrail_penalty=guardrail_penalty,
        raw_components={
            "energy_saved_ratio": energy_saved_ratio,
            "thermal_stability": thermal_stability,
            "mechanical_wear_penalty": mechanical_wear_penalty,
        },
    )


def thermal_stability_from_temp(
    current_temp: float,
    target_temp: float,
    max_safe_temp: float = 80.0,
) -> float:
    """
    Thermal stability score 0-1.
    1 = at target, 0 = at max_safe or far from target.
    """
    if current_temp >= max_safe_temp:
        return 0.0
    dist_from_target = abs(current_temp - target_temp)
    # Linear decay: 0°C from target = 1.0, 15°C = 0.0
    margin = max_safe_temp - target_temp
    return max(0, 1.0 - dist_from_target / max(margin * 0.5, 5.0))


def mechanical_wear_penalty(
    oscillation_ratio: float,
    slew_rate_limited: bool,
    anti_short_cycle_hold: bool,
) -> float:
    """Penalty 0-1 for mechanical wear (oscillation, slew, cycling)."""
    penalty = oscillation_ratio * WEAR_OSCILLATION_WEIGHT / 100.0
    if slew_rate_limited:
        penalty += 0.1
    if anti_short_cycle_hold:
        penalty += 0.05
    return min(1.0, penalty)
