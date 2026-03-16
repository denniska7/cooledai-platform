"""
CooledAI Digital Twin - Newton's Law of Cooling for 10-Rack Room

Physics engine that simulates rack temperatures from fan RPM input.
Designed for stress testing OptimizationBrain and RL agents without hardware.

Physics: dT/dt = Q/(m*c_p) - k*RPM*(T - T_supply)
- Q: heat load (W) from IT equipment
- m*c_p: thermal mass (J/K)
- k*RPM: cooling coefficient (Newton's Law with forced convection)
- T_supply: supply air temperature
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.hal.base_node import BaseNode, ServerNode


@dataclass
class DigitalTwinConfig:
    """Configuration for the digital twin physics engine."""

    num_racks: int = 10
    supply_temp_c: float = 18.0
    thermal_mass_j_per_k: float = 150_000.0  # ~150 kg * 1000 J/(kg·K) per rack
    cooling_coefficient: float = 1.2e-5  # k in dT/dt = Q/(m*c) - k*RPM*(T-T_supply)
    rpm_base: float = 2000.0  # Normalization: 1.0 delta ≈ 2000 RPM
    noise_std_c: float = 0.5  # Gaussian noise on temperature output (°C)
    power_range_w: tuple = (20_000.0, 80_000.0)  # Min/max power per rack
    initial_temp_range_c: tuple = (35.0, 55.0)  # Initial temperature range
    # Rack ID -> cooling unit IDs (which CRACs serve each rack)
    rack_to_cooling: Optional[Dict[str, List[str]]] = None


class DigitalTwin:
    """
    Physics-based digital twin for a 10-rack data center room.

    Newton's Law of Cooling with forced convection:
    dT/dt = Q/(m*c_p) - k * effective_RPM * (T - T_supply)

    Input: Fan RPM per cooling unit (CRAC)
    Output: Rack temperatures with configurable random noise
    """

    # Default: racks 0-4 get CRAC-01, 5-9 get CRAC-02 (with overlap)
    DEFAULT_RACK_TO_COOLING = {
        "rack_01": ["CRAC-01"],
        "rack_02": ["CRAC-01", "CRAC-02"],
        "rack_03": ["CRAC-01", "CRAC-02"],
        "rack_04": ["CRAC-01", "CRAC-02"],
        "rack_05": ["CRAC-01", "CRAC-02"],
        "rack_06": ["CRAC-01", "CRAC-02"],
        "rack_07": ["CRAC-02"],
        "rack_08": ["CRAC-01", "CRAC-02"],
        "rack_09": ["CRAC-01", "CRAC-02"],
        "rack_10": ["CRAC-02"],
    }

    COOLING_UNITS = ["CRAC-01", "CRAC-02"]

    def __init__(self, config: Optional[DigitalTwinConfig] = None, seed: Optional[int] = None):
        self.config = config or DigitalTwinConfig()
        self._rng = np.random.default_rng(seed)

        self.num_racks = self.config.num_racks
        self.rack_ids = [f"rack_{i:02d}" for i in range(1, self.num_racks + 1)]
        self.rack_to_cooling = (
            self.config.rack_to_cooling
            or {r: self.DEFAULT_RACK_TO_COOLING.get(r, self.COOLING_UNITS) for r in self.rack_ids}
        )

        # State
        self._temps: np.ndarray = np.zeros(self.num_racks)
        self._power: np.ndarray = np.zeros(self.num_racks)
        self._fan_rpm: Dict[str, float] = {cu: 2000.0 for cu in self.COOLING_UNITS}
        self._time: float = 0.0

        # Initialize
        low, high = self.config.initial_temp_range_c
        self._temps = self._rng.uniform(low, high, self.num_racks)
        plow, phigh = self.config.power_range_w
        self._power = self._rng.uniform(plow, phigh, self.num_racks)

    def _effective_rpm_for_rack(self, rack_id: str) -> float:
        """Average RPM from CRACs serving this rack."""
        return self._effective_rpm_from_map(rack_id, self._fan_rpm)

    def _effective_rpm_from_map(self, rack_id: str, fan_rpm: Dict[str, float]) -> float:
        """Average RPM from CRACs serving this rack, using given fan_rpm map."""
        units = self.rack_to_cooling.get(rack_id, self.COOLING_UNITS)
        rpms = [fan_rpm.get(u, 2000.0) for u in units]
        return float(np.mean(rpms)) if rpms else 2000.0

    def _compute_dt_dt(self, temps: np.ndarray, power: np.ndarray, fan_rpm: Dict[str, float]) -> np.ndarray:
        """
        Compute dT/dt for each rack using Newton's Law of Cooling.

        dT/dt = Q/(m*c_p) - k * RPM * (T - T_supply)
        """
        mcp = self.config.thermal_mass_j_per_k
        k = self.config.cooling_coefficient
        t_supply = self.config.supply_temp_c

        dtdt = np.zeros(self.num_racks)
        for i, rack_id in enumerate(self.rack_ids):
            rpm = self._effective_rpm_from_map(rack_id, fan_rpm)
            q = power[i]
            t = temps[i]
            heating = q / mcp
            cooling = k * rpm * (t - t_supply)
            dtdt[i] = heating - cooling

        return dtdt

    def step(
        self,
        dt: float = 1.0,
        fan_rpm: Optional[Dict[str, float]] = None,
        power_override: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Advance simulation by dt seconds.

        Args:
            dt: Time step in seconds
            fan_rpm: Cooling unit ID -> RPM. If None, uses current state.
            power_override: Rack ID -> power (W). If None, uses current power.

        Returns:
            New rack temperatures (before noise).
        """
        if fan_rpm is not None:
            self._fan_rpm.update(fan_rpm)

        power = self._power.copy()
        if power_override is not None:
            for rack_id, p in power_override.items():
                idx = self.rack_ids.index(rack_id) if rack_id in self.rack_ids else None
                if idx is not None:
                    power[idx] = p

        dtdt = self._compute_dt_dt(self._temps, power, self._fan_rpm)
        self._temps = self._temps + dtdt * dt
        self._temps = np.clip(self._temps, self.config.supply_temp_c, 95.0)
        self._time += dt

        return self._temps.copy()

    def get_temperatures(self, add_noise: bool = True) -> np.ndarray:
        """Return current rack temperatures, optionally with noise."""
        t = self._temps.copy()
        if add_noise and self.config.noise_std_c > 0:
            t += self._rng.normal(0, self.config.noise_std_c, self.num_racks)
        return np.clip(t, self.config.supply_temp_c, 95.0)

    def get_nodes(self, add_noise: bool = True) -> List[BaseNode]:
        """
        Return List[BaseNode] for OptimizationBrain.analyze().

        Each rack is a ServerNode with thermal_input, power_draw, cooling_output.
        cooling_output = effective fan RPM for that rack (from CRACs).
        """
        temps = self.get_temperatures(add_noise=add_noise)
        nodes: List[BaseNode] = []
        for i, rack_id in enumerate(self.rack_ids):
            rpm = self._effective_rpm_for_rack(rack_id)
            nodes.append(
                ServerNode(
                    thermal_input=float(temps[i]),
                    power_draw=float(self._power[i]),
                    cooling_output=float(rpm),
                    utilization=min(100.0, 100.0 * self._power[i] / 80_000.0),
                    node_id=rack_id,
                    timestamp=datetime.now(),
                    source="digital_twin",
                )
            )
        return nodes

    def set_fan_rpm(self, cooling_unit_id: str, rpm: float) -> None:
        """Set fan RPM for a cooling unit."""
        self._fan_rpm[cooling_unit_id] = max(400.0, min(4000.0, rpm))

    def apply_deltas(
        self,
        recommended_cooling_delta_by_unit: Dict[str, float],
        dt: float = 1.0,
    ) -> List[BaseNode]:
        """
        Apply OptimizationBrain deltas, step simulation, return new nodes.

        Delta is fractional: 0.2 = +20%, -0.1 = -10%.
        Converts to RPM: new_rpm = current_rpm * (1 + delta).

        Args:
            recommended_cooling_delta_by_unit: Cooling unit ID -> delta (-1 to 1).
            dt: Time step for physics step.

        Returns:
            New BaseNode list after applying deltas and stepping.
        """
        new_rpm: Dict[str, float] = {}
        for unit_id, delta in recommended_cooling_delta_by_unit.items():
            current = self._fan_rpm.get(unit_id, 2000.0)
            adjusted = current * (1.0 + float(delta))
            new_rpm[unit_id] = max(400.0, min(4000.0, adjusted))

        if new_rpm:
            self.step(dt=dt, fan_rpm=new_rpm)

        return self.get_nodes()

    def stress_test_loop(
        self,
        brain: Any,
        num_steps: int = 100,
        dt: float = 1.0,
        induce_load_spike_at: Optional[int] = None,
        load_spike_factor: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Run stress test: OptimizationBrain observes nodes, recommends deltas,
        digital twin applies them and steps. Returns metrics.

        Args:
            brain: OptimizationBrain instance
            num_steps: Number of simulation steps
            dt: Time step
            induce_load_spike_at: Step index to spike power (stress test)
            load_spike_factor: Multiply power by this during spike

        Returns:
            Dict with temps_history, efficiency_scores, gap_reasons, etc.
        """
        temps_history: List[List[float]] = []
        efficiency_scores: List[float] = []
        gaps: List[Any] = []

        for step in range(num_steps):
            if induce_load_spike_at is not None and step == induce_load_spike_at:
                for i in range(self.num_racks):
                    self._power[i] *= load_spike_factor

            nodes = self.get_nodes()
            gap = brain.analyze(nodes)
            gaps.append(gap)
            efficiency_scores.append(gap.efficiency_score)

            deltas = getattr(gap, "recommended_cooling_delta_by_unit", None) or {}
            if not deltas:
                deltas = {cu: getattr(gap, "recommended_cooling_delta", 0.0) for cu in self.COOLING_UNITS}

            self.apply_deltas(deltas, dt=dt)
            temps_history.append(self.get_temperatures(add_noise=False).tolist())

        max_temp = max(max(t) for t in temps_history)
        min_score = min(efficiency_scores)
        avg_score = float(np.mean(efficiency_scores))

        return {
            "temps_history": temps_history,
            "efficiency_scores": efficiency_scores,
            "max_temp_reached": max_temp,
            "min_efficiency_score": min_score,
            "avg_efficiency_score": avg_score,
            "num_steps": num_steps,
            "gaps": gaps,
        }

    def get_observation_for_rl(self) -> Dict[str, np.ndarray]:
        """
        Return observation dict for RL agents (e.g. state for Gym env).
        Keys: temperatures, power, fan_rpm, effective_rpm_per_rack.
        """
        temps = self.get_temperatures(add_noise=True)
        effective_rpm = np.array([self._effective_rpm_for_rack(r) for r in self.rack_ids])
        return {
            "temperatures": temps,
            "power": self._power.copy(),
            "fan_rpm": np.array([self._fan_rpm.get(cu, 2000.0) for cu in self.COOLING_UNITS]),
            "effective_rpm_per_rack": effective_rpm,
        }
