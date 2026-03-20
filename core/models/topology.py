"""
CooledAI Topology - Rack Model with ThermalMass and FOPDT

Rack model includes ThermalMass (m·c_p) and First-Order Plus Dead Time (FOPDT)
parameters that account for the time delay between Fan Speed Increase and
Temperature Decrease.

Self-calibration from telemetry fits tau, theta, and gain per rack over time,
solving the variance problem across different hardware and facility layouts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default storage for calibrated FOPDT params
DEFAULT_RACK_CALIBRATION_PATH = "data/rack_fopdt_calibration.json"


@dataclass
class FOPDTParams:
    """
    First-Order Plus Dead Time (FOPDT) model parameters.

    Transfer function: τ·dT/dt = -T + T_ss where T_ss = f(power, cooling).
    Step response: T(t) = T_ss + (T_0 - T_ss)·exp(-(t-θ)/τ) for t > θ.

    Attributes:
        tau: Time constant (s) - thermal inertia.
        theta: Dead time (s) - delay between fan change and temp response.
        gain: K (°C per unit cooling) - negative: more cooling -> lower temp.
        thermal_mass: m·c_p (J/K) - related to tau via tau = thermal_mass / (h·A).
    """
    tau: float = 120.0      # Time constant (s)
    theta: float = 15.0     # Dead time (s)
    gain: float = -0.01     # °C per RPM (negative for cooling)
    thermal_mass: float = 150_000.0  # J/K (~150 kg * 1000 J/(kg·K))

    # Calibration metadata
    calibrated_at: Optional[str] = None
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau": self.tau,
            "theta": self.theta,
            "gain": self.gain,
            "thermal_mass": self.thermal_mass,
            "calibrated_at": self.calibrated_at,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FOPDTParams":
        return cls(
            tau=float(d.get("tau", 120.0)),
            theta=float(d.get("theta", 15.0)),
            gain=float(d.get("gain", -0.01)),
            thermal_mass=float(d.get("thermal_mass", 150_000.0)),
            calibrated_at=d.get("calibrated_at"),
            n_samples=int(d.get("n_samples", 0)),
        )


@dataclass
class Rack:
    """
    Rack model with ThermalMass and FOPDT dynamics.

    Attributes:
        rack_id: Unique identifier (e.g. rack_01).
        thermal_mass: m·c_p (J/K) - heat capacity.
        fopdt: FOPDT parameters (tau, theta, gain).
    """
    rack_id: str
    thermal_mass: float = 150_000.0  # J/K
    fopdt: FOPDTParams = field(default_factory=FOPDTParams)

    @property
    def tau(self) -> float:
        return self.fopdt.tau

    @property
    def theta(self) -> float:
        return self.fopdt.theta

    @property
    def gain(self) -> float:
        return self.fopdt.gain

    def predict_step(
        self,
        t: float,
        t_step: float,
        T_before: float,
        cooling_before: float,
        cooling_after: float,
        power_w: float,
        T_supply: float = 18.0,
    ) -> float:
        """
        FOPDT step response: T(t) after cooling step at t_step.

        For t <= t_step + theta: T unchanged (dead time).
        For t > t_step + theta: T(t) = T_ss + (T_0 - T_ss)·exp(-(t - t_step - theta)/tau).

        T_ss = steady-state temp for (power, cooling_after).
        Simplified: T_ss ~ T_supply + power/(k*cooling) or use gain.
        """
        dt_eff = t - t_step - self.theta
        if dt_eff <= 0:
            return T_before

        # Steady-state shift from cooling change: delta_T_ss = gain * (cooling_after - cooling_before)
        delta_cooling = cooling_after - cooling_before
        delta_T_ss = self.gain * delta_cooling
        T_ss = T_before + delta_T_ss

        # First-order response
        T_pred = T_ss + (T_before - T_ss) * np.exp(-dt_eff / self.tau)
        return float(T_pred)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rack_id": self.rack_id,
            "thermal_mass": self.thermal_mass,
            "fopdt": self.fopdt.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rack":
        return cls(
            rack_id=d.get("rack_id", "unknown"),
            thermal_mass=float(d.get("thermal_mass", 150_000.0)),
            fopdt=FOPDTParams.from_dict(d.get("fopdt", {})),
        )


class RackRegistry:
    """
    Registry of Rack models with per-rack FOPDT calibration.
    Self-calibrates tau, theta, gain from telemetry over time.
    """

    def __init__(self, calibration_path: Optional[Path] = None):
        self._calibration_path = Path(calibration_path or DEFAULT_RACK_CALIBRATION_PATH)
        self._racks: Dict[str, Rack] = {}
        self._telemetry_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._load_calibration()

    def _load_calibration(self) -> None:
        if not self._calibration_path.exists():
            return
        try:
            import json
            with open(self._calibration_path, "r") as f:
                data = json.load(f)
            for r in data.get("racks", []):
                rack = Rack.from_dict(r)
                self._racks[rack.rack_id] = rack
        except Exception:
            pass

    def _save_calibration(self) -> None:
        self._calibration_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json
            from datetime import datetime
            data = {
                "racks": [r.to_dict() for r in self._racks.values()],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            with open(self._calibration_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_rack(self, rack_id: str) -> Rack:
        if rack_id not in self._racks:
            try:
                from core.config import get_rack_thermal_mass
                thermal_mass_map = get_rack_thermal_mass()
                thermal_mass = thermal_mass_map.get(rack_id, 150_000.0)
            except Exception:
                thermal_mass = 150_000.0
            self._racks[rack_id] = Rack(rack_id=rack_id, thermal_mass=thermal_mass)
        return self._racks[rack_id]

    def get_cooling_history(
        self,
        rack_id: str,
        max_samples: int = 100,
    ) -> List[Tuple[float, float]]:
        """Get cooling history [(t, rpm), ...] for FOPDT prediction."""
        buf = self._telemetry_buffer.get(rack_id, [])
        recent = buf[-max_samples:]
        return [(x["t"], x["cooling"]) for x in recent]

    def add_telemetry(
        self,
        rack_id: str,
        timestamp: float,
        temperature: float,
        cooling_rpm: float,
        power_w: float,
    ) -> None:
        """Add telemetry sample for self-calibration."""
        if rack_id not in self._telemetry_buffer:
            self._telemetry_buffer[rack_id] = []
        self._telemetry_buffer[rack_id].append({
            "t": timestamp,
            "T": temperature,
            "cooling": cooling_rpm,
            "power": power_w,
        })
        # Keep last 500 samples per rack
        self._telemetry_buffer[rack_id] = self._telemetry_buffer[rack_id][-500:]

    def calibrate_rack(self, rack_id: str, min_samples: int = 30) -> bool:
        """
        Fit FOPDT (tau, theta, gain) from telemetry using step-response method.

        This is **physics identification only** (step-response fit on temperature vs
        cooling changes). It does **not** optimize for fan noise, energy, or a reward
        function—those come from the efficiency / sweet-spot path in the brain.
        If fans drift too low under load, use `cooling_safety_policy` floors in
        `optimization_brain.apply_guardrails`, not changes here.

        Looks for fan step changes and fits:
        - theta: time from step to first significant temp change
        - tau: from 63.2% of final response
        - gain: delta_T / delta_cooling
        """
        buf = self._telemetry_buffer.get(rack_id, [])
        if len(buf) < min_samples:
            return False

        t_arr = np.array([x["t"] for x in buf])
        T_arr = np.array([x["T"] for x in buf])
        cooling_arr = np.array([x["cooling"] for x in buf])
        power_arr = np.array([x["power"] for x in buf])

        # Detect cooling steps: |d(cooling)| > threshold
        cooling_diff = np.abs(np.diff(cooling_arr, prepend=cooling_arr[0]))
        step_threshold = np.median(cooling_arr) * 0.05
        step_idx = np.where(cooling_diff > max(step_threshold, 50))[0]

        if len(step_idx) < 2:
            return False

        tau_est = []
        theta_est = []
        gain_est = []

        for i in step_idx:
            if i + 20 >= len(buf):
                continue
            t_step = t_arr[i]
            cooling_before = cooling_arr[i - 1] if i > 0 else cooling_arr[i]
            cooling_after = cooling_arr[i]
            delta_cooling = cooling_after - cooling_before
            if abs(delta_cooling) < 10:
                continue

            T_before = T_arr[i]
            win = slice(i, min(i + 60, len(T_arr)))
            T_win = T_arr[win]
            t_win = t_arr[win]

            # Theta: time until T departs from T_before by > 0.2°C
            T_depart = np.where(np.abs(T_win - T_before) > 0.2)[0]
            if len(T_depart) == 0:
                continue
            theta = (t_win[T_depart[0]] - t_step) if T_depart[0] > 0 else 0.0
            theta = max(0.0, min(60.0, theta))
            theta_est.append(theta)

            # Final delta_T (after ~3*tau)
            T_final = T_win[-1] if len(T_win) > 10 else T_before
            delta_T = T_final - T_before
            gain = delta_T / delta_cooling if abs(delta_cooling) > 1e-6 else 0.0
            if -0.1 < gain < 0:
                gain_est.append(gain)

            # Tau: 63.2% of response
            target_T = T_before + 0.632 * (T_final - T_before)
            if abs(T_final - T_before) < 0.1:
                continue
            candidates = np.where(
                (T_win >= target_T - 0.1) & (T_win <= target_T + 0.1)
            )[0]
            if len(candidates) > 0:
                j = candidates[0]
                tau = (t_win[j] - t_step - theta) if j > 0 else 30.0
                tau = max(10.0, min(600.0, tau))
                tau_est.append(tau)

        if not tau_est or not theta_est:
            return False

        rack = self.get_rack(rack_id)
        rack.fopdt.tau = float(np.median(tau_est))
        rack.fopdt.theta = float(np.median(theta_est))
        rack.fopdt.gain = float(np.median(gain_est)) if gain_est else -0.01
        rack.fopdt.thermal_mass = rack.thermal_mass
        rack.fopdt.n_samples = len(buf)
        from datetime import datetime
        rack.fopdt.calibrated_at = datetime.utcnow().isoformat() + "Z"

        self._racks[rack_id] = rack
        self._save_calibration()
        return True

    def predict_fopdt(
        self,
        rack_id: str,
        t_now: float,
        t_pred: float,
        T_current: float,
        cooling_history: List[Tuple[float, float]],
        power_w: float,
    ) -> float:
        """
        Predict T(t_pred) using FOPDT with fan/cooling history via superposition.

        cooling_history: [(t, rpm), ...] - time-ordered.
        Accounts for dead time: fan change at t affects T only after t+theta.

        Uses superposition principle: each cooling step contributes an
        independent delta-T to the baseline. This avoids the error of
        cascading T_pred from one step as T_before of the next.
        """
        rack = self.get_rack(rack_id)
        if not cooling_history:
            return T_current

        cooling_history = sorted(cooling_history, key=lambda x: x[0])
        cooling_prev = cooling_history[0][1]

        # Superposition: accumulate independent step-response deltas
        delta_T_total = 0.0
        for t_c, cooling in cooling_history:
            if t_c >= t_pred:
                break
            if abs(cooling - cooling_prev) < 1e-6:
                continue

            # Effective time since step, accounting for dead time
            dt_eff = t_pred - t_c - rack.theta
            if dt_eff > 0:
                # Steady-state delta from this step
                delta_cooling = cooling - cooling_prev
                delta_T_ss = rack.gain * delta_cooling
                # First-order decay toward steady state
                delta_T_total += delta_T_ss * (1.0 - np.exp(-dt_eff / rack.tau))

            cooling_prev = cooling

        return T_current + delta_T_total


_default_registry: Optional[RackRegistry] = None


def get_default_registry() -> RackRegistry:
    """Singleton registry for topology."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RackRegistry()
    return _default_registry
