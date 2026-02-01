"""
CooledAI Synthetic Sensor - ML-based Missing Data Estimation

When a data source provides incomplete data (e.g., missing fan RPM),
the SyntheticSensor estimates the missing value using:
1. Physics-based heuristic (power/temp correlation)
2. Optional: Trained ML model (when historical data available)

Estimation model: cooling_output ≈ f(power_draw, thermal_input)
- Higher power + higher temp -> higher cooling needed
- Linear regression or simple heuristic as fallback
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from core.hal.base_node import BaseNode


@dataclass
class SyntheticSensorConfig:
    """Configuration for synthetic sensor estimation."""
    # Heuristic: cooling_rpm = base + power_coef * power + temp_coef * (temp - target)
    base_cooling: float = 800.0
    power_coef: float = 8.0   # RPM per Watt
    temp_coef: float = 50.0   # RPM per °C above target
    target_temp: float = 65.0
    min_cooling: float = 500.0
    max_cooling: float = 3000.0


class SyntheticSensor:
    """
    Estimates missing sensor values (e.g., fan RPM) from available data.
    
    Uses physics-based heuristic: cooling scales with power and temp.
    Can be trained on historical data for improved accuracy (future).
    """
    
    def __init__(self, config: Optional[SyntheticSensorConfig] = None):
        """
        Initialize synthetic sensor.
        
        Args:
            config: Estimation parameters (uses defaults if None)
        """
        self.config = config or SyntheticSensorConfig()
        self._model = None  # Placeholder for trained ML model (future)
    
    def estimate_cooling(
        self,
        power_draw: float,
        thermal_input: float,
        utilization: float = 0.0,
    ) -> float:
        """
        Estimate cooling output (fan RPM) from power and temp.
        
        Heuristic: cooling = base + power_coef * power + temp_coef * (temp - target)
        Higher power and higher temp -> more cooling needed.
        
        Args:
            power_draw: Power consumption (W)
            thermal_input: Temperature (°C)
            utilization: Optional utilization (0-100)
            
        Returns:
            Estimated cooling output (RPM or flow rate)
        """
        # Physics-based: cooling scales with heat load (power) and temp excess
        temp_excess = max(0, thermal_input - self.config.target_temp)
        cooling = (
            self.config.base_cooling
            + self.config.power_coef * (power_draw / 100)  # Normalize power
            + self.config.temp_coef * temp_excess
        )
        # Clamp to realistic range
        cooling = np.clip(
            cooling,
            self.config.min_cooling,
            self.config.max_cooling,
        )
        return float(cooling)
    
    def estimate_thermal(
        self,
        power_draw: float,
        cooling_output: float,
        utilization: float = 0.0,
    ) -> float:
        """
        Estimate temperature from power and cooling.
        Inverse of estimate_cooling: temp = target + (cooling - base - power_term) / temp_coef
        
        Args:
            power_draw: Power consumption (W)
            cooling_output: Cooling capacity (RPM)
            utilization: Optional utilization
            
        Returns:
            Estimated temperature (°C)
        """
        power_term = self.config.power_coef * (power_draw / 100)
        cooling_excess = cooling_output - self.config.base_cooling - power_term
        temp_excess = cooling_excess / max(self.config.temp_coef, 1e-6)
        thermal = self.config.target_temp + max(0, temp_excess)
        return float(np.clip(thermal, 20, 95))
    
    def fill_missing_node(self, node: BaseNode) -> BaseNode:
        """
        Fill missing attributes in a BaseNode using synthetic estimation.
        
        If cooling_output is 0/missing but power and temp exist -> estimate cooling
        If thermal_input is 0/missing but power and cooling exist -> estimate temp
        
        Args:
            node: BaseNode with potentially missing attributes
            
        Returns:
            Node with estimated values filled in (mutates original)
        """
        # Estimate cooling if missing (0 or invalid)
        if node.cooling_output <= 0 and node.power_draw > 0 and node.thermal_input > 0:
            node.cooling_output = self.estimate_cooling(
                node.power_draw,
                node.thermal_input,
                node.utilization,
            )
            node.raw_data["_synthetic_cooling"] = True
        
        # Estimate thermal if missing
        if node.thermal_input <= 0 and node.power_draw > 0 and node.cooling_output > 0:
            node.thermal_input = self.estimate_thermal(
                node.power_draw,
                node.cooling_output,
                node.utilization,
            )
            node.raw_data["_synthetic_thermal"] = True
        
        return node
    
    def fill_missing_nodes(self, nodes: list) -> list:
        """
        Fill missing attributes in a list of BaseNodes.
        
        Args:
            nodes: List of BaseNode instances
            
        Returns:
            Same list with estimated values filled in
        """
        for node in nodes:
            self.fill_missing_node(node)
        return nodes
    
    def detect_physical_anomaly(self, nodes: list) -> Tuple[bool, str]:
        """
        Detect physical anomalies inconsistent with thermodynamics.
        
        Examples:
        - Fan reports 0 RPM but temp is high and power/load present (temp would rise)
        - Cooling=0, power>threshold, temp>threshold = impossible steady state
        
        Args:
            nodes: List of BaseNode instances (can be time series)
            
        Returns:
            (anomaly_detected: bool, reason: str)
        """
        if not nodes:
            return False, ""
        
        # Thresholds for anomaly detection
        COOLING_ANOMALY_THRESHOLD = 50.0   # RPM - "fan off" or failed
        POWER_LOAD_THRESHOLD = 20.0        # W - system under load
        TEMP_ANOMALY_THRESHOLD = 50.0      # °C - hot with no cooling = anomaly
        
        for i, node in enumerate(nodes):
            # Anomaly: Fan reports 0/low RPM but system is hot and under load
            if (node.cooling_output < COOLING_ANOMALY_THRESHOLD and
                node.power_draw > POWER_LOAD_THRESHOLD and
                node.thermal_input > TEMP_ANOMALY_THRESHOLD):
                return True, (
                    f"Physical anomaly: Fan reports {node.cooling_output:.0f} RPM "
                    f"but temp={node.thermal_input:.1f}°C and power={node.power_draw:.0f}W. "
                    "Temp would rise with no cooling."
                )
        
        # Time series: temp rising while cooling is 0 (fan failed)
        if len(nodes) >= 3:
            thermals = [n.thermal_input for n in nodes[-5:]]
            coolings = [n.cooling_output for n in nodes[-5:]]
            if (all(c < COOLING_ANOMALY_THRESHOLD for c in coolings) and
                thermals[-1] > thermals[0] + 2.0):  # Temp rose 2+ °C
                return True, (
                    "Physical anomaly: Temp rising while fan reports 0 RPM. "
                    "Possible fan failure or sensor fault."
                )
        
        return False, ""
