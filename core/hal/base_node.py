"""
CooledAI Hardware Abstraction Layer - BaseNode and Subclasses

Defines the standard interface for all data center thermal nodes.
Every physical asset (server, CRAC, sensor) is represented as a node with
normalized attributes: thermal_input, power_draw, cooling_output, utilization.

This abstraction allows the optimization engine to operate on any hardware
without knowing vendor-specific details.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class BaseNode(ABC):
    """Abstract base class for all data center thermal nodes.

    Normalizes vendor-specific telemetry into a unified interface for the
    optimization engine. Physics: thermal_input and power_draw drive heat
    load; cooling_output is the cooling response. Optimization uses these
    to compute thermal lag, over-provisioning, and recommended cooling delta.

    Attributes:
        thermal_input: Temperature in °C (CPU Tdie, inlet, return air, etc.).
        power_draw: Power consumption in Watts (heat source).
        cooling_output: Cooling capacity (fan RPM, flow rate, etc.).
        utilization: Workload percentage 0–100.
        ambient_inlet_temp: Optional rack inlet/ambient °C for correlation.
        thermal_input_rate: Optional dT/dt (°C/s) from telemetry enrichment.
        node_id: Unique identifier. source: Data source (e.g. "modbus", "csv").
        maintenance_mode: If True, optimization skips this node (human on-site).
        raw_data: Optional vendor-specific data for debugging.
    """
    
    # Core normalized attributes - required for optimization
    thermal_input: float = 0.0      # Temperature in °C
    power_draw: float = 0.0        # Power in Watts
    cooling_output: float = 0.0    # Cooling capacity (RPM, flow rate, etc.)
    utilization: float = 0.0       # Utilization 0-100%

    # Multi-variate: ambient inlet temp for correlation (CPU Util vs Power vs Inlet)
    ambient_inlet_temp: Optional[float] = None  # Rack inlet / ambient °C

    # First derivative (thermal momentum) - °C/s, set by telemetry enrichment
    thermal_input_rate: Optional[float] = None  # dT/dt for AI to see rate of change
    
    # Metadata
    node_id: str = ""
    timestamp: Optional[datetime] = None
    source: str = ""               # Data source identifier (e.g., "modbus", "csv")

    # Maintenance mode: when True, optimization skips this node and sends no commands (human on-site)
    maintenance_mode: bool = False

    # Raw data for debugging/auditing (optional)
    raw_data: dict = field(default_factory=dict)
    
    def to_normalized_dict(self) -> dict:
        """
        Export normalized attributes for the optimization engine.
        Strips vendor-specific data - only standard fields.
        """
        out = {
            "thermal_input": self.thermal_input,
            "power_draw": self.power_draw,
            "cooling_output": self.cooling_output,
            "utilization": self.utilization,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
        if getattr(self, "thermal_input_rate", None) is not None:
            out["thermal_input_rate"] = self.thermal_input_rate
        if getattr(self, "maintenance_mode", False):
            out["maintenance_mode"] = True
        return out
    
    def is_valid(self) -> bool:
        """Check if node has minimum required data for optimization.

        Returns:
            True if thermal_input, power_draw, cooling_output are non-negative
            and utilization is in [0, 100].
        """
        return (
            self.thermal_input >= 0 and
            self.power_draw >= 0 and
            self.cooling_output >= 0 and
            0 <= self.utilization <= 100
        )


@dataclass
class ServerNode(BaseNode):
    """
    Represents a compute node (server, blade, GPU node).
    
    thermal_input: CPU/GPU temperature (Tdie, Tctl, or package temp)
    power_draw: Total server power consumption (Watts)
    cooling_output: Local fan RPM (if applicable) or 0 for external cooling
    utilization: CPU/GPU utilization percentage
    
    Used for: EPYC servers, Intel Xeon, NVIDIA DGX, etc.
    """
    
    # Server-specific optional attributes
    cpu_temp: Optional[float] = None      # CPU-specific temp (alias for thermal_input)
    gpu_temp: Optional[float] = None    # GPU temp if applicable
    fan_rpm: Optional[float] = None      # Local fan speed (maps to cooling_output)
    
    def __post_init__(self):
        """Map server-specific fields to base attributes if not set.

        If cpu_temp is set and thermal_input is 0, copies cpu_temp to thermal_input.
        If fan_rpm is set and cooling_output is 0, copies fan_rpm to cooling_output.
        """
        if self.cpu_temp is not None and self.thermal_input == 0:
            self.thermal_input = self.cpu_temp
        if self.fan_rpm is not None and self.cooling_output == 0:
            self.cooling_output = self.fan_rpm


@dataclass
class CRAC_Unit(BaseNode):
    """
    Represents a Computer Room Air Conditioning (CRAC) unit.
    
    thermal_input: Return air temperature (°C)
    power_draw: CRAC power consumption (Watts)
    cooling_output: Supply fan RPM or chilled water flow rate
    utilization: CRAC capacity utilization (0-100%)
    
    Used for: Liebert, Stulz, APC, and other industrial chillers.
    """
    
    # CRAC-specific optional attributes
    supply_temp: Optional[float] = None   # Supply air temp
    return_temp: Optional[float] = None  # Return air temp (maps to thermal_input)
    fan_speed: Optional[float] = None   # Fan RPM (maps to cooling_output)
    
    def __post_init__(self):
        """Map CRAC-specific fields to base attributes if not set.

        If return_temp is set and thermal_input is 0, copies return_temp.
        If fan_speed is set and cooling_output is 0, copies fan_speed.
        """
        if self.return_temp is not None and self.thermal_input == 0:
            self.thermal_input = self.return_temp
        if self.fan_speed is not None and self.cooling_output == 0:
            self.cooling_output = self.fan_speed


@dataclass
class EnvironmentSensor(BaseNode):
    """
    Represents an environmental sensor (hot/cold aisle, inlet/outlet).
    
    thermal_input: Ambient temperature reading (°C)
    power_draw: Typically 0 (sensors don't consume significant power)
    cooling_output: 0 (sensors measure, don't deliver cooling)
    utilization: N/A for sensors - use 0 or humidity as proxy
    
    Used for: Rack inlet/outlet sensors, aisle temp sensors, humidity probes.
    """
    
    # Sensor-specific optional attributes
    humidity: Optional[float] = None
    location: Optional[str] = None  # e.g., "cold_aisle", "rack_inlet"
    
    def __post_init__(self):
        """Environment sensors typically have no power/cooling output.

        No mapping performed; zero values for power_draw and cooling_output
        are expected for passive sensors.
        """
        if self.power_draw == 0 and self.cooling_output == 0:
            pass  # Expected for sensors
