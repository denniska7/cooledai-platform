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
    """
    Abstract base class for all data center thermal nodes.
    
    Standard attributes (normalized across all hardware types):
    - thermal_input: Temperature reading in °C (e.g., CPU Tdie, inlet temp)
    - power_draw: Power consumption in Watts
    - cooling_output: Cooling capacity delivered (e.g., fan RPM, chilled water flow)
    - utilization: Workload/utilization percentage (0-100)
    
    All subclasses must implement these attributes to ensure the
    optimization engine receives consistent data regardless of source.
    """
    
    # Core normalized attributes - required for optimization
    thermal_input: float = 0.0      # Temperature in °C
    power_draw: float = 0.0        # Power in Watts
    cooling_output: float = 0.0    # Cooling capacity (RPM, flow rate, etc.)
    utilization: float = 0.0       # Utilization 0-100%

    # Multi-variate: ambient inlet temp for correlation (CPU Util vs Power vs Inlet)
    ambient_inlet_temp: Optional[float] = None  # Rack inlet / ambient °C
    
    # Metadata
    node_id: str = ""
    timestamp: Optional[datetime] = None
    source: str = ""               # Data source identifier (e.g., "modbus", "csv")
    
    # Raw data for debugging/auditing (optional)
    raw_data: dict = field(default_factory=dict)
    
    def to_normalized_dict(self) -> dict:
        """
        Export normalized attributes for the optimization engine.
        Strips vendor-specific data - only standard fields.
        """
        return {
            "thermal_input": self.thermal_input,
            "power_draw": self.power_draw,
            "cooling_output": self.cooling_output,
            "utilization": self.utilization,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    def is_valid(self) -> bool:
        """Check if node has minimum required data for optimization."""
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
        """Map server-specific fields to base attributes if not set."""
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
        """Map CRAC-specific fields to base attributes if not set."""
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
        """Environment sensors typically have no power/cooling output."""
        if self.power_draw == 0 and self.cooling_output == 0:
            pass  # Expected for sensors
