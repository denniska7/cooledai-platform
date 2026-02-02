"""
CooledAI Base Collector - Abstract Interface for Protocol Gateways

Every protocol collector (BACnet, SNMP, Redfish) must implement BaseCollector
and return standardized CooledAI Telemetry Objects. This ensures the
optimization engine receives consistent data regardless of source.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class NodeType(str, Enum):
    """Standardized node types for telemetry."""
    SERVER = "server"
    CRAC = "crac"
    SENSOR = "sensor"
    PDU = "pdu"


@dataclass
class TelemetryObject:
    """
    CooledAI Telemetry Object - standardized output from all collectors.
    
    All units are normalized:
    - thermal_input: Celsius
    - power_draw: Kilowatts
    - cooling_output: RPM or flow rate (unit in cooling_unit)
    - utilization: 0-100%
    """
    # Core normalized attributes
    thermal_input: float = 0.0       # °C
    power_draw: float = 0.0         # kW
    cooling_output: float = 0.0      # RPM or flow
    utilization: float = 0.0        # 0-100%
    
    # Metadata
    node_id: str = ""
    node_type: NodeType = NodeType.SERVER
    timestamp: Optional[datetime] = None
    source: str = ""                 # Protocol identifier
    protocol: str = ""               # bacnet, snmp, redfish
    
    # Optional
    ambient_inlet_temp: Optional[float] = None  # °C
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export for JSON transmission."""
        return {
            "thermal_input": self.thermal_input,
            "power_draw": self.power_draw,
            "cooling_output": self.cooling_output,
            "utilization": self.utilization,
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "protocol": self.protocol,
            "ambient_inlet_temp": self.ambient_inlet_temp,
        }
    
    def is_valid(self) -> bool:
        """Check if telemetry has minimum required data."""
        return (
            self.thermal_input >= 0 and
            self.power_draw >= 0 and
            self.cooling_output >= 0 and
            0 <= self.utilization <= 100
        )


class BaseCollector(ABC):
    """
    Abstract base class for protocol collectors.
    
    All collectors must:
    - implement collect() -> List[TelemetryObject]
    - return normalized TelemetryObjects (units already converted)
    - implement write() for SET/WRITE commands (intercepted by ControlGate)
    """
    
    def __init__(self, source: str = "", protocol: str = ""):
        self.source = source
        self.protocol = protocol
        self._connected = False
    
    @abstractmethod
    def collect(self) -> List[TelemetryObject]:
        """
        Collect telemetry and return standardized TelemetryObjects.
        
        Returns:
            List of TelemetryObject with normalized units (C, kW).
        """
        pass
    
    def connect(self) -> bool:
        """Establish connection. Override in subclasses."""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
    
    def write(
        self,
        target: str,
        value: Any,
        unit: str = "",
    ) -> bool:
        """
        Write/SET command to hardware.
        
        Called by ControlGate only after safety bounds check (PRODUCTION)
        or intercepted and logged (SHADOW). Override in subclasses.
        
        Args:
            target: OID, register, or property path
            value: Value to write
            unit: Unit of value (e.g., RPM, %)
            
        Returns:
            True if write succeeded
        """
        return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
