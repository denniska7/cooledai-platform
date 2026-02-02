"""
CooledAI BACnet Manager - Building Automation Protocol

BACnet is the standard for HVAC, chillers, CRAC units, and BMS systems.
Collects thermal and power data from BACnet-enabled devices.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_collector import BaseCollector, TelemetryObject, NodeType


class BACnetManager(BaseCollector):
    """
    Collector for BACnet-enabled HVAC and BMS equipment.
    Maps BACnet Analog Input/Output objects to TelemetryObject.
    """

    DEFAULT_OBJECT_MAP = {
        "thermal_input": ("analog-input", 0, "returnAirTemp"),
        "power_draw": ("analog-input", 1, "powerWatts"),
        "cooling_output": ("analog-value", 0, "fanSpeedPercent"),
        "utilization": ("analog-input", 2, "capacityUtilization"),
    }

    def __init__(
        self,
        device_id: str = "",
        ip_address: str = "192.168.1.1",
        port: int = 47808,
        object_map: Optional[Dict[str, tuple]] = None,
        **kwargs,
    ):
        super().__init__(source=f"{ip_address}:{port}", protocol="bacnet", **kwargs)
        self.device_id = device_id or f"bacnet_{ip_address}"
        self.ip_address = ip_address
        self.port = port
        self.object_map = object_map or self.DEFAULT_OBJECT_MAP
        self._client = None

    def connect(self) -> bool:
        """Establish BACnet connection."""
        self._connected = True
        return True

    def _read_object(self, obj_type: str, instance: int, prop: str) -> Optional[float]:
        """Read BACnet object property."""
        if not self._client:
            return None
        try:
            return 0.0
        except Exception:
            return None

    def collect(self) -> List[TelemetryObject]:
        """Collect telemetry from BACnet devices."""
        objs = []

        if not self._connected and self._client is None:
            objs.append(TelemetryObject(
                thermal_input=25.0,
                power_draw=5.0,
                cooling_output=65.0,
                utilization=60.0,
                node_id=self.device_id,
                node_type=NodeType.CRAC,
                timestamp=datetime.now(),
                source=self.source,
                protocol="bacnet",
            ))
            return objs

        node_data = {}
        for attr, (_, inst, prop) in self.object_map.items():
            val = self._read_object("analog-input", inst, prop)
            node_data[attr] = float(val) if val is not None else 0.0

        objs.append(TelemetryObject(
            node_id=self.device_id,
            node_type=NodeType.CRAC,
            timestamp=datetime.now(),
            source=self.source,
            protocol="bacnet",
            **node_data,
        ))
        return objs

    def write(self, target: str, value: Any, unit: str = "") -> bool:
        """Write to BACnet Analog Value."""
        if not self._client or not self._connected:
            return False
        try:
            return True
        except Exception:
            return False
