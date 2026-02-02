"""
CooledAI Protocol Collectors

Each collector implements BaseCollector and returns standardized
CooledAI Telemetry Objects regardless of underlying protocol.
"""

from .base_collector import BaseCollector, TelemetryObject
from .bacnet_manager import BACnetManager
from .snmp_manager import SNMPManager
from .redfish_manager import RedfishManager

__all__ = [
    "BaseCollector",
    "TelemetryObject",
    "BACnetManager",
    "SNMPManager",
    "RedfishManager",
]
