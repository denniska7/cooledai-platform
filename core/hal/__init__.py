"""
CooledAI Hardware Abstraction Layer (HAL)

Provides a hardware-agnostic representation of data center nodes.
All physical equipment (servers, CRACs, sensors) are modeled as BaseNode subclasses,
enabling universal thermal optimization regardless of vendor or protocol.

RedfishBridge: Watchdog Heartbeat—triggers EmergencyCoolingCommand (Full Blast)
after 3 consecutive poll failures. Safety over Savings.

Usage:
    from core.hal import BaseNode, ServerNode, RedfishBridge, EmergencyCoolingCommand
"""

from core.hal.base_node import BaseNode, ServerNode, CRAC_Unit, EnvironmentSensor
from core.hal.redfish_bridge import RedfishBridge, RedfishBridgePool, EmergencyCoolingCommand

__all__ = [
    "BaseNode",
    "ServerNode",
    "CRAC_Unit",
    "EnvironmentSensor",
    "RedfishBridge",
    "RedfishBridgePool",
    "EmergencyCoolingCommand",
]
