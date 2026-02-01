"""
CooledAI Hardware Abstraction Layer (HAL)

Provides a hardware-agnostic representation of data center nodes.
All physical equipment (servers, CRACs, sensors) are modeled as BaseNode subclasses,
enabling universal thermal optimization regardless of vendor or protocol.

Usage:
    from hal import BaseNode, ServerNode, CRAC_Unit, EnvironmentSensor
"""

from core.hal.base_node import BaseNode, ServerNode, CRAC_Unit, EnvironmentSensor

__all__ = ["BaseNode", "ServerNode", "CRAC_Unit", "EnvironmentSensor"]
