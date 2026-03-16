"""
CooledAI Discovery Service - Network Scan & Thermal Ping

Scans IP ranges for Redfish/SNMP devices, queries Chassis/Model info,
and runs Thermal Ping tests to automatically build the Influence Map.
"""

from .discovery_agent import DiscoveryAgent, DiscoveredDevice, DiscoveryResult

__all__ = ["DiscoveryAgent", "DiscoveredDevice", "DiscoveryResult"]
