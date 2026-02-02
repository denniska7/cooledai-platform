"""
CooledAI Universal Protocol Gateway

Enterprise-grade edge agent for high-criticality data center environments.
Collects telemetry from BACnet, SNMP, and Redfish; enforces Shadow/Production
control modes; provides store-and-forward reliability.
"""

from .collectors.base_collector import BaseCollector, TelemetryObject
from .control_gate import ControlGate, CONTROL_MODE
from .telemetry_buffer import TelemetryBuffer
from .normalizer import Normalizer
from .log_scrubber import scrub_log_message

__all__ = [
    "BaseCollector",
    "TelemetryObject",
    "ControlGate",
    "CONTROL_MODE",
    "TelemetryBuffer",
    "Normalizer",
    "scrub_log_message",
]
