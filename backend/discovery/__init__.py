"""
CooledAI Discovery - Automatic equipment and mapping discovery from raw telemetry.

DiscoveryAgent analyzes unmapped Modbus/SNMP data over a 10-minute window and uses
an LLM to compare patterns against known profiles (Liebert, Stulz, Schneider),
returning a Confidence Score and suggested Mapping Schema for onboarding.
"""

from backend.discovery.schemas import (
    RawTelemetryPoint,
    RawTelemetryWindow,
    MappingSchema,
    DiscoveryResult,
    NORMALIZED_ATTRIBUTES,
)
from backend.discovery.equipment_profiles import (
    get_all_profiles,
    get_profiles_by_protocol,
    EquipmentProfile,
)
from backend.discovery.discovery_agent import DiscoveryAgent, DISCOVERY_WINDOW_SECONDS

__all__ = [
    "DiscoveryAgent",
    "DiscoveryResult",
    "MappingSchema",
    "RawTelemetryPoint",
    "RawTelemetryWindow",
    "NORMALIZED_ATTRIBUTES",
    "get_all_profiles",
    "get_profiles_by_protocol",
    "EquipmentProfile",
    "DISCOVERY_WINDOW_SECONDS",
]
