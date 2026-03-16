"""
CooledAI Equipment Profiles - Known vendor patterns for Modbus/SNMP CRAC and BMS.

Used by DiscoveryAgent to compare raw telemetry patterns against Liebert (Vertiv),
Stulz, Schneider (APC/StruxureWare) and suggest a mapping schema during onboarding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

# Normalized attribute names (must match BaseNode / NORMALIZED_ATTRIBUTES)
THERMAL = "thermal_input"
POWER = "power_draw"
COOLING = "cooling_output"
UTIL = "utilization"
AMBIENT = "ambient_inlet_temp"


@dataclass
class TypicalPoint:
    """One typical point for a vendor: address/OID, suggested attribute, value range."""
    address: str          # Modbus register (decimal string) or SNMP OID
    attribute: str
    value_min: float
    value_max: float
    description: str = ""


def _modbus_point(addr: int, attr: str, vmin: float, vmax: float, desc: str = "") -> TypicalPoint:
    return TypicalPoint(address=str(addr), attribute=attr, value_min=vmin, value_max=vmax, description=desc)


def _snmp_point(oid: str, attr: str, vmin: float, vmax: float, desc: str = "") -> TypicalPoint:
    return TypicalPoint(address=oid, attribute=attr, value_min=vmin, value_max=vmax, description=desc)


# Liebert (Vertiv) - CRAC / InRow. Modbus: return air, supply, fan, alarm (common patterns).
# Register 30743 = return air temp (input reg); 0x1000-style holding regs used in many docs.
LIEBERT_MODBUS_POINTS = [
    _modbus_point(4096, THERMAL, 15.0, 45.0, "Return air temperature °C (x0.1)"),
    _modbus_point(4097, COOLING, 0.0, 3500.0, "Fan speed RPM"),
    _modbus_point(4098, POWER, 0.0, 50000.0, "Power Watts or kW x10"),
    _modbus_point(4099, UTIL, 0.0, 100.0, "Capacity %"),
    _modbus_point(4100, AMBIENT, 15.0, 45.0, "Supply/supply air temp °C (x0.1)"),
    # Vertiv IntelliSlot / DS style (decimal)
    _modbus_point(30743, THERMAL, 10.0, 100.0, "Return air temp °F x10 (read as input reg)"),
    _modbus_point(10025, COOLING, 0.0, 100.0, "Fan state / speed %"),
]

# Stulz - CRAC. Modbus register ranges vary by controller (Cybero, E2); typical holding regs.
STULZ_MODBUS_POINTS = [
    _modbus_point(40001, THERMAL, 15.0, 45.0, "Return/setpoint temperature"),
    _modbus_point(40002, COOLING, 0.0, 3000.0, "Fan speed RPM"),
    _modbus_point(40003, POWER, 0.0, 50000.0, "Power consumption W"),
    _modbus_point(40004, UTIL, 0.0, 100.0, "Compressor/capacity %"),
    _modbus_point(40005, AMBIENT, 15.0, 45.0, "Supply air temperature"),
]

# Schneider (APC/StruxureWare) - InRow, PDUs. Often SNMP or Modbus with different maps.
SCHNEIDER_MODBUS_POINTS = [
    _modbus_point(3001, THERMAL, 15.0, 45.0, "Inlet/return temp"),
    _modbus_point(3002, COOLING, 0.0, 3000.0, "Fan RPM"),
    _modbus_point(3003, POWER, 0.0, 50000.0, "Power W"),
    _modbus_point(3004, UTIL, 0.0, 100.0, "Load %"),
]

# Schneider / Cisco-style SNMP (BMS, network gear)
SCHNEIDER_SNMP_POINTS = [
    _snmp_point("1.3.6.1.4.1.9.9.13.1.3.1.3.1", THERMAL, 10.0, 80.0, "Cisco env temp"),
    _snmp_point("1.3.6.1.4.1.9.9.13.1.4.1.2.1", COOLING, 0.0, 10000.0, "Fan RPM"),
    _snmp_point("1.3.6.1.4.1.9.9.13.1.5.1.3.1", POWER, 0.0, 500.0, "Power"),
    _snmp_point("1.3.6.1.4.1.9.9.109.1.1.1.1.6.1", UTIL, 0.0, 100.0, "CPU utilization"),
]

# APC / StruxureWare PDU / InRow SNMP OIDs (common patterns)
APC_SNMP_POINTS = [
    _snmp_point("1.3.6.1.4.1.318.1.1.26.6.3.1.6.1", THERMAL, 10.0, 80.0, "APC inlet temp"),
    _snmp_point("1.3.6.1.4.1.318.1.1.26.6.2.1.4.1", POWER, 0.0, 100000.0, "APC power"),
]


@dataclass
class EquipmentProfile:
    """One vendor/equipment profile: name, protocol, typical points for LLM comparison."""
    vendor: str
    protocol: str  # "modbus" | "snmp"
    description: str
    typical_points: List[TypicalPoint] = field(default_factory=list)

    def to_llm_description(self) -> str:
        """String summary for LLM prompt."""
        lines = [f"{self.vendor} ({self.protocol}): {self.description}"]
        for p in self.typical_points:
            lines.append(f"  - {p.address} -> {p.attribute} (typical range {p.value_min}-{p.value_max}) {p.description}")
        return "\n".join(lines)


def get_all_profiles() -> List[EquipmentProfile]:
    """Return all known equipment profiles for discovery."""
    return [
        EquipmentProfile(
            vendor="Liebert",
            protocol="modbus",
            description="Vertiv Liebert CRAC/InRow; return air, fan, power registers.",
            typical_points=LIEBERT_MODBUS_POINTS,
        ),
        EquipmentProfile(
            vendor="Stulz",
            protocol="modbus",
            description="Stulz CRAC; E2/Cybero controller register map.",
            typical_points=STULZ_MODBUS_POINTS,
        ),
        EquipmentProfile(
            vendor="Schneider",
            protocol="modbus",
            description="Schneider/APC InRow Modbus.",
            typical_points=SCHNEIDER_MODBUS_POINTS,
        ),
        EquipmentProfile(
            vendor="Schneider",
            protocol="snmp",
            description="Schneider/Cisco SNMP BMS and network gear.",
            typical_points=SCHNEIDER_SNMP_POINTS,
        ),
        EquipmentProfile(
            vendor="APC",
            protocol="snmp",
            description="APC PDU/InRow SNMP.",
            typical_points=APC_SNMP_POINTS,
        ),
    ]


def get_profiles_by_protocol(protocol: str) -> List[EquipmentProfile]:
    """Return profiles for the given protocol (modbus or snmp)."""
    return [p for p in get_all_profiles() if p.protocol == protocol]
