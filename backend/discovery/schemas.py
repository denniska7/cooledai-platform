"""
CooledAI Discovery Schemas - Data models for unmapped telemetry and mapping results.

Used by DiscoveryAgent to represent raw Modbus/SNMP points over a time window
and the suggested mapping to normalized BaseNode attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# Normalized attributes that adapters map to (BaseNode / optimization engine)
NORMALIZED_ATTRIBUTES = [
    "thermal_input",      # Temperature °C (return air, CPU temp, inlet)
    "power_draw",        # Power Watts
    "cooling_output",    # Fan RPM, flow rate, cooling capacity
    "utilization",       # Load % 0-100
    "ambient_inlet_temp", # Rack inlet / ambient °C
]


@dataclass
class RawTelemetryPoint:
    """A single unmapped point (Modbus register or SNMP OID) with time-series values."""
    point_id: str                    # e.g. "modbus:40001" or "snmp:1.3.6.1.4.1.9.9.13.1.3.1.3.1"
    protocol: str                    # "modbus" | "snmp"
    address: str                     # Register address (decimal) or OID string
    series: List[tuple] = field(default_factory=list)  # [(timestamp_epoch, value), ...]

    def add_sample(self, timestamp: float, value: float) -> None:
        self.series.append((timestamp, value))

    def stats(self) -> Dict[str, Any]:
        """Min, max, mean, std for pattern summary (exclude invalid)."""
        if not self.series:
            return {"min": None, "max": None, "mean": None, "std": None, "count": 0}
        vals = [v for _, v in self.series if v is not None and isinstance(v, (int, float))]
        if not vals:
            return {"min": None, "max": None, "mean": None, "std": None, "count": 0}
        import statistics
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "count": len(vals),
        }


@dataclass
class RawTelemetryWindow:
    """A time-bounded window of raw telemetry from one protocol (Modbus or SNMP)."""
    protocol: str
    points: List[RawTelemetryPoint] = field(default_factory=list)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_ts is not None and self.end_ts is not None:
            return self.end_ts - self.start_ts
        return 0.0


@dataclass
class MappingSchema:
    """Suggested mapping: point_id -> normalized BaseNode attribute."""
    point_to_attribute: Dict[str, str] = field(default_factory=dict)
    scale_factors: Dict[str, float] = field(default_factory=dict)  # optional scale per point

    def to_adapter_format(self, protocol: str) -> Dict[str, Any]:
        """Produce adapter-ready format: for Modbus register_map, for SNMP oid_map."""
        if protocol == "modbus":
            # attribute -> (register_address, scale, unit)
            attr_to_addr = {}
            for point_id, attr in self.point_to_attribute.items():
                if not point_id.startswith("modbus:"):
                    continue
                try:
                    addr = int(point_id.split(":")[1])
                    scale = self.scale_factors.get(point_id, 1.0)
                    attr_to_addr[attr] = (addr, scale, "")
                except (IndexError, ValueError):
                    continue
            return {"register_map": attr_to_addr}
        if protocol == "snmp":
            # attribute -> OID
            attr_to_oid = {}
            for point_id, attr in self.point_to_attribute.items():
                if not point_id.startswith("snmp:"):
                    continue
                oid = point_id[5:]  # strip "snmp:"
                attr_to_oid[attr] = oid
            return {"oid_map": attr_to_oid}
        return {}


@dataclass
class DiscoveryResult:
    """Result of DiscoveryAgent.discover(): confidence and suggested mapping."""
    confidence_score: float          # 0.0 - 1.0
    suggested_vendor: Optional[str] = None   # e.g. "Liebert", "Stulz", "Schneider"
    suggested_mapping_schema: Optional[MappingSchema] = None
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
