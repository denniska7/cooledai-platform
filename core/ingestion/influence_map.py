"""
CooledAI InfluenceMap - Relationship Between Cooling Units and Rack Zones

Stores which CRACs (or cooling units) influence which rack zones and with what
influence coefficient. Allows the AI to reason about thermal momentum and which
levers affect which areas.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class InfluenceMap:
    """Mapping from cooling units to rack zones with influence coefficients.

    Physics: Each CRAC influences nearby rack zones with a coefficient in [0, 1].
    Used for failing-component redistribution—when one unit is degraded, the
    brain reduces its load and increases healthy units that influence the same
    zones. Sum of coefficients to a zone can exceed 1 (overlapping coverage)
    or be less than 1 (partial coverage).
    """

    # cooling_unit_id -> zone_id -> influence_coefficient
    _influence_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _cooling_units: List[str] = field(default_factory=list)
    _rack_zones: List[str] = field(default_factory=list)

    def set_influence(
        self, cooling_unit_id: str, zone_id: str, influence_coefficient: float
    ) -> None:
        """Set influence coefficient from a cooling unit to a rack zone (0–1)."""
        if cooling_unit_id not in self._influence_coefficients:
            self._influence_coefficients[cooling_unit_id] = {}
            if cooling_unit_id not in self._cooling_units:
                self._cooling_units.append(cooling_unit_id)
        self._influence_coefficients[cooling_unit_id][zone_id] = max(
            0.0, min(1.0, influence_coefficient)
        )
        if zone_id not in self._rack_zones:
            self._rack_zones.append(zone_id)

    def get_influence(self, cooling_unit_id: str, zone_id: str) -> float:
        """Return influence coefficient from cooling unit to zone.

        Args:
            cooling_unit_id: Identifier of the cooling unit.
            zone_id: Identifier of the rack zone.

        Returns:
            Influence coefficient in [0, 1], or 0 if not set.
        """
        return self._influence_coefficients.get(cooling_unit_id, {}).get(zone_id, 0.0)

    def get_zones_for_cooling_unit(self, cooling_unit_id: str) -> List[tuple]:
        """
        Return list of (zone_id, influence_coefficient) for all zones influenced by this unit.
        """
        return list(self._influence_coefficients.get(cooling_unit_id, {}).items())

    def get_cooling_units_for_zone(self, zone_id: str) -> List[tuple]:
        """Return cooling units that influence a zone.

        Args:
            zone_id: Identifier of the rack zone.

        Returns:
            List of (cooling_unit_id, influence_coefficient) tuples.
        """
        out = []
        for cu_id, zones in self._influence_coefficients.items():
            coef = zones.get(zone_id, 0.0)
            if coef > 0:
                out.append((cu_id, coef))
        return out

    @property
    def cooling_units(self) -> List[str]:
        """All cooling unit identifiers."""
        return list(self._cooling_units)

    @property
    def rack_zones(self) -> List[str]:
        """All rack zone identifiers."""
        return list(self._rack_zones)

    def to_dict(self) -> dict:
        """Serialize for API / logging: cooling_units, rack_zones, influence_coefficients."""
        return {
            "cooling_units": self.cooling_units,
            "rack_zones": self.rack_zones,
            "influence_coefficients": {
                cu: dict(zones) for cu, zones in self._influence_coefficients.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InfluenceMap":
        """Build InfluenceMap from a serialized dict.

        Args:
            data: Dict with influence_coefficients or legacy 'weights' key.

        Returns:
            New InfluenceMap instance.
        """
        m = cls()
        coefs = data.get("influence_coefficients") or data.get("weights", {})
        for cu_id, zones in coefs.items():
            for zone_id, val in zones.items():
                m.set_influence(cu_id, zone_id, float(val))
        return m
