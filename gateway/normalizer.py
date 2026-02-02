"""
CooledAI Data Normalizer - Unit Conversion

Converts all incoming units to standard:
- Fahrenheit -> Celsius
- Watts -> Kilowatts
"""

from typing import List
from .collectors.base_collector import TelemetryObject


class Normalizer:
    """Normalize telemetry units before transmission."""

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5 / 9

    @staticmethod
    def watts_to_kilowatts(w: float) -> float:
        return w / 1000.0

    @staticmethod
    def normalize(obj: TelemetryObject) -> TelemetryObject:
        raw = obj.raw_data or {}
        if raw.get("thermal_unit") == "F" or raw.get("temp_unit") == "F":
            obj.thermal_input = Normalizer.fahrenheit_to_celsius(obj.thermal_input)
        if raw.get("power_unit") == "W" or raw.get("power_unit") == "watts":
            obj.power_draw = Normalizer.watts_to_kilowatts(obj.power_draw)
        if obj.power_draw > 1000 and obj.power_draw < 1e6:
            obj.power_draw = Normalizer.watts_to_kilowatts(obj.power_draw)
        return obj

    @staticmethod
    def normalize_batch(objs: List[TelemetryObject]) -> List[TelemetryObject]:
        return [Normalizer.normalize(o) for o in objs]
