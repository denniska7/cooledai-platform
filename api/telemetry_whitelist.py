"""
CooledAI Telemetry Field Whitelist — enforces data isolation.

CooledAI collects ONLY thermal/power/fan telemetry.  Any field outside the
whitelist is stripped before storage and logged as a security violation.
"""

from __future__ import annotations

import logging
from typing import Dict, Set

log = logging.getLogger("cooledai.security")

# ── Allowed telemetry fields ────────────────────────────────────
# These are the ONLY fields that may appear in a telemetry record.
# Everything else is rejected.
ALLOWED_TELEMETRY_FIELDS: Set[str] = {
    # Identification & metadata
    "node_id",
    "agent_id",
    "client_id",
    "timestamp",
    "timestamp_utc",
    "agent_version",
    # GPU thermal & power
    "gpu_temp_c",
    "temperature_c",          # st550_telemetry.py uses this for GPU and CPU temps
    "gpu_power_w",
    "gpu_memory_utilization_pct",
    # CPU thermal
    "cpu_temp_c",
    "sensor_count",            # CPU sensor count from telemetry script
    # Fan
    "fan_rpm",
    "fan_rpms",
    "fan_duty_pct",
    "fan_speed_pct",           # GPU fan speed percentage
    "raw_fan_wattage",         # Estimated fan power from IPMI
    # Ambient
    "ambient_temp_c",
    "inlet_temp_c",
    "exhaust_temp_c",
    # Power
    "power_draw_w",
    "peak_power_w",
    # Optimization feedback
    "target_fan_rpm",
    "optimization_mode",
    "gpu_utilization_pct",
    "utilization",
    # Internal bookkeeping (set by API, not by agent)
    "_received_at",
    "_owner_id",
    "_client_id",
}


class TelemetryViolationError(ValueError):
    """Raised when telemetry contains fields outside the whitelist."""

    def __init__(self, rejected_fields: Set[str]) -> None:
        self.rejected_fields = rejected_fields
        super().__init__(
            f"Telemetry contains disallowed fields: {sorted(rejected_fields)}"
        )


def validate_record(record: Dict, *, strict: bool = False) -> Dict:
    """Validate a telemetry record against the whitelist.

    Parameters
    ----------
    record : dict
        Raw telemetry record from an agent.
    strict : bool
        If True, raise TelemetryViolationError on disallowed fields.
        If False (default), strip disallowed fields and log a warning.

    Returns
    -------
    dict
        Cleaned record with only whitelisted fields.
    """
    rejected = set(record.keys()) - ALLOWED_TELEMETRY_FIELDS
    if not rejected:
        return record

    # Log every rejected field
    for field in sorted(rejected):
        log.warning(
            "[SECURITY] telemetry_violation: field=%s rejected", field
        )

    if strict:
        raise TelemetryViolationError(rejected)

    # Permissive mode: strip disallowed fields
    return {k: v for k, v in record.items() if k in ALLOWED_TELEMETRY_FIELDS}
