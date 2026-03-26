"""
Shared Pydantic models for agent ↔ optimization communication.

Used by both the cloud API (api/main.py) and the edge gateway (gateway/api.py)
so agents can POST telemetry to either endpoint with the same request format.
"""

from typing import Optional
from pydantic import BaseModel


class AgentOptimizeControlInput(BaseModel):
    """Agent telemetry snapshot for real-time optimization control.

    Posted every 3 seconds by the CooledAI agent.
    Identical contract whether the agent targets the cloud API or the edge gateway.
    """
    temp_c: float  # GPU avg or max temp
    fan_rpm: float
    gpu_power_w: float = 50.0
    cpu_temp_c: Optional[float] = None
    # TODO Phase 4: Add cpu_power_w to agent telemetry and thermal history tuple.
    # Currently only GPU power (gpu_power_w) and peak power (peak_power_w) are tracked.
    node_id: str = "ST550-CooledAI-Predictive"
    max_fan_rpm: float = 7000.0  # For duty conversion; agent can override
    # Last fan duty actually applied (0-100). Enables fan-slippage heuristic vs tach.
    last_commanded_duty: Optional[float] = None
    # FIX E: Raw peak GPU power within polling interval (W) for spike detection
    peak_power_w: Optional[float] = None
    # Calibration profile from agent's ThermalCalibrator (dict form)
    calibration_profile: Optional[dict] = None
