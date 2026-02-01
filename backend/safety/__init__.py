"""
CooledAI Safety Module

Watchdog timer, Safety State Machine, and fail-safe systems for thermal control.
Ensures hardware remains in safe state if AI software hangs or crashes.

Usage:
    from safety import Watchdog, SafetyOrchestrator, SystemState
    watchdog = Watchdog(heartbeat_interval=10)
    orchestrator = SafetyOrchestrator(watchdog=watchdog)
    # AI loop: watchdog.kick() every cycle
"""

from backend.safety.watchdog import Watchdog, SafeStateConfig, MockCoolingAdapter
from backend.safety.state_machine import SafetyOrchestrator, SystemState
from backend.safety.guardrails import snap_rpm_to_safe_boundary, RESONANCE_ZONES

__all__ = [
    "Watchdog",
    "SafeStateConfig",
    "MockCoolingAdapter",
    "SafetyOrchestrator",
    "SystemState",
    "snap_rpm_to_safe_boundary",
    "RESONANCE_ZONES",
]
