"""
CooledAI Redfish Bridge - HAL Layer with Watchdog Heartbeat

Bridges Redfish hardware to the optimization engine. Implements a Watchdog
Heartbeat: if the bridge fails to communicate with a server for more than
3 consecutive polls, triggers EmergencyCoolingCommand (OEM Default / Full Blast).

Safety over Savings: Always default to hardware safety when the network is unstable.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

_logger = logging.getLogger("cooledai.redfish_bridge")


# --- Emergency Cooling Command ---

REVERT_TARGET = "revert_to_default"
REVERT_VALUE_FULL_BLAST = 100  # 100% = OEM Default / Full Blast cooling


@dataclass
class EmergencyCoolingCommand:
    """
    Signal to revert hardware to OEM Default or Full Blast cooling.

    Used when Watchdog detects 3+ consecutive poll failures. Bypasses
    optimization—safety over savings. Caller must execute via the bridge's
    write handler.
    """

    server_id: str
    reason: str
    cooling_percent: int = REVERT_VALUE_FULL_BLAST
    target: str = REVERT_TARGET

    def execute(self, write_fn: Callable[[str, Any, str], bool]) -> bool:
        """Send command to hardware. Returns True if write succeeded."""
        return write_fn(self.target, self.cooling_percent, "%")


@runtime_checkable
class RedfishClient(Protocol):
    """Protocol for Redfish clients: collect telemetry and write commands."""

    def collect(self) -> List[Any]:
        """Collect telemetry. Returns list of telemetry objects or empty on failure."""
        ...

    def write(self, target: str, value: Any, unit: str = "") -> bool:
        """Write to hardware. Returns True if succeeded."""
        ...


class RedfishBridge:
    """
    HAL bridge for Redfish servers with Watchdog Heartbeat.

    Polls via the provided client. Tracks consecutive communication failures
    per server. After MAX_CONSECUTIVE_FAILURES (default 3), triggers
    EmergencyCoolingCommand: revert to OEM Default / Full Blast cooling.

    Safety over Savings: When network is unstable, we default to hardware
    safety (100% cooling) rather than risk thermal runaway.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        client: RedfishClient,
        server_id: Optional[str] = None,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        on_emergency_cooling: Optional[Callable[[EmergencyCoolingCommand], None]] = None,
    ):
        """
        Args:
            client: Redfish client with collect() and write() methods.
            server_id: Identifier for this server (e.g. hostname, IP). Defaults to client source.
            max_consecutive_failures: Trigger emergency cooling after this many failed polls (default 3).
            on_emergency_cooling: Optional callback when EmergencyCoolingCommand is triggered.
        """
        self._client = client
        self._server_id = server_id or getattr(client, "source", "redfish_unknown")
        self._max_failures = max_consecutive_failures
        self._on_emergency = on_emergency_cooling
        self._consecutive_failures: int = 0
        self._emergency_triggered: bool = False

    @property
    def server_id(self) -> str:
        return self._server_id

    @property
    def emergency_triggered(self) -> bool:
        """True if EmergencyCoolingCommand was triggered this session."""
        return self._emergency_triggered

    def poll(self) -> tuple[bool, List[Any]]:
        """
        Poll telemetry from the server. Watchdog: track consecutive failures.

        Returns:
            (success, telemetry_objects). success=False on communication failure.
        """
        try:
            objs = self._client.collect()
            if objs is None:
                self._consecutive_failures += 1
                _logger.warning(
                    "RedfishBridge: collect returned None for %s (consecutive=%d/%d)",
                    self._server_id,
                    self._consecutive_failures,
                    self._max_failures,
                )
                if self._consecutive_failures >= self._max_failures:
                    self._trigger_emergency_cooling()
                return False, []
            objs = list(objs) if objs else []
            # Success: got telemetry
            success = True
            if objs:
                # Valid response
                self._consecutive_failures = 0
                return True, list(objs) if objs else []

            self._consecutive_failures += 1
            _logger.warning(
                "RedfishBridge: poll failed for %s (consecutive=%d/%d)",
                self._server_id,
                self._consecutive_failures,
                self._max_failures,
            )
            if self._consecutive_failures >= self._max_failures:
                self._trigger_emergency_cooling()
            return False, []

        except Exception as e:
            self._consecutive_failures += 1
            _logger.warning(
                "RedfishBridge: poll error for %s: %s (consecutive=%d/%d)",
                self._server_id,
                e,
                self._consecutive_failures,
                self._max_failures,
            )
            if self._consecutive_failures >= self._max_failures:
                self._trigger_emergency_cooling()
            return False, []

    def _trigger_emergency_cooling(self) -> None:
        """
        Trigger EmergencyCoolingCommand: revert to OEM Default / Full Blast.

        Sends signal directly to hardware via client.write (bypasses ControlGate).
        Safety over Savings: always default to hardware safety when network unstable.
        """
        if self._emergency_triggered:
            return  # Already sent
        self._emergency_triggered = True
        cmd = EmergencyCoolingCommand(
            server_id=self._server_id,
            reason=f"Watchdog: {self._consecutive_failures} consecutive poll failures",
            cooling_percent=REVERT_VALUE_FULL_BLAST,
            target=REVERT_TARGET,
        )
        _logger.critical(
            "RedfishBridge: EMERGENCY COOLING for %s - %s. Sending Full Blast (100%%) to hardware.",
            self._server_id,
            cmd.reason,
        )
        try:
            ok = cmd.execute(self._client.write)
            if ok:
                _logger.info("RedfishBridge: Emergency cooling command sent successfully for %s", self._server_id)
            else:
                _logger.error("RedfishBridge: Emergency cooling write FAILED for %s", self._server_id)
        except Exception as e:
            _logger.error("RedfishBridge: Emergency cooling write exception for %s: %s", self._server_id, e)
        if self._on_emergency:
            try:
                self._on_emergency(cmd)
            except Exception as e:
                _logger.warning("RedfishBridge: on_emergency_cooling callback failed: %s", e)

    def reset_watchdog(self) -> None:
        """Reset failure count and emergency flag. Call after successful reconnection."""
        self._consecutive_failures = 0
        self._emergency_triggered = False


class RedfishBridgePool:
    """
    Manages multiple RedfishBridge instances (e.g. one per server/chassis).

    Aggregates poll results and tracks Watchdog per bridge.
    """

    def __init__(self, max_consecutive_failures: int = 3):
        self._bridges: Dict[str, RedfishBridge] = {}
        self._max_failures = max_consecutive_failures

    def register(self, server_id: str, client: RedfishClient) -> RedfishBridge:
        """Register a Redfish client. Returns the bridge."""
        bridge = RedfishBridge(
            client=client,
            server_id=server_id,
            max_consecutive_failures=self._max_failures,
        )
        self._bridges[server_id] = bridge
        return bridge

    def poll_all(self) -> List[Any]:
        """Poll all registered bridges. Returns combined telemetry."""
        all_objs: List[Any] = []
        for bridge in self._bridges.values():
            success, objs = bridge.poll()
            all_objs.extend(objs)
        return all_objs

    def any_emergency_triggered(self) -> bool:
        """True if any bridge triggered EmergencyCoolingCommand."""
        return any(b.emergency_triggered for b in self._bridges.values())
