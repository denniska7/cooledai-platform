"""
CooledAI Redfish Bridge Collector - Gateway adapter for Watchdog Heartbeat

Wraps RedfishManager with RedfishBridge so the Watchdog Heartbeat runs during
collection. If 3+ consecutive polls fail, triggers EmergencyCoolingCommand
(OEM Default / Full Blast) directly to hardware—Safety over Savings.
"""

import logging
from typing import List, Optional, Callable, Any

from .base_collector import BaseCollector, TelemetryObject
from .redfish_manager import RedfishManager
from core.hal.redfish_bridge import RedfishBridge, EmergencyCoolingCommand

_logger = logging.getLogger("cooledai.redfish_bridge_collector")


class RedfishBridgeCollector(BaseCollector):
    """
    Collector adapter that runs RedfishManager through RedfishBridge.

    Enables Watchdog Heartbeat: after 3 consecutive poll failures,
    triggers EmergencyCoolingCommand (Full Blast) to hardware.
    """

    def __init__(
        self,
        manager: RedfishManager,
        on_emergency_cooling: Optional[Callable[[EmergencyCoolingCommand], None]] = None,
    ):
        super().__init__(source=manager.source, protocol=manager.protocol)
        self._manager = manager
        self._bridge = RedfishBridge(
            client=manager,
            server_id=manager.source or "redfish_unknown",
            on_emergency_cooling=on_emergency_cooling,
        )
        self._connected = manager._connected

    def connect(self) -> bool:
        """Delegate to underlying manager."""
        ok = self._manager.connect()
        self._connected = self._manager._connected
        return ok

    def collect(self) -> List[TelemetryObject]:
        """Poll via bridge. Watchdog tracks failures; triggers Full Blast on 3+."""
        success, objs = self._bridge.poll()
        return list(objs) if objs else []

    def write(self, target: str, value: Any, unit: str = "") -> bool:
        """Delegate to underlying manager."""
        return self._manager.write(target, value, unit)

    @property
    def emergency_triggered(self) -> bool:
        """True if Watchdog triggered EmergencyCoolingCommand this session."""
        return self._bridge.emergency_triggered
