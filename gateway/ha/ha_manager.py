"""
Active/standby failover manager for the CooledAI edge gateway.

STUB: Full implementation deferred until single-instance gateway is proven
in production on both ST550s. Currently a no-op that always reports
role="standalone".

When activated (Phase 1.5+), this will:
- Send heartbeats to a standby peer every 5 seconds
- Promote standby to primary if 6 consecutive heartbeats are missed
- Coordinate with keepalived/VRRP for Virtual IP failover
- Promote Redis replica to master on failover
- Notify cloud of failover events
"""

import logging

logger = logging.getLogger("cooledai.gateway.ha")


class HAManager:
    """Active/standby failover manager.

    Currently a standalone no-op. Full implementation will be activated
    after Steps 1-7 are validated in production.
    """

    def __init__(self, role: str = "standalone", peer_url: str = ""):
        self.role = role  # "standalone", "primary", "standby"
        self.peer_url = peer_url
        self._heartbeat_interval = 5  # seconds
        self._failover_threshold = 6  # missed heartbeats before promotion

    @property
    def is_active(self) -> bool:
        """Whether this gateway should serve optimization requests."""
        return True  # Standalone always active

    def start_heartbeat(self) -> None:
        """Primary: send heartbeat to standby every 5s. (Not implemented)"""
        if self.role == "standalone":
            return
        raise NotImplementedError("HA heartbeat not yet implemented")

    def monitor_peer(self) -> None:
        """Standby: monitor primary heartbeat, promote if missed. (Not implemented)"""
        if self.role == "standalone":
            return
        raise NotImplementedError("HA peer monitoring not yet implemented")

    def promote_to_primary(self) -> None:
        """Standby becomes primary: claim VIP, promote Redis, notify cloud. (Not implemented)"""
        raise NotImplementedError("HA promotion not yet implemented")

    def demote_to_standby(self) -> None:
        """Primary becomes standby: release VIP, configure Redis as replica. (Not implemented)"""
        raise NotImplementedError("HA demotion not yet implemented")
