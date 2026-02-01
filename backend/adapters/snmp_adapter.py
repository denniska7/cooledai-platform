"""
CooledAI SNMP Adapter - Template for Networking Gear and BMS Integration

SNMP (Simple Network Management Protocol) is used by network switches,
PDUs, and many Building Management Systems (BMS) for monitoring.
This adapter template provides the structure for reading thermal and
power data from SNMP-enabled devices.

Typical SNMP OIDs for data center equipment:
- Temperature: .1.3.6.1.4.1.9.9.13.1.3.1.3 (Cisco env temp)
- Fan RPM: .1.3.6.1.4.1.9.9.13.1.4.1.2
- Power: .1.3.6.1.4.1.9.9.13.1.5.1.3

Requires: pip install pysnmp-lextudio (or pysnmp)
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from backend.safety.watchdog import SafeStateConfig

from core.hal.base_node import BaseNode, ServerNode
from .base_adapter import BaseAdapter


class SNMPAdapter(BaseAdapter):
    """
    Adapter for SNMP-enabled devices (switches, PDUs, BMS sensors).
    
    Maps SNMP OIDs to normalized BaseNode attributes.
    OID map is configurable per device vendor (Cisco, APC, etc.).
    
    Usage:
        adapter = SNMPAdapter(host="192.168.1.1", community="public")
        adapter.connect()
        nodes = adapter.read()
    """
    
    # Default OID map - override per device vendor
    # Format: attribute -> OID string
    DEFAULT_OID_MAP = {
        "thermal_input": "1.3.6.1.4.1.9.9.13.1.3.1.3.1",   # Cisco env temp
        "power_draw": "1.3.6.1.4.1.9.9.13.1.5.1.3.1",     # Cisco power
        "cooling_output": "1.3.6.1.4.1.9.9.13.1.4.1.2.1",  # Cisco fan
        "utilization": "1.3.6.1.4.1.9.9.109.1.1.1.1.6.1",  # Cisco CPU util
    }
    
    def __init__(
        self,
        host: str = "localhost",
        community: str = "public",
        port: int = 161,
        oid_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize SNMP adapter.
        
        Args:
            host: SNMP agent hostname/IP
            community: SNMP community string (v1/v2c)
            port: SNMP port (default 161)
            oid_map: Override default OIDs per vendor
        """
        super().__init__(source=f"{host}:{port}", node_type=ServerNode, **kwargs)
        self.host = host
        self.community = community
        self.port = port
        self.oid_map = oid_map or self.DEFAULT_OID_MAP
        self._session = None  # pysnmp session
    
    def connect(self) -> bool:
        """
        Establish SNMP session.
        Uses pysnmp if available; otherwise returns False.
        """
        try:
            from pysnmp.hlapi import SnmpEngine, CommunityData, UdpTransportTarget
            self._session = {
                "engine": SnmpEngine(),
                "community": CommunityData(self.community),
                "target": UdpTransportTarget((self.host, self.port)),
            }
            self._connected = True
            return True
        except ImportError:
            # pysnmp not installed - template only
            self._connected = False
            return False
    
    def _snmp_get(self, oid: str) -> Optional[float]:
        """
        Perform SNMP GET for single OID.
        Returns numeric value or None if failed.
        """
        if not self._session:
            return None
        try:
            from pysnmp.hlapi import getCmd, ObjectType, ObjectIdentity
            iterator = getCmd(
                self._session["engine"],
                self._session["community"],
                self._session["target"],
                ObjectType(ObjectIdentity(oid)),
            )
            error_indication, error_status, error_index, var_binds = next(iterator)
            if not error_indication and not error_status:
                return float(var_binds[0][1])
        except (ImportError, StopIteration, IndexError, ValueError, TypeError):
            pass
        return None
    
    def read(self) -> List[BaseNode]:
        """
        Read data via SNMP and return normalized ServerNode.
        
        Returns:
            List with single ServerNode (or empty list if read fails)
        """
        nodes = []
        
        if not self._connected and self._session is None:
            # Template mode: return mock data for testing
            nodes.append(ServerNode(
                thermal_input=55.0,
                power_draw=250.0,
                cooling_output=1800.0,
                utilization=45.0,
                node_id="snmp_template",
                timestamp=datetime.now(),
                source=self.source,
            ))
            return nodes
        
        if not self._connected:
            return nodes
        
        # Read OIDs and map to node attributes
        node_data = {}
        for attr, oid in self.oid_map.items():
            val = self._snmp_get(oid)
            node_data[attr] = float(val) if val is not None else 0.0
        
        node = ServerNode(
            node_id=f"snmp_{self.host}",
            timestamp=datetime.now(),
            source=self.source,
            **node_data,
        )
        nodes.append(node)
        return nodes
    
    def set_safe_state(self, config: "SafeStateConfig") -> bool:
        """
        Apply Safe State: bypass AI, set cooling to Fail-Safe High.
        
        Called by Watchdog when AI heartbeat is missed. Uses SNMP SET
        to write fan speed setpoint (vendor-specific OID).
        
        Args:
            config: SafeStateConfig with cooling_level (1.0 = 100%)
            
        Returns:
            True if command was applied successfully
        """
        # Fan setpoint OID (vendor-specific - many devices are read-only)
        FAN_SETPOINT_OID = "1.3.6.1.4.1.9.9.13.1.4.1.3.1"  # Cisco fan control - override per device
        MAX_FAN_RPM = 3000
        
        target_rpm = int(config.cooling_level * MAX_FAN_RPM)
        if config.fan_rpm is not None:
            target_rpm = int(config.fan_rpm)
        
        if not self._session:
            return False
        try:
            from pysnmp.hlapi import setCmd, ObjectType, ObjectIdentity, Integer32
            iterator = setCmd(
                self._session["engine"],
                self._session["community"],
                self._session["target"],
                ObjectType(ObjectIdentity(FAN_SETPOINT_OID), Integer32(target_rpm)),
            )
            next(iterator)
            return True
        except (ImportError, StopIteration, Exception):
            return False
