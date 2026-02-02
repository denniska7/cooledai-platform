"""
CooledAI Redfish Manager - Server and Chassis Management

Redfish is the modern REST API for servers, chassis, and power.
Collects thermal and power from Dell, HPE, Supermicro, etc.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_collector import BaseCollector, TelemetryObject, NodeType


class RedfishManager(BaseCollector):
    """
    Collector for Redfish-enabled servers and chassis.
    
    Reads /redfish/v1/Chassis/.../Thermal and Power.
    """
    
    def __init__(
        self,
        base_url: str = "https://192.168.1.100",
        user: str = "",
        password: str = "",
        verify_ssl: bool = True,
        **kwargs,
    ):
        super().__init__(source=base_url, protocol="redfish", **kwargs)
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.password = password
        self.verify_ssl = verify_ssl
        self._session = None
    
    def connect(self) -> bool:
        """Establish Redfish session."""
        try:
            import requests
            self._session = requests.Session()
            self._session.auth = (self.user, self.password)
            self._session.verify = self.verify_ssl
            r = self._session.get(f"{self.base_url}/redfish/v1/")
            self._connected = r.status_code == 200
            return self._connected
        except ImportError:
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False
    
    def _get_json(self, path: str) -> Optional[Dict]:
        """GET Redfish path, return JSON."""
        if not self._session:
            return None
        try:
            r = self._session.get(f"{self.base_url}{path}", timeout=5)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None
    
    def collect(self) -> List[TelemetryObject]:
        """Collect thermal and power from Redfish."""
        objs = []
        
        if not self._session:
            objs.append(TelemetryObject(
                thermal_input=45.0,
                power_draw=0.35,
                cooling_output=2400.0,
                utilization=70.0,
                node_id=f"redfish_{self.base_url.split('//')[-1].split('/')[0]}",
                node_type=NodeType.SERVER,
                timestamp=datetime.now(),
                source=self.source,
                protocol="redfish",
            ))
            return objs
        
        thermal = self._get_json("/redfish/v1/Chassis/1/Thermal")
        power = self._get_json("/redfish/v1/Chassis/1/Power")
        
        thermal_input = 0.0
        if thermal and "Temperatures" in thermal:
            for t in thermal["Temperatures"]:
                if t.get("ReadingCelsius"):
                    thermal_input = max(thermal_input, t["ReadingCelsius"])
                elif t.get("Reading"):
                    thermal_input = max(thermal_input, (t["Reading"] - 32) * 5/9)
        
        power_draw = 0.0
        if power and "PowerControl" in power:
            for p in power["PowerControl"]:
                w = p.get("PowerConsumedWatts") or p.get("PowerMetrics", {}).get("AverageConsumedWatts", 0)
                power_draw += float(w or 0)
        power_draw = power_draw / 1000.0  # W -> kW
        
        cooling_output = 0.0
        if thermal and "Fans" in thermal:
            for f in thermal["Fans"]:
                rpm = f.get("Reading") or f.get("ReadingUnits", 0)
                cooling_output = max(cooling_output, float(rpm or 0))
        
        objs.append(TelemetryObject(
            thermal_input=thermal_input,
            power_draw=power_draw,
            cooling_output=cooling_output,
            utilization=0.0,
            node_id=f"redfish_{self.base_url.split('//')[-1].split('/')[0]}",
            node_type=NodeType.SERVER,
            timestamp=datetime.now(),
            source=self.source,
            protocol="redfish",
        ))
        return objs
    
    def write(self, target: str, value: Any, unit: str = "") -> bool:
        """Redfish PATCH - e.g., fan curve. Called by ControlGate."""
        if not self._session:
            return False
        try:
            path = target or "/redfish/v1/Chassis/1"
            r = self._session.patch(f"{self.base_url}{path}", json=value, timeout=5)
            return r.ok
        except Exception:
            return False
