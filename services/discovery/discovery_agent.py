"""
CooledAI Discovery Agent - Network Scanner & Thermal Ping

Scans IP ranges for Redfish (443) and SNMP (161) ports. When a device is found,
queries Chassis and Model info. Runs a Thermal Ping test: momentarily increase
a fan's speed, observe which racks see a temperature drop, and use that
correlation to automatically build the Influence Map.
"""

import ipaddress
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

_logger = logging.getLogger(__name__)

# Default config path
_DISCOVERY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "core" / "config" / "discovery_config.yaml"


@dataclass
class DiscoveredDevice:
    """A device found by network scan with chassis/model info."""

    ip: str
    protocol: str  # "redfish" | "snmp"
    port: int
    chassis_id: Optional[str] = None
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    name: Optional[str] = None
    raw_info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.protocol}:{self.ip}:{self.port} ({self.manufacturer or '?'} {self.model or '?'})"


@dataclass
class DiscoveryResult:
    """Result of full discovery run: devices + optional Influence Map."""

    devices: List[DiscoveredDevice]
    influence_map: Optional[Any] = None  # InfluenceMap from core.ingestion
    thermal_ping_results: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


class DiscoveryAgent:
    """
    Network scanner that discovers Redfish/SNMP devices, queries Chassis/Model,
    and runs Thermal Ping to build Influence Map automatically.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        ip_ranges: Optional[List[str]] = None,
        redfish_ports: Optional[List[int]] = None,
        snmp_ports: Optional[List[int]] = None,
        redfish_user: str = "",
        redfish_password: str = "",
        redfish_verify_ssl: bool = False,
        snmp_community: str = "public",
        port_timeout: float = 2.0,
        max_concurrent: int = 32,
    ):
        self._config_path = config_path or _DISCOVERY_CONFIG_PATH
        self._config = self._load_config()

        self.ip_ranges = ip_ranges or self._config.get("ip_ranges") or ["192.168.1.0/24"]
        self.redfish_ports = redfish_ports or self._config.get("redfish_ports") or [443]
        self.snmp_ports = snmp_ports or self._config.get("snmp_ports") or [161]
        self.redfish_user = redfish_user or self._config.get("redfish_user") or ""
        self.redfish_password = redfish_password or self._config.get("redfish_password") or ""
        self.redfish_verify_ssl = redfish_verify_ssl if "redfish_verify_ssl" not in self._config else self._config["redfish_verify_ssl"]
        self.snmp_community = snmp_community or self._config.get("snmp_community") or "public"
        self.port_timeout = port_timeout or self._config.get("port_timeout") or 2.0
        self.max_concurrent = max_concurrent or self._config.get("max_concurrent") or 32

    def _load_config(self) -> Dict[str, Any]:
        """Load discovery config from YAML."""
        if not self._config_path.exists() or yaml is None:
            return {}
        try:
            with open(self._config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            _logger.warning("DiscoveryAgent: Could not load config %s: %s", self._config_path, e)
            return {}

    def _expand_ip_range(self, spec: str) -> List[str]:
        """Expand IP range spec (CIDR or start-end) to list of IPs."""
        ips: List[str] = []
        spec = spec.strip()
        if "-" in spec and "/" not in spec:
            try:
                start_s, end_s = spec.split("-", 1)
                start_ip = ipaddress.ip_address(start_s.strip())
                end_ip = ipaddress.ip_address(end_s.strip())
                if start_ip.version != end_ip.version:
                    return []
                while start_ip <= end_ip:
                    ips.append(str(start_ip))
                    start_ip += 1
            except ValueError:
                return []
        else:
            try:
                network = ipaddress.ip_network(spec, strict=False)
                ips = [str(ip) for ip in network.hosts()]
            except ValueError:
                try:
                    ip = ipaddress.ip_address(spec)
                    ips = [str(ip)]
                except ValueError:
                    pass
        return ips

    def _probe_port(self, ip: str, port: int) -> bool:
        """Probe if port is open (TCP for Redfish, UDP for SNMP)."""
        try:
            if port == 161:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.port_timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _scan_ips_for_port(self, ips: List[str], port: int) -> List[str]:
        """Return list of IPs where port is open."""
        found: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            futures = {ex.submit(self._probe_port, ip, port): ip for ip in ips}
            for future in as_completed(futures):
                ip = futures[future]
                try:
                    if future.result():
                        found.append(ip)
                except Exception:
                    pass
        return found

    def scan_network(self) -> List[Tuple[str, int, str]]:
        """
        Scan IP ranges for Redfish and SNMP ports.

        Returns:
            List of (ip, port, protocol) tuples for discovered endpoints.
        """
        all_ips: List[str] = []
        for spec in self.ip_ranges:
            all_ips.extend(self._expand_ip_range(spec))
        all_ips = list(dict.fromkeys(all_ips))

        endpoints: List[Tuple[str, int, str]] = []

        for ip in all_ips:
            for port in self.redfish_ports:
                if self._probe_port(ip, port):
                    endpoints.append((ip, port, "redfish"))

        for ip in all_ips:
            for port in self.snmp_ports:
                if self._probe_port(ip, port):
                    endpoints.append((ip, port, "snmp"))

        return endpoints

    def _query_redfish_chassis(self, ip: str, port: int) -> Optional[DiscoveredDevice]:
        """Query Redfish /redfish/v1/Chassis for Model, Manufacturer."""
        base_url = f"https://{ip}:{port}" if port != 443 else f"https://{ip}"
        try:
            import requests
            session = requests.Session()
            session.auth = (self.redfish_user, self.redfish_password)
            session.verify = self.redfish_verify_ssl
            session.headers.update({"Accept": "application/json"})

            r = session.get(f"{base_url}/redfish/v1/", timeout=5)
            if r.status_code != 200:
                return DiscoveredDevice(ip=ip, protocol="redfish", port=port)

            chassis_url = r.json().get("Chassis", {}).get("@odata.id")
            if not chassis_url:
                return DiscoveredDevice(ip=ip, protocol="redfish", port=port)

            r2 = session.get(f"{base_url}{chassis_url}", timeout=5)
            if r2.status_code != 200:
                return DiscoveredDevice(ip=ip, protocol="redfish", port=port)

            chassis_collection = r2.json()
            members = chassis_collection.get("Members", [])
            if not members:
                return DiscoveredDevice(ip=ip, protocol="redfish", port=port, raw_info=chassis_collection)

            first = members[0]
            member_url = first.get("@odata.id", "")
            r3 = session.get(f"{base_url}{member_url}", timeout=5)
            if r3.status_code != 200:
                return DiscoveredDevice(ip=ip, protocol="redfish", port=port)

            ch = r3.json()
            return DiscoveredDevice(
                ip=ip,
                protocol="redfish",
                port=port,
                chassis_id=ch.get("Id"),
                model=ch.get("Model"),
                manufacturer=ch.get("Manufacturer"),
                name=ch.get("Name"),
                raw_info={"odata_type": ch.get("@odata.type")},
            )
        except ImportError:
            _logger.warning("DiscoveryAgent: requests not installed; pip install requests")
            return DiscoveredDevice(ip=ip, protocol="redfish", port=port)
        except Exception as e:
            _logger.debug("DiscoveryAgent: Redfish query failed for %s:%s: %s", ip, port, e)
            return DiscoveredDevice(ip=ip, protocol="redfish", port=port)

    def _query_snmp_sysdescr(self, ip: str, port: int) -> Optional[DiscoveredDevice]:
        """Query SNMP sysDescr (1.3.6.1.2.1.1.1.0) for device info."""
        try:
            from pysnmp.hlapi import (
                getCmd,
                SnmpEngine,
                CommunityData,
                UdpTransportTarget,
                ObjectType,
                ObjectIdentity,
            )
        except ImportError:
            _logger.warning("DiscoveryAgent: pysnmp not installed; pip install pysnmp-lextudio")
            return DiscoveredDevice(ip=ip, protocol="snmp", port=port)

        try:
            iterator = getCmd(
                SnmpEngine(),
                CommunityData(self.snmp_community),
                UdpTransportTarget((ip, port), timeout=self.port_timeout),
                ObjectType(ObjectIdentity("1.3.6.1.2.1.1.1.0")),
            )
            error_indication, error_status, error_index, var_binds = next(iterator)
            if error_indication or error_status:
                return DiscoveredDevice(ip=ip, protocol="snmp", port=port)

            for var_bind in var_binds:
                val = str(var_bind[1]) if var_bind[1] else ""
                return DiscoveredDevice(
                    ip=ip,
                    protocol="snmp",
                    port=port,
                    model=val[:200] if val else None,
                    manufacturer=None,
                    raw_info={"sysDescr": val},
                )
        except Exception as e:
            _logger.debug("DiscoveryAgent: SNMP query failed for %s:%s: %s", ip, port, e)

        return DiscoveredDevice(ip=ip, protocol="snmp", port=port)

    def query_devices(self, endpoints: Optional[List[Tuple[str, int, str]]] = None) -> List[DiscoveredDevice]:
        """
        Query Chassis/Model for each endpoint. If endpoints is None, runs scan first.
        """
        if endpoints is None:
            endpoints = self.scan_network()

        devices: List[DiscoveredDevice] = []
        for ip, port, protocol in endpoints:
            if protocol == "redfish":
                dev = self._query_redfish_chassis(ip, port)
            else:
                dev = self._query_snmp_sysdescr(ip, port)
            if dev:
                devices.append(dev)

        return devices

    def discover(self, scan_only: bool = False) -> DiscoveryResult:
        """
        Full discovery: scan network, query devices, optionally run thermal ping.

        Args:
            scan_only: If True, skip thermal ping (scan and query only).
        """
        errors: List[str] = []
        devices: List[DiscoveredDevice] = []

        endpoints = self.scan_network()
        _logger.info("DiscoveryAgent: Found %d endpoints", len(endpoints))

        for ip, port, protocol in endpoints:
            try:
                if protocol == "redfish":
                    dev = self._query_redfish_chassis(ip, port)
                else:
                    dev = self._query_snmp_sysdescr(ip, port)
                if dev:
                    devices.append(dev)
            except Exception as e:
                errors.append(f"{ip}:{port}/{protocol}: {e}")

        influence_map = None
        thermal_ping_results = None

        cooling_units = self._config.get("cooling_units") or {}
        rack_sensors = self._config.get("rack_sensors") or {}
        if not scan_only and cooling_units and rack_sensors:
            try:
                influence_map, thermal_ping_results = self.run_thermal_ping(
                    cooling_units=cooling_units,
                    rack_sensors=rack_sensors,
                )
            except Exception as e:
                _logger.exception("DiscoveryAgent: Thermal ping failed: %s", e)
                errors.append(f"Thermal ping: {e}")

        return DiscoveryResult(
            devices=devices,
            influence_map=influence_map,
            thermal_ping_results=thermal_ping_results,
            errors=errors,
        )

    def run_thermal_ping(
        self,
        cooling_units: Optional[Dict[str, Dict[str, Any]]] = None,
        rack_sensors: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        fan_boost_percent: float = 20,
        boost_duration_seconds: float = 15,
        settle_before_seconds: float = 5,
        settle_after_seconds: float = 10,
        min_temp_drop_celsius: float = 0.3,
        influence_coefficient: float = 0.8,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Thermal Ping: For each cooling unit, momentarily boost fan speed,
        measure temperature change at each rack. Racks that see a drop
        are influenced by that cooling unit. Builds InfluenceMap.

        Returns:
            (InfluenceMap, results_dict)
        """
        from core.ingestion.influence_map import InfluenceMap

        cooling_units = cooling_units or self._config.get("cooling_units") or {}
        rack_sensors = rack_sensors or self._config.get("rack_sensors") or {}

        tp_config = self._config.get("thermal_ping") or {}
        fan_boost_percent = tp_config.get("fan_boost_percent", fan_boost_percent)
        boost_duration_seconds = tp_config.get("boost_duration_seconds", boost_duration_seconds)
        settle_before_seconds = tp_config.get("settle_before_seconds", settle_before_seconds)
        settle_after_seconds = tp_config.get("settle_after_seconds", settle_after_seconds)
        min_temp_drop_celsius = tp_config.get("min_temp_drop_celsius", min_temp_drop_celsius)
        influence_coefficient = tp_config.get("influence_coefficient", influence_coefficient)

        influence_map = InfluenceMap()
        results: Dict[str, Any] = {
            "cooling_units_tested": [],
            "rack_deltas": {},
            "influence_edges": [],
        }

        def get_rack_temp(rack_id: str) -> Optional[float]:
            """Read temperature from rack sensors (max of all sensors)."""
            sensors = rack_sensors.get(rack_id, [])
            temps: List[float] = []
            for s in sensors:
                ip = s.get("ip")
                protocol = s.get("protocol", "redfish")
                if not ip:
                    continue
                t = self._read_temperature(ip, protocol)
                if t is not None:
                    temps.append(t)
            return max(temps) if temps else None

        baseline_percent = tp_config.get("baseline_percent", 50)
        restore_percent = tp_config.get("restore_percent", 50)

        def set_fan_percent(cooling_unit_id: str, percent: int) -> bool:
            """Set fan to given percent on cooling unit."""
            cu = cooling_units.get(cooling_unit_id, {})
            ip = cu.get("ip")
            protocol = cu.get("protocol", "redfish")
            if not ip:
                return False
            return self._set_fan_percent(ip, protocol, percent)

        for cooling_unit_id, cu_config in cooling_units.items():
            ip = cu_config.get("ip")
            if not ip:
                continue

            results["cooling_units_tested"].append(cooling_unit_id)

            baseline_temps: Dict[str, float] = {}
            time.sleep(settle_before_seconds)
            for rack_id in rack_sensors:
                t = get_rack_temp(rack_id)
                if t is not None:
                    baseline_temps[rack_id] = t

            if not baseline_temps:
                _logger.warning("DiscoveryAgent: No baseline temps for cooling unit %s", cooling_unit_id)
                continue

            set_fan_percent(cooling_unit_id, baseline_percent + fan_boost_percent)
            time.sleep(boost_duration_seconds)

            boosted_temps: Dict[str, float] = {}
            for rack_id in rack_sensors:
                t = get_rack_temp(rack_id)
                if t is not None:
                    boosted_temps[rack_id] = t

            set_fan_percent(cooling_unit_id, restore_percent)

            time.sleep(settle_after_seconds)

            for rack_id, baseline in baseline_temps.items():
                boosted = boosted_temps.get(rack_id)
                if boosted is None:
                    continue
                delta = baseline - boosted
                if rack_id not in results["rack_deltas"]:
                    results["rack_deltas"][rack_id] = {}
                results["rack_deltas"][rack_id][cooling_unit_id] = delta

                if delta >= min_temp_drop_celsius:
                    influence_map.set_influence(cooling_unit_id, rack_id, influence_coefficient)
                    results["influence_edges"].append(
                        {"cooling_unit": cooling_unit_id, "rack": rack_id, "delta_c": delta}
                    )

        return influence_map, results

    def _read_temperature(self, ip: str, protocol: str) -> Optional[float]:
        """Read temperature from device (Redfish or SNMP)."""
        if protocol == "redfish":
            return self._redfish_get_temperature(ip)
        if protocol == "snmp":
            return self._snmp_get_temperature(ip)
        return None

    def _redfish_get_temperature(self, ip: str) -> Optional[float]:
        """Read max temperature from Redfish Thermal."""
        try:
            import requests
            base_url = f"https://{ip}"
            session = requests.Session()
            session.auth = (self.redfish_user, self.redfish_password)
            session.verify = self.redfish_verify_ssl

            thermal = session.get(f"{base_url}/redfish/v1/Chassis/1/Thermal", timeout=5)
            if thermal.status_code != 200:
                return None
            data = thermal.json()
            temps = data.get("Temperatures", [])
            max_t = 0.0
            for t in temps:
                rc = t.get("ReadingCelsius")
                r = t.get("Reading")
                if rc is not None:
                    max_t = max(max_t, float(rc))
                elif r is not None:
                    max_t = max(max_t, (float(r) - 32) * 5 / 9)
            return max_t if max_t > 0 else None
        except Exception:
            return None

    def _snmp_get_temperature(self, ip: str) -> Optional[float]:
        """Read temperature from SNMP (Cisco env temp OID)."""
        try:
            from pysnmp.hlapi import (
                getCmd,
                SnmpEngine,
                CommunityData,
                UdpTransportTarget,
                ObjectType,
                ObjectIdentity,
            )
            iterator = getCmd(
                SnmpEngine(),
                CommunityData(self.snmp_community),
                UdpTransportTarget((ip, 161), timeout=self.port_timeout),
                ObjectType(ObjectIdentity("1.3.6.1.4.1.9.9.13.1.3.1.3.1")),
            )
            _, _, _, var_binds = next(iterator)
            for v in var_binds:
                val = float(v[1])
                return val
        except Exception:
            pass
        return None

    def _set_fan_percent(self, ip: str, protocol: str, percent: int) -> bool:
        """Set fan to percent (Redfish CoolingPercent or SNMP)."""
        if protocol == "redfish":
            return self._redfish_set_fan_percent(ip, percent)
        if protocol == "snmp":
            return self._snmp_set_fan_percent(ip, percent)
        return False

    def _redfish_set_fan_percent(self, ip: str, percent: int) -> bool:
        """PATCH Redfish Thermal with CoolingPercent."""
        try:
            import requests
            base_url = f"https://{ip}"
            session = requests.Session()
            session.auth = (self.redfish_user, self.redfish_password)
            session.verify = self.redfish_verify_ssl
            payload = {"CoolingPercent": min(100, max(0, percent))}
            r = session.patch(
                f"{base_url}/redfish/v1/Chassis/1/Thermal",
                json=payload,
                timeout=5,
            )
            return r.ok
        except Exception:
            return False

    def _snmp_set_fan_percent(self, ip: str, percent: int) -> bool:
        """SNMP SET fan setpoint (Cisco-style OID)."""
        try:
            from pysnmp.hlapi import (
                setCmd,
                SnmpEngine,
                CommunityData,
                UdpTransportTarget,
                ObjectType,
                ObjectIdentity,
                Integer32,
            )
            rpm = int(percent * 50)
            iterator = setCmd(
                SnmpEngine(),
                CommunityData(self.snmp_community),
                UdpTransportTarget((ip, 161), timeout=self.port_timeout),
                ObjectType(ObjectIdentity("1.3.6.1.4.1.9.9.13.1.4.1.3.1"), Integer32(rpm)),
            )
            next(iterator)
            return True
        except Exception:
            return False

    def save_influence_map(self, influence_map: Any, path: Optional[Path] = None) -> bool:
        """
        Save InfluenceMap to YAML for plug-and-play. The optimization brain
        can load this via InfluenceMap.from_dict().
        """
        if influence_map is None:
            return False
        out_path = path or Path(__file__).resolve().parents[2] / "core" / "config" / "influence_map.yaml"
        try:
            data = influence_map.to_dict()
            with open(out_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            _logger.info("DiscoveryAgent: Saved InfluenceMap to %s", out_path)
            return True
        except Exception as e:
            _logger.error("DiscoveryAgent: Failed to save InfluenceMap: %s", e)
            return False


def main() -> None:
    """CLI entry point: run discovery and optionally save Influence Map."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="CooledAI Discovery Agent - Network scan & Thermal Ping")
    parser.add_argument("--config", type=Path, default=_DISCOVERY_CONFIG_PATH, help="Discovery config path")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and query devices, skip thermal ping")
    parser.add_argument("--thermal-ping-only", action="store_true", help="Only run thermal ping (use existing config)")
    parser.add_argument("--save", action="store_true", help="Save discovered Influence Map to core/config/influence_map.yaml")
    parser.add_argument("--output", type=Path, help="Custom path for Influence Map YAML")
    args = parser.parse_args()

    agent = DiscoveryAgent(config_path=args.config)

    if args.thermal_ping_only:
        result = DiscoveryResult(devices=[], errors=[])
        try:
            influence_map, tp_results = agent.run_thermal_ping()
            result.influence_map = influence_map
            result.thermal_ping_results = tp_results
            print("Thermal Ping complete. Influence edges:", len(tp_results.get("influence_edges", [])))
            if args.save and influence_map:
                agent.save_influence_map(influence_map, args.output)
        except Exception as e:
            print(f"Thermal Ping failed: {e}")
            raise
    else:
        result = agent.discover(scan_only=args.scan_only)
        print(f"Discovered {len(result.devices)} devices")
        for d in result.devices:
            print(f"  {d}")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.influence_map:
            print(f"Influence Map: {len(result.influence_map.cooling_units)} units, {len(result.influence_map.rack_zones)} zones")
            if args.save:
                agent.save_influence_map(result.influence_map, args.output)

    if result.influence_map and args.save:
        print("Influence Map saved. Restart API to load.")


if __name__ == "__main__":
    main()
