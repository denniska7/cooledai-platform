"""
CooledAI Modbus Adapter - Template for Industrial Chiller/CRAC Integration

Modbus is the standard protocol for industrial HVAC equipment (chillers,
CRAC units, BMS systems). This adapter template provides the structure
for reading thermal and power data from Modbus-enabled devices.

Typical Modbus registers for HVAC:
- Temperature: Holding register 0x1000 (return air temp)
- Fan RPM: Holding register 0x1001
- Power kW: Holding register 0x1002
- Supply temp: Holding register 0x1003

Requires: pip install pymodbus
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from backend.safety.watchdog import SafeStateConfig

from core.hal.base_node import BaseNode, CRAC_Unit
from .base_adapter import BaseAdapter


class ModbusAdapter(BaseAdapter):
    """
    Adapter for Modbus RTU/TCP devices (industrial chillers, CRAC units).
    
    Maps Modbus holding/input registers to normalized BaseNode attributes.
    Register addresses are configurable per device vendor.
    
    Usage:
        adapter = ModbusAdapter(host="192.168.1.100", port=502)
        adapter.connect()
        nodes = adapter.read()
    """
    
    # Default register map - override per device vendor
    # Format: attribute -> (register_address, scale_factor, unit)
    DEFAULT_REGISTER_MAP = {
        "thermal_input": (0x1000, 0.1, "°C"),   # Return air temp
        "power_draw": (0x1001, 1.0, "W"),      # Power consumption
        "cooling_output": (0x1002, 1.0, "RPM"), # Fan speed
        "utilization": (0x1003, 0.1, "%"),      # Capacity utilization
    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 502,
        unit_id: int = 1,
        register_map: Optional[Dict[str, tuple]] = None,
        **kwargs,
    ):
        """
        Initialize Modbus adapter.
        
        Args:
            host: Modbus TCP host (or serial port for RTU)
            port: Modbus TCP port (default 502)
            unit_id: Modbus slave/unit ID
            register_map: Override default register addresses per vendor
        """
        super().__init__(source=f"{host}:{port}", node_type=CRAC_Unit, **kwargs)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.register_map = register_map or self.DEFAULT_REGISTER_MAP
        self._client = None  # pymodbus client instance
    
    def connect(self) -> bool:
        """
        Establish Modbus TCP connection.
        Uses pymodbus if available; otherwise returns False.
        """
        try:
            from pymodbus.client import ModbusTcpClient
            self._client = ModbusTcpClient(self.host, port=self.port)
            self._connected = self._client.connect()
            return self._connected
        except ImportError:
            # pymodbus not installed - template only
            self._connected = False
            return False
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Modbus connection failed: {e}") from e
    
    def disconnect(self) -> None:
        """Close Modbus connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        super().disconnect()
    
    def _read_register(self, address: int, count: int = 1) -> Optional[List[int]]:
        """
        Read holding registers from Modbus device.
        Returns None if client not connected.
        """
        if not self._client or not self._connected:
            return None
        try:
            result = self._client.read_holding_registers(address, count, slave=self.unit_id)
            if result and not result.isError():
                return result.registers
        except Exception:
            pass
        return None
    
    def read(self) -> List[BaseNode]:
        """
        Read data from Modbus device and return normalized CRAC_Unit.
        
        Returns:
            List with single CRAC_Unit (or empty list if read fails)
        """
        nodes = []
        
        if not self._connected and self._client is None:
            # Template mode: return mock data for testing
            nodes.append(CRAC_Unit(
                thermal_input=25.0,
                power_draw=5000.0,
                cooling_output=1200.0,
                utilization=65.0,
                node_id="modbus_template",
                timestamp=datetime.now(),
                source=self.source,
            ))
            return nodes
        
        if not self._connected:
            return nodes
        
        # Read registers and map to node attributes
        node_data = {}
        for attr, (addr, scale, _) in self.register_map.items():
            regs = self._read_register(addr)
            if regs:
                node_data[attr] = float(regs[0]) * scale
            else:
                node_data[attr] = 0.0
        
        node = CRAC_Unit(
            node_id=f"modbus_{self.host}",
            timestamp=datetime.now(),
            source=self.source,
            **node_data,
        )
        nodes.append(node)
        return nodes
    
    def set_safe_state(self, config: "SafeStateConfig") -> bool:
        """
        Apply Safe State: bypass AI, set cooling to Fail-Safe High.
        
        Called by Watchdog when AI heartbeat is missed. Writes cooling
        setpoint to Modbus holding register (vendor-specific address).
        
        Args:
            config: SafeStateConfig with cooling_level (1.0 = 100%)
            
        Returns:
            True if command was applied successfully
        """
        # Fan control register (vendor-specific - override per device)
        FAN_SETPOINT_REGISTER = 0x2000
        MAX_FAN_RPM = 3000  # Device max - override per hardware
        
        target_rpm = int(config.cooling_level * MAX_FAN_RPM)
        if config.fan_rpm is not None:
            target_rpm = int(config.fan_rpm)
        
        if not self._client or not self._connected:
            return False
        try:
            from pymodbus.payload import BinaryPayloadBuilder
            self._client.write_register(FAN_SETPOINT_REGISTER, target_rpm, slave=self.unit_id)
            return True
        except Exception:
            return False
