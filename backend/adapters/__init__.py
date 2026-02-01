"""
CooledAI Protocol Adapters

Adapter Design Pattern: Each adapter translates a specific protocol
(Modbus, SNMP, CSV, Redfish) into normalized BaseNode data.

All adapters implement a common interface: read() returns list of BaseNode.
This allows the optimization engine to operate on any data source
without knowing the underlying protocol.

Usage:
    from adapters import ModbusAdapter, SNMPAdapter, CSVAdapter
"""

from .base_adapter import BaseAdapter
from .modbus_adapter import ModbusAdapter
from .snmp_adapter import SNMPAdapter

__all__ = ["BaseAdapter", "ModbusAdapter", "SNMPAdapter"]
