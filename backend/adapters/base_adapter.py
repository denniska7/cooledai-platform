"""
CooledAI Base Adapter - Adapter Design Pattern

Abstract base class for all protocol adapters.
Each adapter (Modbus, SNMP, CSV, Redfish) implements read() to return
normalized BaseNode instances regardless of the underlying protocol.

This enables hardware-agnostic optimization: the OptimizationBrain
receives consistent data from any source.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.hal.base_node import BaseNode


class BaseAdapter(ABC):
    """
    Abstract base class for protocol adapters.
    
    All adapters must implement:
    - read(): Return list of BaseNode instances
    - connect(): Establish connection to data source (optional)
    - disconnect(): Close connection (optional)
    """
    
    def __init__(self, source: str = "", node_type: type = BaseNode):
        """
        Initialize adapter.
        
        Args:
            source: Data source identifier (host, path, etc.)
            node_type: BaseNode subclass to instantiate
        """
        self.source = source
        self.node_type = node_type
        self._connected = False
    
    @abstractmethod
    def read(self) -> List[BaseNode]:
        """
        Read data from source and return normalized BaseNode instances.
        
        Returns:
            List of BaseNode (or subclass) instances with
            thermal_input, power_draw, cooling_output, utilization populated.
        """
        pass
    
    def connect(self) -> bool:
        """
        Establish connection to data source.
        Override in subclasses that require connection (Modbus, SNMP).
        
        Returns:
            True if connected successfully
        """
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Close connection to data source."""
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected to source."""
        return self._connected
