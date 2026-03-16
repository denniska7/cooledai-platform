"""
CooledAI Simulation - Digital Twin for Offline Testing

Physics-based simulation of data center thermal dynamics.
Enables stress testing of OptimizationBrain and RL agents without hardware.
"""

from .digital_twin import DigitalTwin, DigitalTwinConfig

__all__ = ["DigitalTwin", "DigitalTwinConfig"]
