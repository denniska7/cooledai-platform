"""
CooledAI Core Models - Topology and Physics

Rack model with ThermalMass, FOPDT (First-Order Plus Dead Time) for
thermal dynamics, and self-calibration from telemetry.
"""

from .topology import Rack, FOPDTParams, RackRegistry

__all__ = ["Rack", "FOPDTParams", "RackRegistry"]
