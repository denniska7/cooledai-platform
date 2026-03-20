"""
CooledAI Universal Data Ingestion

Provides hardware-agnostic data ingestion from any source.
Uses fuzzy column matching to automatically map vendor-specific column names
(e.g., Tdie, Tctl, CPU Package Power) to normalized BaseNode attributes.

Thermal momentum: compute_derivatives() and TelemetryDerivativeEnricher add
first derivative (dT/dt) for all temperature sensors. InfluenceMap stores
weighted cooling-unit-to-zone relationships for the AI.

Usage:
    from core.ingestion import DataIngestor, compute_derivatives, TelemetryDerivativeEnricher, InfluenceMap
    ingestor = DataIngestor()
    nodes = ingestor.ingest_csv("data/raw/Log-013026.csv")
"""

from core.ingestion.data_ingestor import DataIngestor, apply_telemetry_filters
from core.ingestion.thermal_derivatives import compute_derivatives, TelemetryDerivativeEnricher
from core.ingestion.influence_map import InfluenceMap
from core.ingestion.telemetry_smoothing import EwmaTelemetryFilter, MovingAverageFilter

__all__ = [
    "DataIngestor",
    "apply_telemetry_filters",
    "compute_derivatives",
    "TelemetryDerivativeEnricher",
    "InfluenceMap",
    "EwmaTelemetryFilter",
    "MovingAverageFilter",
]
