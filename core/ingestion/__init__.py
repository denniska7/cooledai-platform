"""
CooledAI Universal Data Ingestion

Provides hardware-agnostic data ingestion from any source.
Uses fuzzy column matching to automatically map vendor-specific column names
(e.g., Tdie, Tctl, CPU Package Power) to normalized BaseNode attributes.

Usage:
    from ingestion import DataIngestor
    ingestor = DataIngestor()
    nodes = ingestor.ingest_csv("data/raw/Log-013026.csv")
"""

from core.ingestion.data_ingestor import DataIngestor

__all__ = ["DataIngestor"]
