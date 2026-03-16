"""
CooledAI Baseline Reporter - 24h current-state recording and comparison report.

Records 24 hours of 'Current State' data without making adjustments, then
generates a comparison report: Current PUE vs Projected CooledAI PUE, with
CO2 reduction and estimated utility rebates.
"""

from backend.baseline.baseline_storage import (
    init_baseline_db,
    record_baseline_snapshot,
    get_baseline_snapshots,
    clear_baseline_snapshots,
)
from backend.baseline.baseline_reporter import (
    BaselineReporter,
    BaselineComparisonReport,
)

__all__ = [
    "init_baseline_db",
    "record_baseline_snapshot",
    "get_baseline_snapshots",
    "clear_baseline_snapshots",
    "BaselineReporter",
    "BaselineComparisonReport",
]
