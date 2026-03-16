"""
CooledAI Potential Savings Report - Compare AI Proposed PUE vs Actual PUE.

Uses shadow_logs data to compute average proposed PUE and average actual PUE
over a time range and produce a Potential Savings Report.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.shadow.shadow_logs import get_shadow_logs, DEFAULT_DB_PATH
from pathlib import Path


@dataclass
class PotentialSavingsReport:
    """Result of comparing proposed vs actual PUE over a period."""
    start_ts: datetime
    end_ts: datetime
    sample_count: int
    avg_proposed_pue: float
    avg_actual_pue: float
    pue_improvement: float          # avg_actual - avg_proposed (positive = AI would save)
    pue_improvement_pct: float     # (actual - proposed) / actual * 100
    potential_energy_savings_pct: float  # same as pue_improvement_pct for facility energy
    total_it_power_kwh: float       # sum(it_power_kw) over interval (approx)
    estimated_cooling_savings_kwh: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def generate_potential_savings_report(
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> PotentialSavingsReport:
    """
    Compare AI Proposed PUE vs Actual PUE from shadow_logs and return a
    Potential Savings Report.

    Uses shadow_logs rows in [start_ts, end_ts]. If not provided, uses last 24 hours.
    """
    path = db_path or DEFAULT_DB_PATH
    if end_ts is None:
        end_ts = datetime.utcnow()
    if start_ts is None:
        start_ts = end_ts - timedelta(hours=24)
    rows = get_shadow_logs(start_ts=start_ts, end_ts=end_ts, db_path=path)
    if not rows:
        return PotentialSavingsReport(
            start_ts=start_ts,
            end_ts=end_ts,
            sample_count=0,
            avg_proposed_pue=0.0,
            avg_actual_pue=0.0,
            pue_improvement=0.0,
            pue_improvement_pct=0.0,
            potential_energy_savings_pct=0.0,
            total_it_power_kwh=0.0,
            estimated_cooling_savings_kwh=0.0,
            message="No shadow log entries in the selected period. Run with Shadow Mode on to collect data.",
        )
    proposed_pues = [r["proposed_pue"] for r in rows if r["proposed_pue"] is not None]
    actual_pues = [r["actual_pue"] for r in rows if r["actual_pue"] is not None]
    it_powers = [r["it_power_kw"] or 0.0 for r in rows]
    cooling_powers = [r["cooling_power_kw"] or 0.0 for r in rows]
    proposed_cooling = [r["proposed_cooling_power_kw"] or 0.0 for r in rows]
    n = len(rows)
    avg_proposed = sum(proposed_pues) / len(proposed_pues) if proposed_pues else 0.0
    avg_actual = sum(actual_pues) / len(actual_pues) if actual_pues else 0.0
    pue_improvement = avg_actual - avg_proposed
    pue_improvement_pct = (pue_improvement / avg_actual * 100.0) if avg_actual > 1e-6 else 0.0
    total_it_kwh = sum(it_powers)
    total_actual_cooling = sum(cooling_powers)
    total_proposed_cooling = sum(proposed_cooling)
    estimated_cooling_savings_kwh = max(0.0, total_actual_cooling - total_proposed_cooling)
    return PotentialSavingsReport(
        start_ts=start_ts,
        end_ts=end_ts,
        sample_count=n,
        avg_proposed_pue=round(avg_proposed, 4),
        avg_actual_pue=round(avg_actual, 4),
        pue_improvement=round(pue_improvement, 4),
        pue_improvement_pct=round(pue_improvement_pct, 2),
        potential_energy_savings_pct=round(pue_improvement_pct, 2),
        total_it_power_kwh=round(total_it_kwh, 2),
        estimated_cooling_savings_kwh=round(estimated_cooling_savings_kwh, 2),
        message=(
            f"Over {n} samples: Actual PUE {avg_actual:.3f} vs Proposed PUE {avg_proposed:.3f}. "
            f"Potential PUE improvement {pue_improvement:.3f} ({pue_improvement_pct:.1f}%)."
        ),
        details={
            "proposed_pue_samples": len(proposed_pues),
            "actual_pue_samples": len(actual_pues),
        },
    )
