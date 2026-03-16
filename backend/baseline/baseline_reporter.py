"""
CooledAI Baseline Reporter - Comparison report: Current PUE vs Projected CooledAI PUE.

Generates a report with CO2 reduction and estimated utility rebates based on
local $/kWh rates. Uses 24 hours of baseline snapshots (recorded without adjustments).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.baseline.baseline_storage import (
    init_baseline_db,
    record_baseline_snapshot,
    get_baseline_snapshots,
    clear_baseline_snapshots,
    DEFAULT_DB_PATH,
    BASELINE_WINDOW_HOURS,
)


@dataclass
class BaselineComparisonReport:
    """Result of comparing 24h baseline (current state) vs projected CooledAI PUE."""
    current_pue: float
    projected_cooledai_pue: float
    pue_improvement: float
    pue_improvement_pct: float
    current_total_kwh_24h: float
    projected_total_kwh_24h: float
    kwh_saved_24h: float
    co2_reduction_kg: float
    co2_reduction_tonnes: float
    estimated_utility_rebates_usd: float
    dollars_saved_24h: float
    sample_count: int
    start_ts: datetime
    end_ts: datetime
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# Default CO2 factor: kg CO2 per kWh (US grid average ~0.4; vary by region)
DEFAULT_CO2_KG_PER_KWH = 0.4
# Default utility rate $/kWh for savings and rebate estimate
DEFAULT_UTILITY_RATE_PER_KWH = 0.12
# Rebate programs often offer $/kWh saved (e.g. $0.05–0.10/kWh); default as fraction of rate
DEFAULT_REBATE_PER_KWH = 0.05


class BaselineReporter:
    """
    Records 24 hours of 'Current State' data without making adjustments,
    then generates a comparison report: Current PUE vs Projected CooledAI PUE,
    with CO2 reduction and estimated utility rebates.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        utility_rate_per_kwh: float = DEFAULT_UTILITY_RATE_PER_KWH,
        co2_kg_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
        rebate_per_kwh: Optional[float] = None,
        projected_pue_improvement_pct: float = 12.0,
    ):
        """
        Args:
            db_path: SQLite path for baseline_snapshots.
            utility_rate_per_kwh: Local $/kWh for savings and rebate context.
            co2_kg_per_kwh: CO2 emissions factor (kg per kWh) for reduction calculation.
            rebate_per_kwh: $/kWh for utility rebate estimate; if None, uses 0.05.
            projected_pue_improvement_pct: Default assumed PUE improvement with CooledAI (e.g. 12%).
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.utility_rate = utility_rate_per_kwh
        self.co2_kg_per_kwh = co2_kg_per_kwh
        self.rebate_per_kwh = rebate_per_kwh if rebate_per_kwh is not None else DEFAULT_REBATE_PER_KWH
        self.projected_pue_improvement_pct = projected_pue_improvement_pct

    def record_snapshot(
        self,
        it_power_kw: float,
        cooling_power_kw: float,
        pue: Optional[float] = None,
        total_facility_kw: Optional[float] = None,
        ups_losses_kw: float = 0.0,
        lighting_kw: float = 0.0,
        other_kw: float = 0.0,
    ) -> int:
        """
        Record one 'current state' snapshot (no adjustments). Call every 5–10 minutes
        over 24 hours to build the baseline.
        """
        return record_baseline_snapshot(
            it_power_kw=it_power_kw,
            cooling_power_kw=cooling_power_kw,
            pue=pue,
            total_facility_kw=total_facility_kw,
            ups_losses_kw=ups_losses_kw,
            lighting_kw=lighting_kw,
            other_kw=other_kw,
            db_path=self.db_path,
        )

    def get_baseline_summary(
        self,
        hours: float = BASELINE_WINDOW_HOURS,
        end_ts: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return summary of baseline over the last `hours`: avg PUE, total kWh, etc."""
        if end_ts is None:
            end_ts = datetime.utcnow()
        start_ts = end_ts - timedelta(hours=hours)
        rows = get_baseline_snapshots(start_ts=start_ts, end_ts=end_ts, db_path=self.db_path)
        if not rows:
            return {
                "sample_count": 0,
                "avg_pue": 0.0,
                "total_it_kwh": 0.0,
                "total_facility_kwh": 0.0,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        n = len(rows)
        pues = [r["pue"] for r in rows if r["pue"] is not None]
        it_kw_list = [r["it_power_kw"] or 0.0 for r in rows]
        total_kw_list = [r["total_facility_kw"] or (r["it_power_kw"] or 0) + (r["cooling_power_kw"] or 0) for r in rows]
        avg_pue = sum(pues) / len(pues) if pues else 0.0
        avg_it_kw = sum(it_kw_list) / n
        avg_facility_kw = sum(total_kw_list) / n
        def _parse(s: str):
            s = (s or "").rstrip("Z").replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None
        try:
            t_first = _parse(rows[0].get("created_at") or "")
            t_last = _parse(rows[-1].get("created_at") or "")
            if t_first and t_last:
                span_hours = max(0.0, (t_last - t_first).total_seconds() / 3600.0) if n > 1 else hours
            else:
                span_hours = hours
        except Exception:
            span_hours = hours
        if span_hours < 1e-6:
            span_hours = hours
        total_it_kwh = avg_it_kw * span_hours
        total_facility_kwh = avg_facility_kw * span_hours
        return {
            "sample_count": n,
            "avg_pue": round(avg_pue, 4),
            "total_it_kwh": round(total_it_kwh, 2),
            "total_facility_kwh": round(total_facility_kwh, 2),
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

    def generate_comparison_report(
        self,
        projected_pue: Optional[float] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        hours: float = BASELINE_WINDOW_HOURS,
    ) -> BaselineComparisonReport:
        """
        Generate comparison report: Current PUE vs Projected CooledAI PUE,
        with CO2 reduction and estimated utility rebates.

        Args:
            projected_pue: If provided, use as Projected CooledAI PUE; else derive from
                current PUE and projected_pue_improvement_pct.
            start_ts, end_ts: Window for baseline; if None, use last `hours`.
            hours: Default 24.

        Returns:
            BaselineComparisonReport with current_pue, projected_cooledai_pue,
            kwh_saved_24h, co2_reduction_kg, estimated_utility_rebates_usd, etc.
        """
        if end_ts is None:
            end_ts = datetime.utcnow()
        if start_ts is None:
            start_ts = end_ts - timedelta(hours=hours)
        rows = get_baseline_snapshots(start_ts=start_ts, end_ts=end_ts, db_path=self.db_path)
        if not rows:
            return BaselineComparisonReport(
                current_pue=0.0,
                projected_cooledai_pue=0.0,
                pue_improvement=0.0,
                pue_improvement_pct=0.0,
                current_total_kwh_24h=0.0,
                projected_total_kwh_24h=0.0,
                kwh_saved_24h=0.0,
                co2_reduction_kg=0.0,
                co2_reduction_tonnes=0.0,
                estimated_utility_rebates_usd=0.0,
                dollars_saved_24h=0.0,
                sample_count=0,
                start_ts=start_ts,
                end_ts=end_ts,
                message="No baseline data in the selected window. Record 24 hours of current-state snapshots first.",
            )
        n = len(rows)
        # Time span for kWh: use first/last snapshot; fallback to requested hours if single sample
        def _parse_ts(s: str) -> datetime:
            s = (s or "").rstrip("Z").replace("Z", "+00:00")
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.fromisoformat(s)
                except ValueError:
                    continue
            return start_ts
        try:
            t_first = _parse_ts(rows[0].get("created_at") or "")
            t_last = _parse_ts(rows[-1].get("created_at") or "")
            span_hours = max(0.0, (t_last - t_first).total_seconds() / 3600.0) if n > 1 else hours
        except Exception:
            span_hours = hours
        if span_hours < 1e-6:
            span_hours = hours
        pues = [r["pue"] for r in rows if r["pue"] is not None]
        it_kw_list = [r["it_power_kw"] or 0.0 for r in rows]
        total_kw_list = [
            (r["total_facility_kw"] or (r["it_power_kw"] or 0) + (r["cooling_power_kw"] or 0))
            for r in rows
        ]
        current_pue = sum(pues) / len(pues) if pues else 0.0
        avg_it_kw = sum(it_kw_list) / n
        avg_facility_kw = sum(total_kw_list) / n
        total_it_kwh = avg_it_kw * span_hours
        current_total_kwh_24h = avg_facility_kw * span_hours
        if projected_pue is not None:
            proj_pue = projected_pue
        else:
            improvement = self.projected_pue_improvement_pct / 100.0
            proj_pue = current_pue * (1.0 - improvement)
        pue_improvement = current_pue - proj_pue
        pue_improvement_pct = (pue_improvement / current_pue * 100.0) if current_pue > 1e-6 else 0.0
        if total_it_kwh > 1e-6:
            current_facility_kwh = current_total_kwh_24h
            projected_facility_kwh_24h = total_it_kwh * proj_pue
            kwh_saved_24h = max(0.0, current_facility_kwh - projected_facility_kwh_24h)
        else:
            projected_facility_kwh_24h = 0.0
            kwh_saved_24h = 0.0
        co2_kg = kwh_saved_24h * self.co2_kg_per_kwh
        rebates_usd = kwh_saved_24h * self.rebate_per_kwh
        dollars_saved_24h = kwh_saved_24h * self.utility_rate
        return BaselineComparisonReport(
            current_pue=round(current_pue, 4),
            projected_cooledai_pue=round(proj_pue, 4),
            pue_improvement=round(pue_improvement, 4),
            pue_improvement_pct=round(pue_improvement_pct, 2),
            current_total_kwh_24h=round(current_total_kwh_24h, 2),
            projected_total_kwh_24h=round(projected_facility_kwh_24h if total_it_kwh > 1e-6 else 0.0, 2),
            kwh_saved_24h=round(kwh_saved_24h, 2),
            co2_reduction_kg=round(co2_kg, 2),
            co2_reduction_tonnes=round(co2_kg / 1000.0, 4),
            estimated_utility_rebates_usd=round(rebates_usd, 2),
            dollars_saved_24h=round(dollars_saved_24h, 2),
            sample_count=n,
            start_ts=start_ts,
            end_ts=end_ts,
            message=(
                f"Over {n} baseline samples: Current PUE {current_pue:.3f} vs Projected CooledAI PUE {proj_pue:.3f} "
                f"({pue_improvement_pct:.1f}% improvement). "
                f"Estimated {kwh_saved_24h:.0f} kWh saved, {co2_kg:.0f} kg CO2 reduction, "
                f"${rebates_usd:.2f} utility rebates (${dollars_saved_24h:.2f} savings at ${self.utility_rate}/kWh)."
            ),
            details={
                "utility_rate_per_kwh": self.utility_rate,
                "co2_kg_per_kwh": self.co2_kg_per_kwh,
                "rebate_per_kwh": self.rebate_per_kwh,
            },
        )

    def clear_baseline(self) -> int:
        """Clear all baseline snapshots. Returns count deleted."""
        return clear_baseline_snapshots(self.db_path)
