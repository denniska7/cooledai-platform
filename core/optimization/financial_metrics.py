"""
CooledAI FinancialMetrics - Convert kWh saved to dollars and ROI for LLM/diagnostics.

Takes kwh_saved from the optimization engine and user-defined utility_rate ($/kWh)
to produce dollars_saved_today and estimated_monthly_roi for API and diagnostic reasoning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FinancialResult:
    """Result of financial computation for one optimization cycle.

    Attributes:
        dollars_saved_this_hour: Estimated $ saved this hour.
        dollars_saved_today: Estimated $ saved today (extrapolated).
        estimated_monthly_roi: Estimated monthly $ savings (30-day extrapolation).
        kwh_saved_this_hour: kWh saved this hour.
        utility_rate_per_kwh: Utility rate used ($/kWh).
    """
    dollars_saved_this_hour: float
    dollars_saved_today: float
    estimated_monthly_roi: float
    kwh_saved_this_hour: float
    utility_rate_per_kwh: float


class FinancialMetrics:
    """Converts kWh saved to dollar amounts for API and LLM diagnostics.

    Uses user-defined utility rate. Extrapolates hourly savings to daily
    and monthly ROI for facility manager visibility.
    """

    def __init__(self, utility_rate: float = 0.12):
        """Initialize with utility rate.

        Args:
            utility_rate: Price per kWh in dollars (e.g. 0.12 for $0.12/kWh).
        """
        self.utility_rate = max(0.0, float(utility_rate))

    def dollars_from_kwh(self, kwh: float) -> float:
        """Convert kWh to dollars at configured utility rate."""
        return kwh * self.utility_rate

    def compute(
        self,
        kwh_saved: float,
        kwh_saved_today_cumulative: Optional[float] = None,
        hours_for_today: float = 24.0,
        days_per_month: float = 30.0,
    ) -> FinancialResult:
        """
        Compute dollars saved and estimated monthly ROI from kWh saved.

        Args:
            kwh_saved: Estimated kWh saved (e.g. per hour from this cycle).
            kwh_saved_today_cumulative: Optional running total of kWh saved today;
                if None, dollars_saved_today = kwh_saved * hours_for_today * rate.
            hours_for_today: Hours to extrapolate for "today" (default 24).
            days_per_month: Days for monthly ROI (default 30).

        Returns:
            FinancialResult with dollars_saved_this_hour, dollars_saved_today,
            estimated_monthly_roi, kwh_saved_this_hour, utility_rate_per_kwh.
        """
        kwh_saved = max(0.0, float(kwh_saved))
        dollars_this_hour = self.dollars_from_kwh(kwh_saved)
        if kwh_saved_today_cumulative is not None:
            kwh_today = max(0.0, float(kwh_saved_today_cumulative))
        else:
            kwh_today = kwh_saved * hours_for_today
        dollars_today = self.dollars_from_kwh(kwh_today)
        # Monthly ROI: extrapolate daily savings to 30 days
        dollars_per_day = dollars_today / (hours_for_today / 24.0) if hours_for_today else 0.0
        estimated_monthly_roi = dollars_per_day * days_per_month
        return FinancialResult(
            dollars_saved_this_hour=round(dollars_this_hour, 2),
            dollars_saved_today=round(dollars_today, 2),
            estimated_monthly_roi=round(estimated_monthly_roi, 2),
            kwh_saved_this_hour=round(kwh_saved, 4),
            utility_rate_per_kwh=self.utility_rate,
        )

    def diagnostic_snippet(self, result: FinancialResult) -> str:
        """Format short sentence for LLM diagnostic reasoning.

        Args:
            result: FinancialResult from compute().

        Returns:
            Empty string if no savings; else e.g. "This adjustment saved $4.50 in the last hour...".
        """
        if result.dollars_saved_this_hour <= 0:
            return ""
        return (
            f" This adjustment is estimated to save ${result.dollars_saved_this_hour:.2f} in the last hour "
            f"(${result.dollars_saved_today:.2f} today at current rate; ~${result.estimated_monthly_roi:.0f}/month ROI)."
        )
