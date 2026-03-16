"""
CooledAI Reports - Audit-ready outputs for utility rebates and compliance.

UtilityRebateGenerator: 30-day baseline + financial data formatted into
a professional PDF (PUE, MWh saved, carbon offset, verification log).

ShadowComparisonEngine: Actual vs Theoretical Minimum (RL agent) — Cumulative
Energy Waste and Carbon Offset Potential charts for sales demos.
"""

from reports.utility_rebate_generator import (
    UtilityRebateGenerator,
    generate_utility_rebate_pdf,
    ShadowComparisonEngine,
    ShadowComparisonResult,
)

__all__ = [
    "UtilityRebateGenerator",
    "generate_utility_rebate_pdf",
    "ShadowComparisonEngine",
    "ShadowComparisonResult",
]
