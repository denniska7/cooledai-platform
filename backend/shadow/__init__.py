"""
CooledAI Shadow Mode - Log proposed actions without sending writes to hardware.

When OptimizationBrain.shadow_mode is True, proposed actions are saved to
shadow_logs table. Use generate_potential_savings_report() to compare
AI Proposed PUE vs Actual PUE. When the AI would have saved > $10, a
'Missed Opportunity' event is logged (FOMO for facility manager); use
get_daily_missed_savings_notification() for the daily Missed Savings digest.
"""

from backend.shadow.shadow_logs import (
    init_shadow_db,
    log_proposed_action,
    get_shadow_logs,
    update_actual_cooling_power,
)
from backend.shadow.savings_report import (
    generate_potential_savings_report,
    PotentialSavingsReport,
)
from backend.shadow.missed_opportunity import (
    log_missed_opportunity,
    get_missed_opportunities,
    get_daily_missed_savings_notification,
    DailyMissedSavingsNotification,
    MissedOpportunityEvent,
    DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD,
)

__all__ = [
    "init_shadow_db",
    "log_proposed_action",
    "get_shadow_logs",
    "update_actual_cooling_power",
    "log_failing_component_event",
    "get_failing_component_events",
    "generate_potential_savings_report",
    "PotentialSavingsReport",
    "log_missed_opportunity",
    "get_missed_opportunities",
    "get_daily_missed_savings_notification",
    "DailyMissedSavingsNotification",
    "MissedOpportunityEvent",
    "DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD",
]
