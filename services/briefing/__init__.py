"""
CooledAI Briefing Agent - Morning Briefing summary for facility managers.

Summarizes the last 24 hours into a JSON object: Financial Win, System Health,
Watchlist (cooling units with severity > 20), and an LLM executive summary.
Feeds into GET /briefing/daily for Slack/Email notifications.
"""

from services.briefing.briefing_agent import (
    BriefingAgent,
    MorningBriefing,
    generate_daily_briefing,
)

__all__ = [
    "BriefingAgent",
    "MorningBriefing",
    "generate_daily_briefing",
]
