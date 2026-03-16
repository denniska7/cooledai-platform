"""
CooledAI Briefing Agent - Morning Briefing (last 24h summary).

Produces a JSON object with: Financial Win, System Health, Watchlist, LLM Summary.
For GET /briefing/daily and Slack/Email notifications.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Watchlist threshold: cooling units with severity_score above this appear in the watchlist
WATCHLIST_SEVERITY_THRESHOLD = 20
# System health: penalty per severity point (sum of severities in 24h); health = max(0, 100 - penalty)
HEALTH_PENALTY_PER_SEVERITY = 0.5


@dataclass
class MorningBriefing:
    """Morning Briefing JSON: last 24 hours summary."""
    period_start_utc: str
    period_end_utc: str
    financial_win: Dict[str, Any]  # total_dollars_saved_yesterday_vs_baseline, etc.
    system_health: Dict[str, Any]  # score_pct, anomaly_count, message
    watchlist: List[Dict[str, Any]]  # items with severity_score > 20
    llm_summary: str  # 3-sentence executive summary
    generated_at_utc: str = ""


def _efficiency_drop_pct(delta_t_prev: float, delta_t_now: float) -> Optional[float]:
    """Efficiency drop as percentage (ΔT decrease)."""
    if delta_t_prev is None or delta_t_prev < 1e-6:
        return None
    if delta_t_now is None:
        return None
    drop = (float(delta_t_prev) - float(delta_t_now)) / float(delta_t_prev) * 100.0
    return round(max(0.0, drop), 1)


def _diagnostic_reasoning_from_event(e: Dict[str, Any]) -> str:
    """Short diagnostic e.g. 'Unit-02 shows 5% efficiency drop'."""
    node_id = e.get("node_id") or "Unit"
    msg = e.get("message") or ""
    delta_t_prev = e.get("delta_t_prev")
    delta_t_now = e.get("delta_t_now")
    drop_pct = _efficiency_drop_pct(delta_t_prev, delta_t_now)
    if drop_pct is not None:
        return f"{node_id} shows {drop_pct}% efficiency drop (power up, ΔT down)."
    if msg:
        return msg[:200] + ("..." if len(msg) > 200 else "")
    return f"{node_id} flagged for predictive maintenance."


def _call_llm_for_summary(context: Dict[str, Any]) -> Optional[str]:
    """Call OpenAI or Anthropic for 3-sentence executive summary. Returns None if no API key or failure."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("COOLEDAI_LLM_MODEL", "gpt-4o-mini")

    prompt = f"""You are a data center facility analyst. Write exactly 3 sentences as an executive summary of this facility's last 24 hours.

Data:
- Financial: {json.dumps(context.get('financial_win', {}), indent=2)}
- System Health: {json.dumps(context.get('system_health', {}), indent=2)}
- Watchlist (units needing attention): {json.dumps(context.get('watchlist_summary', []), indent=2)}

Output only the 3 sentences, no bullet points or headers. Be concise and actionable."""

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text if text else None
        except ImportError:
            logger.warning("openai not installed; pip install openai for LLM summary")
            return None
        except Exception as e:
            logger.warning("OpenAI briefing summary failed: %s", e)
            return None

    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model=model or "claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (msg.content[0].text if msg.content else "").strip()
            return text if text else None
        except ImportError:
            logger.warning("anthropic not installed for LLM summary")
            return None
        except Exception as e:
            logger.warning("Anthropic briefing summary failed: %s", e)
            return None

    return None


def _fallback_llm_summary(briefing: "MorningBriefing") -> str:
    """Template summary when no LLM is configured."""
    dollars = briefing.financial_win.get("total_dollars_saved_yesterday_vs_baseline") or 0
    health = briefing.system_health.get("score_pct") or 0
    n = len(briefing.watchlist)
    s1 = f"Yesterday the facility saved ${dollars:.2f} versus baseline with CooledAI optimization."
    s2 = f"System health is {health:.0f}% based on cooling unit anomalies in the last 24 hours."
    s3 = f"{n} cooling unit(s) are on the watchlist for predictive maintenance." if n else "No cooling units currently on the watchlist."
    return f"{s1} {s2} {s3}"


class BriefingAgent:
    """
    Builds the Morning Briefing for the last 24 hours: Financial Win, System Health,
    Watchlist (severity > 20), and LLM executive summary.
    """

    def __init__(
        self,
        baseline_db_path: Optional[Path] = None,
        shadow_db_path: Optional[Path] = None,
        watchlist_severity_threshold: float = WATCHLIST_SEVERITY_THRESHOLD,
        use_llm: bool = True,
    ):
        self.baseline_db_path = baseline_db_path or Path("data/baseline.db")
        self.shadow_db_path = shadow_db_path or Path("data/shadow_logs.db")
        self.watchlist_severity_threshold = watchlist_severity_threshold
        self.use_llm = use_llm

    def _get_financial_win(self, end_ts: datetime, start_ts: datetime) -> Dict[str, Any]:
        """Total dollars saved yesterday vs. baseline (last 24h)."""
        try:
            from backend.baseline.baseline_reporter import BaselineReporter
            reporter = BaselineReporter(db_path=self.baseline_db_path)
            report = reporter.generate_comparison_report(start_ts=start_ts, end_ts=end_ts, hours=24)
            return {
                "total_dollars_saved_yesterday_vs_baseline": report.dollars_saved_24h,
                "kwh_saved_24h": report.kwh_saved_24h,
                "current_pue": report.current_pue,
                "projected_cooledai_pue": report.projected_cooledai_pue,
                "sample_count": report.sample_count,
            }
        except Exception as e:
            logger.warning("BriefingAgent financial win failed: %s", e)
            return {
                "total_dollars_saved_yesterday_vs_baseline": 0.0,
                "kwh_saved_24h": 0.0,
                "current_pue": None,
                "projected_cooledai_pue": None,
                "sample_count": 0,
                "error": str(e),
            }

    def _get_system_health(self, anomalies_24h: List[Dict[str, Any]]) -> Dict[str, Any]:
        """System Health percentage based on absence of anomalies."""
        if not anomalies_24h:
            return {
                "score_pct": 100.0,
                "anomaly_count": 0,
                "message": "No cooling anomalies in the last 24 hours.",
            }
        total_severity = sum(float(a.get("severity_score") or 0) for a in anomalies_24h)
        penalty = total_severity * HEALTH_PENALTY_PER_SEVERITY
        score = max(0.0, min(100.0, 100.0 - penalty))
        return {
            "score_pct": round(score, 1),
            "anomaly_count": len(anomalies_24h),
            "total_severity_points": round(total_severity, 1),
            "message": f"{len(anomalies_24h)} anomaly/anomalies in last 24h (total severity {total_severity:.0f}).",
        }

    def _get_watchlist(self, anomalies_24h: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cooling units with severity_score > threshold, with diagnostic reasoning."""
        items = [
            a for a in anomalies_24h
            if (float(a.get("severity_score") or 0)) > self.watchlist_severity_threshold
        ]
        out = []
        for e in items:
            delta_t_prev = e.get("delta_t_prev")
            delta_t_now = e.get("delta_t_now")
            drop_pct = _efficiency_drop_pct(delta_t_prev, delta_t_now)
            out.append({
                "node_id": e.get("node_id"),
                "severity_score": float(e.get("severity_score") or 0),
                "diagnostic_reasoning": _diagnostic_reasoning_from_event(e),
                "created_at": e.get("created_at"),
                "efficiency_drop_pct": drop_pct,
            })
        return out

    def generate(self, end_ts: Optional[datetime] = None) -> MorningBriefing:
        """Build the Morning Briefing for the last 24 hours ending at end_ts."""
        end_ts = end_ts or datetime.utcnow()
        start_ts = end_ts - timedelta(hours=24)

        # Failing component events in window (from shadow DB)
        try:
            from backend.shadow.shadow_logs import get_failing_component_events
            anomalies_24h = get_failing_component_events(
                start_ts=start_ts, end_ts=end_ts, limit=500, db_path=self.shadow_db_path
            )
        except Exception as e:
            logger.warning("BriefingAgent get_failing_component_events failed: %s", e)
            anomalies_24h = []

        financial_win = self._get_financial_win(end_ts, start_ts)
        system_health = self._get_system_health(anomalies_24h)
        watchlist = self._get_watchlist(anomalies_24h)

        # LLM summary
        llm_summary = ""
        if self.use_llm:
            context = {
                "financial_win": financial_win,
                "system_health": system_health,
                "watchlist_summary": [{"node_id": w["node_id"], "severity_score": w["severity_score"], "diagnostic_reasoning": w["diagnostic_reasoning"]} for w in watchlist],
            }
            llm_summary = _call_llm_for_summary(context)
        if not llm_summary:
            # Build briefing temporarily to use fallback
            temp = MorningBriefing(
                period_start_utc=start_ts.isoformat() + "Z",
                period_end_utc=end_ts.isoformat() + "Z",
                financial_win=financial_win,
                system_health=system_health,
                watchlist=watchlist,
                llm_summary="",
            )
            llm_summary = _fallback_llm_summary(temp)

        return MorningBriefing(
            period_start_utc=start_ts.isoformat() + "Z",
            period_end_utc=end_ts.isoformat() + "Z",
            financial_win=financial_win,
            system_health=system_health,
            watchlist=watchlist,
            llm_summary=llm_summary,
            generated_at_utc=datetime.utcnow().isoformat() + "Z",
        )


def generate_daily_briefing(
    end_ts: Optional[datetime] = None,
    baseline_db_path: Optional[Path] = None,
    shadow_db_path: Optional[Path] = None,
    use_llm: bool = True,
) -> MorningBriefing:
    """Convenience: generate Morning Briefing for the last 24 hours."""
    agent = BriefingAgent(
        baseline_db_path=baseline_db_path,
        shadow_db_path=shadow_db_path,
        use_llm=use_llm,
    )
    return agent.generate(end_ts=end_ts)
