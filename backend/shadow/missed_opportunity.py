"""
CooledAI Missed Opportunity - FOMO for facility managers.

When Shadow Mode is on and the AI would have made an adjustment that saved more than
a threshold (default $10), we log a 'Missed Opportunity' event with a specific reason
(e.g. 'Over-cooling detected in Zone B'). Supports a daily 'Missed Savings' notification.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.shadow.shadow_logs import DEFAULT_DB_PATH, init_shadow_db
from contextlib import contextmanager
import sqlite3

logger = logging.getLogger(__name__)


def _ensure_data_dir(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn(db_path: Path):
    _ensure_data_dir(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Minimum dollars missed (per hour equivalent) to log a Missed Opportunity event
DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD = 10.0


@dataclass
class MissedOpportunityEvent:
    """Single missed opportunity (AI would have saved > threshold)."""
    id: int
    created_at: datetime
    shadow_log_id: Optional[int]
    dollars_missed: float
    reason: str
    cooling_savings_kw: Optional[float]
    it_power_kw: Optional[float]
    recommendations_snapshot: Optional[List[str]] = None


@dataclass
class DailyMissedSavingsNotification:
    """Daily aggregate: Missed Savings notification for facility manager."""
    date: datetime  # day (UTC)
    total_dollars_missed: float
    event_count: int
    events: List[MissedOpportunityEvent] = field(default_factory=list)
    message: str = ""
    summary: str = ""


def _parse_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.rstrip("Z").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _reason_from_gap(
    recommendations: Optional[List[str]] = None,
    diagnostic_reasoning: str = "",
    raw_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Derive a short, specific reason for the missed opportunity (e.g. 'Over-cooling detected in Zone B').
    """
    # Prefer first recommendation (e.g. "Over-provisioning: 15% excess cooling. Reduce cooling...")
    if recommendations:
        first = (recommendations[0] or "").strip()
        if first:
            # Shorten to one line; optionally add zone from raw_metrics
            zone = None
            if raw_metrics:
                zone = raw_metrics.get("zone_id") or raw_metrics.get("zone")
                if zone is None and isinstance(raw_metrics.get("influence_map"), dict):
                    rz = raw_metrics["influence_map"].get("rack_zones")
                    if isinstance(rz, list) and rz:
                        zone = rz[0]
            if zone:
                return f"{first.split('.')[0].strip()} (Zone {zone})"
            return first.split(".")[0].strip() if "." in first else first[:200]
    # Fallback: first sentence of diagnostic reasoning
    if diagnostic_reasoning:
        return diagnostic_reasoning.split(".")[0].strip()[:200] or "Efficiency opportunity (cooling could be reduced)."
    return "Over-cooling detected. AI would have reduced cooling to save energy."


def log_missed_opportunity(
    shadow_log_id: Optional[int],
    dollars_missed: float,
    reason: str,
    cooling_savings_kw: Optional[float] = None,
    it_power_kw: Optional[float] = None,
    recommendations_snapshot: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Log a 'Missed Opportunity' event (AI would have saved > threshold; adjustment was not applied).
    Returns inserted row id.
    """
    path = db_path or DEFAULT_DB_PATH
    init_shadow_db(path)
    created_at = datetime.utcnow().isoformat() + "Z"
    rec_json = json.dumps(recommendations_snapshot or [])
    with _conn(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO missed_opportunity_events (
                created_at, shadow_log_id, dollars_missed, reason,
                cooling_savings_kw, it_power_kw, recommendations_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, shadow_log_id, round(dollars_missed, 2), reason, cooling_savings_kw, it_power_kw, rec_json),
        )
        row_id = cur.lastrowid
    logger.info("Missed opportunity logged id=%s $%.2f reason=%s", row_id, dollars_missed, reason[:60])
    return row_id or 0


def get_missed_opportunities(
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    limit: int = 1000,
    db_path: Optional[Path] = None,
) -> List[MissedOpportunityEvent]:
    """Return missed opportunity events in the given time range."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    with _conn(path) as conn:
        q = "SELECT * FROM missed_opportunity_events WHERE 1=1"
        params: List[Any] = []
        if start_ts:
            q += " AND created_at >= ?"
            params.append(start_ts.isoformat())
        if end_ts:
            q += " AND created_at <= ?"
            params.append(end_ts.isoformat())
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cur = conn.execute(q, params)
        rows = cur.fetchall()
    out = []
    for r in rows:
        rec = r["recommendations_snapshot"]
        try:
            rec_list = json.loads(rec) if rec else None
        except Exception:
            rec_list = None
        out.append(
            MissedOpportunityEvent(
                id=r["id"],
                created_at=_parse_ts(r["created_at"]) or datetime.utcnow(),
                shadow_log_id=r["shadow_log_id"],
                dollars_missed=float(r["dollars_missed"]),
                reason=r["reason"] or "",
                cooling_savings_kw=float(r["cooling_savings_kw"]) if r["cooling_savings_kw"] is not None else None,
                it_power_kw=float(r["it_power_kw"]) if r["it_power_kw"] is not None else None,
                recommendations_snapshot=rec_list,
            )
        )
    return out


def get_daily_missed_savings_notification(
    date: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> DailyMissedSavingsNotification:
    """
    Generate the daily 'Missed Savings' notification: aggregate of all missed opportunities
    for the given day (UTC). Use for daily digest / FOMO for facility manager.
    """
    if date is None:
        date = datetime.utcnow()
    start_ts = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = start_ts + timedelta(days=1)
    events = get_missed_opportunities(start_ts=start_ts, end_ts=end_ts, limit=500, db_path=db_path)
    total = sum(e.dollars_missed for e in events)
    day_str = start_ts.strftime("%Y-%m-%d")
    if not events:
        return DailyMissedSavingsNotification(
            date=start_ts,
            total_dollars_missed=0.0,
            event_count=0,
            events=[],
            message=f"No missed opportunities recorded for {day_str}. Shadow Mode may be off or no opportunities exceeded the threshold.",
            summary="No missed savings today.",
        )
    n = len(events)
    summary = f"${total:.2f} in missed savings across {n} opportunity today." if n == 1 else f"${total:.2f} in missed savings across {n} opportunities today."
    message = (
        f"Missed Savings ({day_str}): {summary} "
        "Enable CooledAI to apply optimizations and capture these savings. "
        "Top reasons: " + "; ".join(e.reason[:80] for e in events[:3])
    )
    return DailyMissedSavingsNotification(
        date=start_ts,
        total_dollars_missed=round(total, 2),
        event_count=len(events),
        events=events,
        message=message,
        summary=summary,
    )
