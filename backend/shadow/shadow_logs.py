"""
CooledAI Shadow Logs - Persist proposed actions when ShadowMode is active.

Stores each optimization cycle's proposed action (recommended_cooling_delta, context)
so we can compare AI Proposed PUE vs Actual PUE for a Potential Savings Report.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/shadow_logs.db")


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


def init_shadow_db(db_path: Optional[Path] = None) -> None:
    """Create shadow_logs table if it does not exist."""
    path = db_path or DEFAULT_DB_PATH
    with _conn(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                recommended_cooling_delta REAL NOT NULL,
                current_cooling REAL NOT NULL,
                max_cooling REAL NOT NULL,
                current_thermal REAL,
                power_draw REAL,
                it_power_kw REAL,
                cooling_power_kw REAL,
                proposed_cooling REAL,
                proposed_cooling_power_kw REAL,
                proposed_pue REAL,
                actual_pue REAL,
                diagnostic_reasoning TEXT,
                recommendations_json TEXT,
                raw_metrics_json TEXT,
                system_state TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_logs_created_at ON shadow_logs(created_at)")
        # Missed Opportunity events: when AI would have saved > $10 (FOMO for facility manager)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS missed_opportunity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                shadow_log_id INTEGER,
                dollars_missed REAL NOT NULL,
                reason TEXT NOT NULL,
                cooling_savings_kw REAL,
                it_power_kw REAL,
                recommendations_snapshot TEXT,
                FOREIGN KEY (shadow_log_id) REFERENCES shadow_logs(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_missed_opp_created_at ON missed_opportunity_events(created_at)")
        # Failing component findings (historical) for Maintenance Task List / GET /health/anomalies
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failing_component_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                node_id TEXT NOT NULL,
                message TEXT NOT NULL,
                power_draw_prev REAL NOT NULL,
                power_draw_now REAL NOT NULL,
                delta_t_prev REAL NOT NULL,
                delta_t_now REAL NOT NULL,
                severity_score REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_failing_component_created_at ON failing_component_events(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_failing_component_node_id ON failing_component_events(node_id)")


def log_proposed_action(
    recommended_cooling_delta: float,
    current_cooling: float,
    max_cooling: float,
    current_thermal: Optional[float] = None,
    power_draw: Optional[float] = None,
    it_power_kw: Optional[float] = None,
    cooling_power_kw: Optional[float] = None,
    diagnostic_reasoning: str = "",
    recommendations: Optional[List[str]] = None,
    raw_metrics: Optional[Dict[str, Any]] = None,
    system_state: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Append one proposed action to shadow_logs (call when ShadowMode is active).
    Computes proposed_cooling, proposed_cooling_power_kw, proposed_pue.
    Returns inserted row id.
    """
    path = db_path or DEFAULT_DB_PATH
    init_shadow_db(path)
    created_at = datetime.utcnow().isoformat() + "Z"
    proposed_cooling = current_cooling * (1.0 + recommended_cooling_delta)
    proposed_cooling = max(0.0, min(proposed_cooling, max_cooling))
    if it_power_kw is None and power_draw is not None:
        it_power_kw = power_draw / 1000.0
    it_kw = it_power_kw or 0.0
    # Estimate cooling power if not provided: assume cooling is ~25% of IT at full cooling
    cooling_kw = cooling_power_kw
    if cooling_kw is None and it_kw > 1e-6 and max_cooling > 1e-6:
        cooling_kw = (current_cooling / max_cooling) * (it_kw * 0.35)
    if current_cooling > 1e-6 and cooling_kw is not None:
        proposed_cooling_power_kw = cooling_kw * (proposed_cooling / current_cooling)
    else:
        proposed_cooling_power_kw = (proposed_cooling / max_cooling) * (it_kw * 0.35) if max_cooling > 1e-6 else 0.0
    proposed_pue = (it_kw + proposed_cooling_power_kw) / it_kw if it_kw > 1e-6 else None
    actual_pue = (it_kw + (cooling_kw or 0.0)) / it_kw if it_kw > 1e-6 else None
    rec_json = json.dumps(recommendations or [])
    raw_json = json.dumps(raw_metrics or {}, default=str)
    with _conn(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO shadow_logs (
                created_at, recommended_cooling_delta, current_cooling, max_cooling,
                current_thermal, power_draw, it_power_kw, cooling_power_kw,
                proposed_cooling, proposed_cooling_power_kw, proposed_pue, actual_pue,
                diagnostic_reasoning, recommendations_json, raw_metrics_json, system_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                recommended_cooling_delta,
                current_cooling,
                max_cooling,
                current_thermal,
                power_draw,
                it_kw,
                cooling_kw,
                proposed_cooling,
                proposed_cooling_power_kw,
                proposed_pue,
                actual_pue,
                diagnostic_reasoning,
                rec_json,
                raw_json,
                system_state,
            ),
        )
        row_id = cur.lastrowid
    logger.debug("Shadow log inserted id=%s proposed_pue=%s", row_id, proposed_pue)
    return row_id or 0


def get_shadow_logs(
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    limit: int = 10000,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return shadow_logs rows in the given time range."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    with _conn(path) as conn:
        q = "SELECT * FROM shadow_logs WHERE 1=1"
        params: List[Any] = []
        if start_ts:
            q += " AND created_at >= ?"
            params.append(start_ts.isoformat())
        if end_ts:
            q += " AND created_at <= ?"
            params.append(end_ts.isoformat())
        q += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        cur = conn.execute(q, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def update_actual_cooling_power(
    row_id: int,
    actual_cooling_power_kw: float,
    db_path: Optional[Path] = None,
) -> None:
    """Update a shadow_logs row with measured actual cooling power and recompute actual_pue."""
    path = db_path or DEFAULT_DB_PATH
    with _conn(path) as conn:
        cur = conn.execute(
            "SELECT it_power_kw FROM shadow_logs WHERE id = ?", (row_id,)
        )
        row = cur.fetchone()
        if not row:
            return
        it_kw = row["it_power_kw"] or 0.0
        actual_pue = (it_kw + actual_cooling_power_kw) / it_kw if it_kw > 1e-6 else None
        conn.execute(
            "UPDATE shadow_logs SET cooling_power_kw = ?, actual_pue = ? WHERE id = ?",
            (actual_cooling_power_kw, actual_pue, row_id),
        )


def log_failing_component_event(
    node_id: str,
    message: str,
    power_draw_prev: float,
    power_draw_now: float,
    delta_t_prev: float,
    delta_t_now: float,
    severity_score: float,
    db_path: Optional[Path] = None,
) -> int:
    """Append one failing component finding to the historical list. Returns inserted row id."""
    path = db_path or DEFAULT_DB_PATH
    init_shadow_db(path)
    created_at = datetime.utcnow().isoformat() + "Z"
    with _conn(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO failing_component_events (
                created_at, node_id, message, power_draw_prev, power_draw_now,
                delta_t_prev, delta_t_now, severity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, node_id, message, power_draw_prev, power_draw_now, delta_t_prev, delta_t_now, severity_score),
        )
        row_id = cur.lastrowid
    return row_id or 0


def get_failing_component_events(
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    node_id: Optional[str] = None,
    limit: int = 500,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return historical failing component events (for Maintenance Task List / GET /health/anomalies)."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    with _conn(path) as conn:
        q = "SELECT id, created_at, node_id, message, power_draw_prev, power_draw_now, delta_t_prev, delta_t_now, severity_score FROM failing_component_events WHERE 1=1"
        params: List[Any] = []
        if start_ts:
            q += " AND created_at >= ?"
            params.append(start_ts.isoformat())
        if end_ts:
            q += " AND created_at <= ?"
            params.append(end_ts.isoformat())
        if node_id:
            q += " AND node_id = ?"
            params.append(node_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cur = conn.execute(q, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]
