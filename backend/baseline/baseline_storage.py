"""
CooledAI Baseline Storage - 24 hours of 'Current State' snapshots (no adjustments).

Each snapshot records it_power_kw, cooling_power_kw, pue, and optional fields
for generating the baseline comparison report.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/baseline.db")
BASELINE_WINDOW_HOURS = 24


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


def init_baseline_db(db_path: Optional[Path] = None) -> None:
    """Create baseline_snapshots table if it does not exist."""
    path = db_path or DEFAULT_DB_PATH
    with _conn(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS baseline_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                it_power_kw REAL NOT NULL,
                cooling_power_kw REAL NOT NULL,
                pue REAL NOT NULL,
                total_facility_kw REAL,
                ups_losses_kw REAL,
                lighting_kw REAL,
                other_kw REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_baseline_created_at ON baseline_snapshots(created_at)")


def record_baseline_snapshot(
    it_power_kw: float,
    cooling_power_kw: float,
    pue: Optional[float] = None,
    total_facility_kw: Optional[float] = None,
    ups_losses_kw: float = 0.0,
    lighting_kw: float = 0.0,
    other_kw: float = 0.0,
    db_path: Optional[Path] = None,
) -> int:
    """
    Record one 'current state' snapshot (no adjustments). Call periodically over 24h.
    PUE = (it + cooling + ups + lighting + other) / it; computed if not provided.
    """
    path = db_path or DEFAULT_DB_PATH
    init_baseline_db(path)
    it_kw = max(0.0, float(it_power_kw))
    cooling_kw = max(0.0, float(cooling_power_kw))
    total = total_facility_kw
    if total is None:
        total = it_kw + cooling_kw + ups_losses_kw + lighting_kw + other_kw
    if pue is None:
        pue = total / it_kw if it_kw > 1e-6 else 0.0
    created_at = datetime.utcnow().isoformat() + "Z"
    with _conn(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO baseline_snapshots (
                created_at, it_power_kw, cooling_power_kw, pue,
                total_facility_kw, ups_losses_kw, lighting_kw, other_kw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, it_kw, cooling_kw, float(pue), total, ups_losses_kw, lighting_kw, other_kw),
        )
        row_id = cur.lastrowid or 0
    logger.debug("Baseline snapshot id=%s pue=%.3f", row_id, pue)
    return row_id


def get_baseline_snapshots(
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    max_hours: float = BASELINE_WINDOW_HOURS,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return baseline snapshots in the given window. Default: last 24 hours."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    if end_ts is None:
        end_ts = datetime.utcnow()
    if start_ts is None:
        start_ts = end_ts - timedelta(hours=max_hours)
    with _conn(path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM baseline_snapshots
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at ASC
            """,
            (start_ts.isoformat(), end_ts.isoformat()),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def clear_baseline_snapshots(db_path: Optional[Path] = None) -> int:
    """Delete all baseline snapshots. Returns count deleted."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return 0
    with _conn(path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM baseline_snapshots")
        n = cur.fetchone()[0]
        conn.execute("DELETE FROM baseline_snapshots")
    return n
