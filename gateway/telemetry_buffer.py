"""
CooledAI Telemetry Buffer - Store-and-Forward Reliability

If Railway backend connection fails, store telemetry in SQLite.
On reconnect, burst stored data in chronological order.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager

from .collectors.base_collector import TelemetryObject

_logger = logging.getLogger("cooledai.telemetry_buffer")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


class TelemetryBuffer:
    """
    SQLite buffer for telemetry when backend is unreachable.
    Bursts in chronological order on reconnect.
    """

    def __init__(self, db_path: str = "telemetry_buffer.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(ts)")
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def push(self, telemetry: List[TelemetryObject]) -> None:
        """Store telemetry in buffer."""
        if not telemetry:
            return
        with self._conn() as conn:
            for obj in telemetry:
                ts = obj.timestamp.isoformat() if obj.timestamp else datetime.now().isoformat()
                payload = json.dumps(obj.to_dict())
                conn.execute(
                    "INSERT INTO telemetry (ts, payload) VALUES (?, ?)",
                    (ts, payload),
                )
            conn.commit()
        _logger.debug("TelemetryBuffer: pushed %d records", len(telemetry))

    def pop_all(self) -> List[dict]:
        """
        Pop all buffered records in chronological order.
        Returns list of dict payloads. Caller must delete after successful send.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT id, ts, payload FROM telemetry ORDER BY ts ASC"
            )
            rows = cur.fetchall()
        return [{"id": r[0], "ts": r[1], "payload": json.loads(r[2])} for r in rows]

    def delete_ids(self, ids: List[int]) -> None:
        """Delete records by id after successful burst."""
        if not ids:
            return
        with self._conn() as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM telemetry WHERE id IN ({placeholders})", ids)
            conn.commit()
        _logger.info("TelemetryBuffer: deleted %d records after burst", len(ids))

    def count(self) -> int:
        """Number of buffered records."""
        with self._conn() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM telemetry")
            return cur.fetchone()[0]
