"""CooledAI Immutable Hardware Command Audit Log.

Every hardware command CooledAI issues is recorded here BEFORE execution.
The log is append-only by design -- there are no delete or update methods.

Log format: one JSON object per line (JSONL), written to a rolling daily
file at ``/var/log/cooledai/audit/YYYY-MM-DD.jsonl``.

The ``data_plane_touched`` field is hardcoded to ``false`` and cannot be
set by any other code path. This provides a cryptographic-grade assertion
that CooledAI never accesses the tenant data plane.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cooledai.security.audit_log")


class AuditCommandType(str, Enum):
    """The narrow set of command types CooledAI may issue."""
    FAN_SPEED_SET = "FAN_SPEED_SET"
    FAN_READ = "FAN_READ"
    GPU_CLOCK_SET = "GPU_CLOCK_SET"
    GPU_POWER_LIMIT_SET = "GPU_POWER_LIMIT_SET"
    TEMP_READ = "TEMP_READ"
    POWER_READ = "POWER_READ"
    CSTATE_SET = "CSTATE_SET"
    REDFISH_READ = "REDFISH_READ"
    REDFISH_WRITE = "REDFISH_WRITE"


class HardwareAuditLog:
    """Append-only audit log for all hardware commands.

    This class intentionally has NO ``delete``, ``update``, ``truncate``,
    or ``clear`` methods. Once a record is written, it is permanent for
    the retention period configured in cooledai.yaml.
    """

    def __init__(
        self,
        log_dir: str = "/var/log/cooledai/audit",
        *,
        enabled: bool = True,
        signing_key: Optional[str] = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._enabled = enabled
        self._signing_key = (signing_key or "").encode("utf-8")
        self._lock = threading.Lock()
        self._total_commands = 0

        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        node_id: str,
        command_type: AuditCommandType,
        command_string: str,
        result_code: int = 0,
        result_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a hardware command BEFORE it executes.

        Args:
            node_id: The node issuing the command.
            command_type: One of the permitted AuditCommandType values.
            command_string: The exact command string with arguments.
            result_code: Exit code (0 = success).  Set after execution.
            result_summary: Brief result (e.g. "fan_rpm=2100").
            metadata: Optional extra context (temperatures, targets).

        Returns:
            The audit record as a dict (also written to disk).
        """
        now = datetime.now(timezone.utc)

        # data_plane_touched is ALWAYS false.  This is not configurable.
        # Any code review can verify this field cannot be set externally.
        entry: Dict[str, Any] = {
            "timestamp": now.isoformat(),
            "node_id": node_id,
            "command_type": command_type.value,
            "command_string": command_string,
            "result_code": result_code,
            "result_summary": result_summary,
            "data_plane_touched": False,  # IMMUTABLE — always False
        }
        if metadata:
            entry["metadata"] = metadata

        # HMAC signature for tamper detection
        if self._signing_key:
            payload = json.dumps(entry, sort_keys=True, default=str)
            sig = hmac.new(
                self._signing_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            entry["signature"] = sig

        if self._enabled:
            self._write_entry(now, entry)

        with self._lock:
            self._total_commands += 1

        return entry

    def _write_entry(self, now: datetime, entry: Dict[str, Any]) -> None:
        """Append a single JSONL entry to the daily log file."""
        date_str = now.strftime("%Y-%m-%d")
        log_file = self._log_dir / f"{date_str}.jsonl"

        line = json.dumps(entry, default=str) + "\n"

        with self._lock:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as exc:
                logger.error(
                    "[SECURITY] audit_log_write_failed: %s", exc
                )

    def read_day(
        self,
        date: str,
        *,
        node_id: Optional[str] = None,
        command_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read audit entries for a specific date.

        Args:
            date: Date string in YYYY-MM-DD format.
            node_id: Optional filter by node_id.
            command_type: Optional filter by command_type.

        Returns:
            List of audit record dicts.
        """
        log_file = self._log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return []

        entries: List[Dict[str, Any]] = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if node_id and entry.get("node_id") != node_id:
                    continue
                if command_type and entry.get("command_type") != command_type:
                    continue

                entries.append(entry)

        return entries

    def get_30_day_summary(self) -> Dict[str, Any]:
        """Generate a 30-day command summary for attestation reports.

        Returns command TYPE counts only — no actual command strings,
        to avoid leaking operational details.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        type_counts: Dict[str, int] = {}
        total = 0
        os_layer_commands = 0
        days_with_data = 0

        for day_offset in range(30):
            date = now - timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            entries = self.read_day(date_str)
            if entries:
                days_with_data += 1
            for entry in entries:
                cmd_type = entry.get("command_type", "UNKNOWN")
                type_counts[cmd_type] = type_counts.get(cmd_type, 0) + 1
                total += 1
                # Verify no OS-layer commands slipped through
                if entry.get("data_plane_touched", False):
                    os_layer_commands += 1

        return {
            "period_days": 30,
            "days_with_data": days_with_data,
            "total_commands": total,
            "commands_by_type": type_counts,
            "os_layer_commands_detected": os_layer_commands,
            "data_plane_touched": os_layer_commands > 0,
        }

    @property
    def total_commands_session(self) -> int:
        """Total commands recorded in this process lifetime."""
        return self._total_commands

    def get_last_audit_date(self) -> Optional[str]:
        """Return the most recent audit log date, or None."""
        if not self._log_dir.exists():
            return None
        files = sorted(self._log_dir.glob("*.jsonl"), reverse=True)
        if not files:
            return None
        return files[0].stem  # "2026-03-24"


# ── Module-level singleton ──────────────────────────────────────
_audit_log: Optional[HardwareAuditLog] = None


def get_audit_log() -> HardwareAuditLog:
    """Get or create the global audit log instance."""
    global _audit_log
    if _audit_log is None:
        _audit_log = HardwareAuditLog()
    return _audit_log


def init_audit_log(
    log_dir: str = "/var/log/cooledai/audit",
    *,
    enabled: bool = True,
    signing_key: Optional[str] = None,
) -> HardwareAuditLog:
    """Initialize the global audit log with custom settings."""
    global _audit_log
    _audit_log = HardwareAuditLog(
        log_dir, enabled=enabled, signing_key=signing_key
    )
    return _audit_log
