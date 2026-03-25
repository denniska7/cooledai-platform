"""
CooledAI Append-Only Audit Log.

Every API request, XCC fan command, and security rejection is recorded as a
single JSON-lines entry to ``/var/log/cooledai/audit.log`` (configurable).

The file is opened with ``O_APPEND`` so concurrent writers cannot corrupt
existing entries.  Rotation: daily, 90-day retention (see SECURITY.md).
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("cooledai.audit")

_DEFAULT_PATH = os.environ.get(
    "COOLEDAI_AUDIT_LOG", "/var/log/cooledai/audit.log"
)

# Module-level file handle — opened once, append-only.
_fh: Optional[Any] = None
_audit_path: Optional[str] = None

# HMAC-SHA256 signing key (from env; empty = signing disabled).
_signing_key: bytes = os.environ.get("COOLEDAI_AUDIT_SIGNING_KEY", "").encode("utf-8")


def _ensure_open(path: Optional[str] = None) -> Any:
    """Open the audit log (create parent dirs if needed)."""
    global _fh, _audit_path
    target = path or _audit_path or _DEFAULT_PATH
    if _fh is not None and _audit_path == target:
        return _fh
    try:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        _fh = open(target, "a", encoding="utf-8")  # noqa: SIM115
        _audit_path = target
    except OSError as exc:
        log.warning("[AUDIT] Cannot open audit log %s: %s", target, exc)
        _fh = None
    return _fh


def configure(path: str) -> None:
    """Set the audit log path (called during startup from config)."""
    global _fh, _audit_path
    if _fh is not None:
        try:
            _fh.close()
        except OSError:
            pass
    _fh = None
    _audit_path = path
    _ensure_open(path)


def _write(entry: Dict[str, Any]) -> None:
    """Append a single JSON-line to the audit log."""
    fh = _ensure_open()
    if fh is None:
        return  # Graceful degradation — don't crash the request
    entry["data_plane_touched"] = False
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    # HMAC-SHA256 sign the entry if signing key is configured
    if _signing_key:
        payload = json.dumps(entry, sort_keys=True, default=str)
        sig = hmac_mod.new(
            _signing_key, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        entry["signature"] = sig
    try:
        line = json.dumps(entry, default=str, separators=(",", ":"))
        fh.write(line + "\n")
        fh.flush()
    except OSError as exc:
        log.warning("[AUDIT] Write failed: %s", exc)


def configure_signing_key(key: str) -> None:
    """Set the signing key (called during startup from config)."""
    global _signing_key
    _signing_key = key.encode("utf-8")


def verify_audit_entry(entry: dict, signing_key: str) -> bool:
    """Verify an audit log entry's HMAC-SHA256 signature.

    Returns False for unsigned entries (missing signature field).
    """
    sig = entry.get("signature")
    if not sig:
        return False
    check = dict(entry)
    del check["signature"]
    payload = json.dumps(check, sort_keys=True, default=str)
    expected = hmac_mod.new(
        signing_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return hmac_mod.compare_digest(sig, expected)


# ── Public API ──────────────────────────────────────────────────

def log_api_request(
    *,
    key_id: str,
    client_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    ip: str,
) -> None:
    """Record an API request."""
    _write({
        "type": "api_request",
        "key_id": key_id[:12] + "..." if len(key_id) > 12 else key_id,
        "client_id": client_id,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "ip": ip,
    })


def log_fan_command(
    *,
    node_id: str,
    target_rpm: int,
    method: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Record an XCC fan command."""
    _write({
        "type": "fan_command",
        "node_id": node_id,
        "target_rpm": target_rpm,
        "method": method,
        "success": success,
        "error": error,
    })


def log_security_rejection(
    *,
    reason: str,
    key_id: str = "",
    ip: str = "",
    endpoint: str = "",
) -> None:
    """Record a security rejection (bad key, rate limit, violation, etc.)."""
    _write({
        "type": "security_rejection",
        "reason": reason,
        "key_id": key_id[:12] + "..." if len(key_id) > 12 else key_id,
        "ip": ip,
        "endpoint": endpoint,
    })
