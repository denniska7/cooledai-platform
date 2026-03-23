"""
CooledAI Alert Notifier — webhook & email notifications.

All webhook POST calls run in a ``ThreadPoolExecutor`` with a 5-second
timeout.  Exceptions are logged as ``[ALERTS] webhook_failed`` and
**never propagated** to the caller.
"""

from __future__ import annotations

import json
import logging
import smtplib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Optional
from urllib.request import Request, urlopen

log = logging.getLogger("cooledai.alerts")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alert")

# Debounce: last fire time per (event_type, node_id) — min 5 min between fires
_DEBOUNCE_S = 300
_last_fire: dict[tuple, float] = {}


def _should_fire(event_type: str, node_id: str = "") -> bool:
    key = (event_type, node_id)
    now = time.monotonic()
    if now - _last_fire.get(key, 0) < _DEBOUNCE_S:
        return False
    _last_fire[key] = now
    return True


def _post_webhook(url: str, payload: dict) -> None:
    """POST JSON to webhook URL.  Called inside thread pool."""
    data = json.dumps(payload, default=str).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    urlopen(req, timeout=5)  # noqa: S310


def _send_email(to: str, subject: str, body: str) -> None:
    """Send a plain-text email via localhost SMTP (best-effort)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "cooledai-alerts@localhost"
    msg["To"] = to
    msg.set_content(body)
    try:
        with smtplib.SMTP("localhost", 25, timeout=5) as smtp:
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        log.warning("[ALERTS] email_failed: %s", exc)


def fire_alert(
    *,
    event: str,
    status: str,
    node_id: str = "",
    temp_c: Optional[float] = None,
    detail: str = "",
    webhook_url: str = "",
    email: str = "",
) -> None:
    """Fire an alert notification (webhook and/or email).

    Exceptions from webhooks are caught and logged — never propagated.
    """
    if not _should_fire(event, node_id):
        return  # Debounced

    payload = {
        "event": event,
        "status": status,
        "node_id": node_id,
        "temp_c": temp_c,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── Webhook ──
    if webhook_url:
        try:
            future = _executor.submit(_post_webhook, webhook_url, payload)
            future.result(timeout=5)
            log.info("[ALERTS] webhook_sent: event=%s node=%s", event, node_id)
        except FutureTimeout:
            log.warning("[ALERTS] webhook_failed: timeout (5s)")
        except Exception as exc:  # noqa: BLE001
            log.warning("[ALERTS] webhook_failed: %s", exc)

    # ── Email ──
    if email:
        subject = f"[CooledAI] {event}: {status}"
        body = json.dumps(payload, indent=2, default=str)
        try:
            _executor.submit(_send_email, email, subject, body)
        except Exception as exc:  # noqa: BLE001
            log.warning("[ALERTS] email_submit_failed: %s", exc)
