#!/usr/bin/env python3
"""
Lenovo ST550 "lenovo_live" telemetry logger

Behavior:
- Sample GPU temperatures every 5 seconds (via pynvml).
- Append samples to a persistent local log file `lenovo_live.log` (JSON Lines).
- Buffer samples and upload them once per hour to:
    POST {API_URL}/api/v1/telemetry
  using the strict telemetry schema used by st550_telemetry.py.

This script is designed to keep running (use nohup) even after you close
your laptop.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
import psutil


# ----------------------------
# Config (env overrides)
# ----------------------------
USER_ID = os.environ.get("USER_ID", "dennis@cooledai.com")  # used only for local log lines
API_URL = os.environ.get(
    "API_URL",
    "https://proactive-creativity-production.up.railway.app",
).rstrip("/")
API_KEY = os.environ.get("API_KEY", "sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo")
NODE_ID = os.environ.get("NODE_ID", "st550-primary")

POLL_INTERVAL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "5"))
UPLOAD_INTERVAL_SEC = float(os.environ.get("UPLOAD_INTERVAL_SEC", "3600"))

REQUEST_TIMEOUT_SEC = float(os.environ.get("REQUEST_TIMEOUT_SEC", "30"))
MAX_UPLOAD_RETRIES = int(os.environ.get("MAX_UPLOAD_RETRIES", "5"))
BACKOFF_BASE_SEC = float(os.environ.get("BACKOFF_BASE_SEC", "2"))

LOG_FILE = os.environ.get("LOG_FILE", "")  # if empty, auto-detect


def _resolve_log_file() -> Path:
    """
    Choose a writable log location.
    - Prefer /var/log/lenovo_live.log
    - Fall back to /tmp/lenovo_live.log
    """
    if LOG_FILE:
        return Path(LOG_FILE).expanduser().resolve()

    var_log = Path("/var/log/lenovo_live.log")
    try:
        var_log.parent.mkdir(parents=True, exist_ok=True)
        # attempt to open for append
        with var_log.open("a", encoding="utf-8"):
            pass
        return var_log
    except Exception:
        return Path("/tmp/lenovo_live.log")


def _cpu_temp_fallback_c() -> float:
    """
    Try psutil sensors; if not available return 50.0.
    (The backend uses `temperature_c` field; we primarily log GPU temps.)
    """
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        if not temps:
            return 50.0
        # pick max current
        best = 50.0
        for _name, entries in temps.items():
            for e in entries:
                current = getattr(e, "current", None)
                if current is not None:
                    best = max(best, float(current))
        return float(best)
    except Exception:
        return 50.0


def _read_gpu_temps(nvidia_pynvml: Any) -> list[dict]:
    """
    Return one telemetry record per GPU using pynvml.
    Uses node_id convention: {NODE_ID}/gpu{i}
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    count = nvidia_pynvml.nvmlDeviceGetCount()
    for i in range(count):
        try:
            handle = nvidia_pynvml.nvmlDeviceGetHandleByIndex(i)
            temp_c = nvidia_pynvml.nvmlDeviceGetTemperature(
                handle, nvidia_pynvml.NVML_TEMPERATURE_GPU
            )
            records.append(
                {
                    "node_id": f"{NODE_ID}/gpu{i}",
                    "timestamp": now_iso,
                    "temperature_c": float(temp_c),
                }
            )
        except Exception:
            # Keep schema stable; skip bad sensors but don't crash.
            continue
    return records


def _upload_batch(session: requests.Session, telemetry_batch: list[dict]) -> bool:
    """
    Upload buffered telemetry to the backend.
    Returns True on success.
    """
    url = f"{API_URL}/api/v1/telemetry"
    payload = {
        "agent_id": NODE_ID,
        "telemetry": telemetry_batch,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    last_status: Optional[int] = None
    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            last_status = resp.status_code
            if resp.status_code == 200:
                return True

            # Retry on non-200 transient failures
            # (application errors should still retry a few times)
        except requests.exceptions.RequestException:
            last_status = None

        # backoff
        if attempt < MAX_UPLOAD_RETRIES:
            backoff = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            jitter = random.uniform(0, 1)
            time.sleep(min(60.0, backoff + jitter))

    # exhausted
    print(f"[lenovo_live] upload failed after {MAX_UPLOAD_RETRIES} retries status={last_status}")
    return False


def main() -> None:
    log_path = _resolve_log_file()
    print(f"[lenovo_live] logging to {log_path} node_id={NODE_ID}")
    print(f"[lenovo_live] polling={POLL_INTERVAL_SEC}s upload_interval={UPLOAD_INTERVAL_SEC}s")

    # Load NVML
    try:
        import pynvml
    except ImportError as exc:
        raise SystemExit("pynvml is required. Install with: pip install --break-system-packages nvidia-ml-py3") from exc

    pynvml.nvmlInit()

    telemetry_batch: list[dict] = []
    next_upload_at = time.time() + UPLOAD_INTERVAL_SEC

    session = requests.Session()

    while True:
        now_iso = datetime.now(timezone.utc).isoformat()

        # Read GPU temps (required for backend max_temp_c)
        records = _read_gpu_temps(pynvml)
        cpu_temp_c = _cpu_temp_fallback_c()

        sample_obj = {
            "ts": now_iso,
            "user_id": USER_ID,
            "node_id": NODE_ID,
            "cpu_temp_c": cpu_temp_c,
            "records": records,
        }

        # Append to log file (JSONL)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample_obj) + "\n")
        except Exception:
            # don't crash on logging issues
            pass

        if records:
            telemetry_batch.extend(records)

        # Upload hourly
        now = time.time()
        if now >= next_upload_at:
            if telemetry_batch:
                print(f"[lenovo_live] uploading {len(telemetry_batch)} records...")
                ok = _upload_batch(session, telemetry_batch)
                if ok:
                    print(f"[lenovo_live] upload success (records={len(telemetry_batch)})")
                    telemetry_batch = []
                else:
                    # Keep buffer; try again at the next interval.
                    # (We could also retry immediately, but you asked for hourly uploads.)
                    print("[lenovo_live] keeping buffer (upload failed)")

            next_upload_at = now + UPLOAD_INTERVAL_SEC

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()

