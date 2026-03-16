#!/usr/bin/env python3
"""
CooledAI Bridge-to-Cloud — ST550 Edge Telemetry Agent

Reads GPU telemetry from nvidia-smi every 5 seconds and POSTs it to the
CooledAI portal so the Live System Pulse dashboard shows real hardware
data.  Sends a lightweight heartbeat every 60 seconds so the portal can
distinguish "bridge idle" from "bridge down".

Network resilience: on POST failure the agent retries with exponential
backoff (2 s base, 60 s cap, jitter) so brief Wi-Fi drops don't cause
the dashboard to flatline.

No external dependencies — stdlib only (runs on a bare Ubuntu 22.04 box).

Usage
-----
  # Fill in TOKEN and NODE_ID, then:
  python3 scripts/bridge_to_cloud.py

  # Dry-run (fake GPU data, no nvidia-smi required):
  python3 scripts/bridge_to_cloud.py --dry-run

  # Override portal URL:
  python3 scripts/bridge_to_cloud.py --portal-url https://api.cooledai.com
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ===================================================================
# CONFIGURATION — fill these in once the dev team provides them
# ===================================================================
TOKEN: str = os.environ.get("COOLEDAI_API_KEY", "")
NODE_ID: str = os.environ.get("COOLEDAI_NODE_ID", "st550-lenovo-01")
PORTAL_URL: str = os.environ.get(
    "COOLEDAI_PORTAL_URL", "https://api.cooledai.com"
)
POLL_INTERVAL_S: float = 5.0
HEARTBEAT_INTERVAL_S: float = 60.0
# ===================================================================

BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
WARN_AFTER_FAILURES = 3

_log = logging.getLogger("bridge_to_cloud")
_running = True


# ------------------------------------------------------------------
# nvidia-smi reader
# ------------------------------------------------------------------

def read_gpu_telemetry(dry_run: bool = False) -> List[Dict[str, Any]]:
    """Query nvidia-smi for temp, fan speed, utilisation, and power draw
    for every GPU.  Returns one dict per GPU."""
    if dry_run:
        t = time.time()
        return [
            {
                "gpu_index": 0,
                "temperature_c": round(54 + 12 * math.sin(t * 0.01), 1),
                "fan_speed_pct": round(40 + 15 * math.sin(t * 0.008), 1),
                "utilization_pct": round(55 + 30 * math.sin(t * 0.005), 1),
                "power_draw_w": round(45 + 20 * math.sin(t * 0.007), 1),
            },
            {
                "gpu_index": 1,
                "temperature_c": round(56 + 11 * math.sin(t * 0.01 + 1), 1),
                "fan_speed_pct": round(42 + 14 * math.sin(t * 0.008 + 1), 1),
                "utilization_pct": round(50 + 28 * math.sin(t * 0.005 + 1), 1),
                "power_draw_w": round(44 + 18 * math.sin(t * 0.007 + 1), 1),
            },
        ]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,fan.speed,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except FileNotFoundError:
        _log.error("nvidia-smi not found — is the NVIDIA driver installed?")
        return []
    except Exception as exc:
        _log.warning("nvidia-smi failed: %s", exc)
        return []

    gpus: List[Dict[str, Any]] = []
    for idx, line in enumerate(out.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "gpu_index": idx,
                "temperature_c": float(parts[0]),
                "fan_speed_pct": float(parts[1]),
                "utilization_pct": float(parts[2]),
                "power_draw_w": float(parts[3]),
            })
        except ValueError:
            _log.warning("Could not parse nvidia-smi line: %s", line)
    return gpus


# ------------------------------------------------------------------
# HTTP POST with exponential backoff
# ------------------------------------------------------------------

def _post_json(
    url: str,
    payload: Dict[str, Any],
    token: str,
    timeout_s: float = 10.0,
) -> bool:
    """POST JSON to *url* with X-API-Key.  Returns True on 2xx."""
    data = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": token,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                return True
            _log.warning("Server returned %d", resp.status)
            return False
    except urllib.error.HTTPError as exc:
        _log.warning("HTTP %d from %s: %s", exc.code, url, exc.reason)
        return False
    except Exception as exc:
        _log.debug("POST failed: %s", exc)
        return False


def post_with_retry(
    url: str,
    payload: Dict[str, Any],
    token: str,
) -> bool:
    """Try to POST; on failure retry with exponential backoff + jitter
    until success or the process is shutting down."""
    consecutive_failures = 0
    backoff = BACKOFF_BASE_S

    while _running:
        if _post_json(url, payload, token):
            if consecutive_failures > 0:
                _log.info(
                    "Connection recovered after %d retries.", consecutive_failures,
                )
            return True

        consecutive_failures += 1
        if consecutive_failures == WARN_AFTER_FAILURES:
            _log.warning(
                "Portal unreachable for %d attempts — retrying with backoff.",
                consecutive_failures,
            )

        jitter = random.uniform(0, backoff * 0.3)
        sleep_s = min(backoff + jitter, BACKOFF_CAP_S)
        _log.debug("Retry in %.1fs (attempt %d)", sleep_s, consecutive_failures)
        time.sleep(sleep_s)
        backoff = min(backoff * 2, BACKOFF_CAP_S)

    return False


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run_bridge(
    portal_url: str = PORTAL_URL,
    token: str = TOKEN,
    node_id: str = NODE_ID,
    dry_run: bool = False,
) -> None:
    global _running

    telemetry_url = f"{portal_url.rstrip('/')}/api/v1/telemetry"

    _log.info(
        "Bridge started — node=%s  portal=%s  poll=%ss  dry_run=%s",
        node_id, telemetry_url, POLL_INTERVAL_S, dry_run,
    )
    if not token:
        _log.warning(
            "TOKEN is empty. Set COOLEDAI_API_KEY or edit TOKEN at the top "
            "of this script. Requests will likely be rejected (401)."
        )

    last_heartbeat = 0.0

    while _running:
        loop_start = time.monotonic()

        # --- Read GPU data ---
        gpus = read_gpu_telemetry(dry_run)
        if not gpus:
            _log.warning("No GPU data — skipping this cycle.")
            time.sleep(POLL_INTERVAL_S)
            continue

        now_iso = datetime.now(timezone.utc).isoformat()
        records = []
        for g in gpus:
            records.append({
                "node_id": f"{node_id}/gpu{g['gpu_index']}",
                "timestamp": now_iso,
                "temperature_c": g["temperature_c"],
                "fan_speed_pct": g["fan_speed_pct"],
                "utilization_pct": g["utilization_pct"],
                "power_draw_w": g["power_draw_w"],
            })

        # --- POST telemetry ---
        payload = {"agent_id": node_id, "telemetry": records}
        post_with_retry(telemetry_url, payload, token)

        # --- Heartbeat (every 60 s) ---
        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            hb_payload = {
                "heartbeat": {
                    "agent_id": node_id,
                    "timestamp": now_iso,
                    "gpu_count": len(gpus),
                    "uptime_s": round(now_mono, 1),
                },
            }
            _post_json(telemetry_url, hb_payload, token)
            last_heartbeat = now_mono

        # --- Pace the loop ---
        elapsed = time.monotonic() - loop_start
        sleep_remaining = max(0, POLL_INTERVAL_S - elapsed)
        if sleep_remaining > 0 and _running:
            time.sleep(sleep_remaining)


# ------------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------------

def _shutdown(signum: int, _frame: Any) -> None:
    global _running
    sig_name = signal.Signals(signum).name
    _log.info("Received %s — shutting down.", sig_name)
    _running = False


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI Bridge-to-Cloud — ST550 Edge Telemetry Agent",
    )
    parser.add_argument(
        "--portal-url", type=str,
        default=os.environ.get("COOLEDAI_PORTAL_URL", PORTAL_URL),
        help="CooledAI portal base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--token", type=str,
        default=os.environ.get("COOLEDAI_API_KEY", TOKEN),
        help="API token / X-API-Key (default: $COOLEDAI_API_KEY).",
    )
    parser.add_argument(
        "--node-id", type=str,
        default=os.environ.get("COOLEDAI_NODE_ID", NODE_ID),
        help="Unique node identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate fake GPU data (no nvidia-smi required).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    run_bridge(
        portal_url=args.portal_url,
        token=args.token,
        node_id=args.node_id,
        dry_run=args.dry_run,
    )
    _log.info("Bridge stopped.")


if __name__ == "__main__":
    main()
