#!/usr/bin/env python3
"""
ST550 Pilot Node — Predictive Cooling Engine

Runs on the Pilot node (.100) alongside workload_sim. Monitors inference load
(via Ollama process activity) and reports load level to the CooledAI API.
Integrates with telemetry so the dashboard can show predictive vs reactive cooling.

Usage:
    python3 scripts/predictive_engine.py
    python3 scripts/predictive_engine.py --api-url https://...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    requests = None

API_URL = os.environ.get(
    "COOLEDAI_API_URL",
    "https://proactive-creativity-production.up.railway.app",
)
API_KEY = os.environ.get(
    "COOLEDAI_API_KEY",
    "sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo",
)
AGENT_ID = "ST550-CooledAI-Predictive"
POLL_INTERVAL = 5
LOAD_FILE = "/tmp/cooledai_inference_load"


def _inference_load_from_file() -> float:
    """Read load indicator from file (0.0–1.0). workload_sim writes this."""
    try:
        if os.path.exists(LOAD_FILE):
            with open(LOAD_FILE) as f:
                return float(f.read().strip() or 0)
    except (ValueError, OSError):
        pass
    return 0.0


OLLAMA_URL = "http://localhost:11434"


def _inference_load_from_ollama() -> float:
    """Estimate load by checking if Ollama is up (best-effort)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            return 0.5  # Ollama is up; assume moderate load when we can't detect
    except Exception:
        pass
    return 0.0


def _inference_load() -> float:
    """Return current inference load 0.0–1.0. Prefer file from workload_sim."""
    load = _inference_load_from_file()
    if load > 0:
        return min(1.0, load)
    return _inference_load_from_ollama()


def _post_load_heartbeat(load: float, api_url: str) -> None:
    """POST load level to CooledAI API (extends telemetry)."""
    if not requests:
        return
    url = f"{api_url.rstrip('/')}/api/v1/telemetry"
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    payload = {
        "agent_id": AGENT_ID,
        "heartbeat": {"inference_load": load, "predictive_active": load > 0.1},
    }
    try:
        requests.post(url, json=payload, headers=headers, timeout=5)
    except Exception as e:
        print(f"[predictive] API heartbeat failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive cooling engine for Pilot node")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL)
    args = parser.parse_args()

    if not requests:
        print("[predictive] Install requests: pip install requests")
        sys.exit(1)

    api_url = args.api_url
    print(f"[predictive] Running — reporting to {api_url} every {args.interval}s")
    while True:
        load = _inference_load()
        _post_load_heartbeat(load, api_url)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
