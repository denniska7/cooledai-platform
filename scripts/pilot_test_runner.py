#!/usr/bin/env python3
"""
CooledAI 5-Day Automated Pilot Test Runner

Alternates between 'Control Mode' (baseline — fixed fan RPM) and
'CooledAI Mode' (AI-optimised RPM via /optimize/thermal) every 24 hours
over a 5-day window:

    Day 1: Control
    Day 2: CooledAI
    Day 3: Control
    Day 4: CooledAI
    Day 5: Control

Each simulated "day" replays a realistic data-centre workload profile
(20-95 % CPU load), logs telemetry every `SAMPLE_INTERVAL_S` seconds,
and writes results to ``logs/pilot_results.csv``.

Usage:
    # Full 5-day run (compressed to minutes with --speedup)
    python scripts/pilot_test_runner.py --speedup 1440

    # Quick smoke test (one simulated day in ~10 s)
    python scripts/pilot_test_runner.py --days 1 --speedup 8640

    # Against a remote API server
    python scripts/pilot_test_runner.py --api-url http://10.0.0.5:8000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import core.* when run standalone
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

_log = logging.getLogger("pilot_test_runner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_S = 10          # seconds between telemetry samples (wall-clock)
SECONDS_PER_DAY = 86_400        # 24 h
DEFAULT_API_URL = "http://127.0.0.1:8000"
CSV_PATH = _project_root / "logs" / "pilot_results.csv"
CSV_FIELDNAMES = [
    "timestamp",
    "mode",
    "cpu_temp",
    "fan_rpm",
    "kwh_consumed",
    "target_rpm",
    "predicted_temp_delta",
    "power_efficiency_score",
    "utilization",
    "power_kw",
    "day",
    "elapsed_s",
]
CONTROL_FAN_RPM = 2400          # Fixed RPM baseline in Control Mode
MAX_FAN_RPM = 3000.0
RATED_FAN_POWER_KW = 12.0      # kW at 100 % RPM (Fan Affinity Law)

# Day schedule: True = CooledAI, False = Control
DAY_SCHEDULE = [False, True, False, True, False]  # 5 days


# ---------------------------------------------------------------------------
# Mock workload — "Day in the Life" of a data centre
# ---------------------------------------------------------------------------
def simulate_day_workload(t_frac: float) -> Dict[str, float]:
    """Return a synthetic telemetry snapshot for a fractional time-of-day.

    ``t_frac`` is in [0, 1) where 0 = midnight, 0.5 = noon.

    The profile models:
      * Overnight lull   (00:00-06:00)  ~20-35 % util
      * Morning ramp-up  (06:00-09:00)  35-70 %
      * Midday plateau   (09:00-13:00)  65-80 %
      * Afternoon spike  (13:00-15:00)  80-95 % (training jobs)
      * Gradual cool-off (15:00-21:00)  60-45 %
      * Evening taper    (21:00-00:00)  30-20 %

    Returns dict with ``utilization`` (0-1), ``cpu_temp`` (°C),
    ``power_kw``, ``fan_rpm`` (baseline).
    """
    hour = t_frac * 24.0

    if hour < 6:
        util = 0.20 + 0.15 * math.sin(math.pi * hour / 12)
    elif hour < 9:
        util = 0.35 + 0.35 * ((hour - 6) / 3)
    elif hour < 13:
        util = 0.65 + 0.15 * math.sin(math.pi * (hour - 9) / 8)
    elif hour < 15:
        util = 0.80 + 0.15 * math.sin(math.pi * (hour - 13) / 4)
    elif hour < 21:
        util = 0.60 - 0.15 * ((hour - 15) / 6)
    else:
        util = 0.30 - 0.10 * ((hour - 21) / 3)

    # Jitter to avoid perfectly smooth curves
    jitter = 0.03 * math.sin(hour * 7.3) + 0.02 * math.cos(hour * 13.1)
    util = max(0.05, min(1.0, util + jitter))

    ambient = 22.0 + 3.0 * math.sin(math.pi * (hour - 6) / 12)
    cpu_temp = ambient + util * 45.0 + 2.0 * math.sin(hour * 1.7)
    power_kw = 40.0 + util * 140.0 + 5.0 * math.sin(hour * 2.3)

    baseline_rpm = 1200 + util * 1800
    baseline_rpm = min(MAX_FAN_RPM, max(400, baseline_rpm))

    return {
        "utilization": round(util, 4),
        "cpu_temp": round(cpu_temp, 2),
        "power_kw": round(power_kw, 2),
        "fan_rpm": round(baseline_rpm, 1),
    }


def _fan_power_kw(rpm: float) -> float:
    """Fan Affinity Law: P = P_rated * (N / N_max)^3."""
    ratio = max(0.0, rpm / MAX_FAN_RPM)
    return RATED_FAN_POWER_KW * ratio ** 3


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call_optimize_thermal(
    api_url: str,
    telemetry: Dict[str, float],
    api_key: str = "",
) -> Optional[Dict[str, float]]:
    """POST to ``/optimize/thermal`` and return the prediction dict.

    Returns ``None`` on network / server error so the runner can fall
    back to control-mode behaviour.
    """
    import urllib.request
    import json as _json

    url = f"{api_url.rstrip('/')}/optimize/thermal"
    payload = _json.dumps(telemetry).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return _json.loads(resp.read().decode())
    except Exception as exc:
        _log.warning("API call failed (%s): %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class TelemetryLogger:
    """Append-only CSV writer for ``logs/pilot_results.csv``."""

    def __init__(self, path: Path = CSV_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def open(self) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDNAMES)
        if write_header:
            self._writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        if self._writer is None:
            self.open()
        assert self._writer is not None
        self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


# ---------------------------------------------------------------------------
# Main pilot loop
# ---------------------------------------------------------------------------

def run_pilot(
    days: int = 5,
    speedup: float = 1.0,
    api_url: str = DEFAULT_API_URL,
    api_key: str = "",
    offline: bool = False,
) -> Path:
    """Execute the pilot test.

    Args:
        days: Number of simulated days (max 5 for the default schedule).
        speedup: Wall-clock speedup factor.  ``1440`` compresses one day
            into one minute.
        api_url: Base URL of the FastAPI server.
        api_key: Optional ``X-API-Key`` value.
        offline: When True, skip API calls entirely and use the local
            ``ThermalModelLoader`` directly (no server required).

    Returns:
        Path to the CSV file with results.
    """
    schedule = (DAY_SCHEDULE * ((days // len(DAY_SCHEDULE)) + 1))[:days]
    logger = TelemetryLogger()
    logger.open()

    local_model = None
    if offline:
        try:
            from api.main import ThermalModelLoader
            local_model = ThermalModelLoader()
        except ImportError:
            _log.warning("Could not import ThermalModelLoader — running Control-only offline.")

    total_sim_seconds = days * SECONDS_PER_DAY
    wall_step = SAMPLE_INTERVAL_S / speedup  # actual sleep per sample
    cumulative_kwh = 0.0
    sample = 0

    _log.info(
        "Starting %d-day pilot (speedup=%.0fx, step=%.3fs wall, API=%s)",
        days, speedup, wall_step, api_url if not offline else "offline",
    )

    for day_idx, is_cooledai in enumerate(schedule):
        mode_label = "CooledAI" if is_cooledai else "Control"
        _log.info("=== Day %d/%d — %s Mode ===", day_idx + 1, days, mode_label)

        day_start_sim = day_idx * SECONDS_PER_DAY
        steps_per_day = SECONDS_PER_DAY // SAMPLE_INTERVAL_S

        for step in range(steps_per_day):
            sim_time = day_start_sim + step * SAMPLE_INTERVAL_S
            t_frac = (step * SAMPLE_INTERVAL_S) / SECONDS_PER_DAY

            workload = simulate_day_workload(t_frac)

            target_rpm = CONTROL_FAN_RPM
            predicted_temp_delta = 0.0
            power_efficiency_score = 0.0

            if is_cooledai:
                telemetry_payload = {
                    "cpu_temp": workload["cpu_temp"],
                    "fan_rpm": workload["fan_rpm"],
                    "power_kw": workload["power_kw"],
                    "utilization": workload["utilization"],
                }

                prediction = None
                if offline and local_model is not None:
                    prediction = local_model.predict(telemetry_payload)
                elif not offline:
                    prediction = _call_optimize_thermal(api_url, telemetry_payload, api_key)

                if prediction:
                    target_rpm = prediction.get("target_rpm", CONTROL_FAN_RPM)
                    predicted_temp_delta = prediction.get("predicted_temp_delta", 0.0)
                    power_efficiency_score = prediction.get("power_efficiency_score", 0.0)
            else:
                target_rpm = CONTROL_FAN_RPM
                power_efficiency_score = 100.0 * (
                    0.6 if (CONTROL_FAN_RPM / MAX_FAN_RPM) > 0.80 else 0.8
                )

            fan_power = _fan_power_kw(target_rpm)
            interval_kwh = fan_power * (SAMPLE_INTERVAL_S / 3600.0)
            cumulative_kwh += interval_kwh

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": mode_label,
                "cpu_temp": workload["cpu_temp"],
                "fan_rpm": target_rpm,
                "kwh_consumed": round(cumulative_kwh, 4),
                "target_rpm": target_rpm,
                "predicted_temp_delta": predicted_temp_delta,
                "power_efficiency_score": power_efficiency_score,
                "utilization": workload["utilization"],
                "power_kw": workload["power_kw"],
                "day": day_idx + 1,
                "elapsed_s": sim_time,
            }
            logger.log(row)
            sample += 1

            if sample % 500 == 0 or (step == steps_per_day - 1):
                _log.info(
                    "  Day %d | step %d/%d | util=%.0f%% | cpu=%.1f°C | "
                    "rpm=%d | kWh=%.2f | mode=%s",
                    day_idx + 1, step + 1, steps_per_day,
                    workload["utilization"] * 100, workload["cpu_temp"],
                    int(target_rpm), cumulative_kwh, mode_label,
                )

            if wall_step > 0.001:
                time.sleep(wall_step)

    logger.close()
    _log.info(
        "Pilot complete — %d samples, %.2f cumulative kWh → %s",
        sample, cumulative_kwh, logger.path,
    )
    return logger.path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI 5-Day Automated Pilot Test Runner",
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of simulated days (default: 5).",
    )
    parser.add_argument(
        "--speedup", type=float, default=1.0,
        help="Wall-clock speedup factor (e.g. 1440 = 1 day per minute).",
    )
    parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL,
        help="Base URL of the FastAPI server.",
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("COOLEDAI_API_KEY", ""),
        help="X-API-Key header value (default: $COOLEDAI_API_KEY).",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip API calls; use local ThermalModelLoader directly.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Override CSV output path (default: logs/pilot_results.csv).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    if args.csv:
        global CSV_PATH
        CSV_PATH = Path(args.csv)

    csv_out = run_pilot(
        days=args.days,
        speedup=args.speedup,
        api_url=args.api_url,
        api_key=args.api_key,
        offline=args.offline,
    )
    print(f"\nResults written to: {csv_out}")


if __name__ == "__main__":
    main()
