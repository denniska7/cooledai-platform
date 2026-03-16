#!/usr/bin/env python3
"""
CooledAI 7-Day Hardware Pilot — Lenovo ST550 + 2× Quadro P2000

A/B test comparing standard BIOS-Auto cooling vs. CooledAI-optimised fan
control on real hardware.  Collects GPU temps (nvidia-smi), CPU temps
(lm-sensors), and fan speeds (ipmitool) every 5 seconds.

Schedule
--------
  Days 1-3  BASELINE   Fans on BIOS Auto — observe only.
  Day  4    SHADOW     CooledAI predicts + logs what it *would* set.
                       Fans stay on BIOS Auto.
  Days 5-7  ACTIVE     CooledAI takes manual control via IPMI and
                       optimises for P2000 thermal curves.

Safety
------
  * If *any* GPU exceeds 85 °C the watchdog immediately reverts to BIOS
    Auto (``ipmitool raw 0x32 0x9b 0x00``), logs the event, and exits.
  * On SIGINT / SIGTERM / unhandled exception the script also reverts.

Outputs
-------
  logs/seven_day_pilot.csv    — every-5-second telemetry rows
  logs/pilot_report.txt       — daily summary appended after each 24 h block

Usage
-----
  # Production (runs for 7 real days)
  sudo python3 scripts/seven_day_pilot.py

  # Dry-run (no IPMI writes, uses fake sensor data)
  python3 scripts/seven_day_pilot.py --dry-run

  # Shorter test
  sudo python3 scripts/seven_day_pilot.py --days 2

  # Custom API server
  sudo python3 scripts/seven_day_pilot.py --api-url http://10.0.0.5:8000
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

_log = logging.getLogger("seven_day_pilot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_S = 5
SECONDS_PER_DAY = 86_400
GPU_WATCHDOG_LIMIT_C = 85.0
DEFAULT_API_URL = "http://127.0.0.1:8000"
CSV_PATH = _PROJECT_ROOT / "logs" / "seven_day_pilot.csv"
REPORT_PATH = _PROJECT_ROOT / "logs" / "pilot_report.txt"
CSV_FIELDS = [
    "timestamp", "day", "phase", "elapsed_s",
    "gpu0_temp_c", "gpu1_temp_c", "gpu_max_c",
    "cpu_temp_c",
    "fan_speeds_rpm",
    "ai_target_duty_pct", "ai_predicted_temp_60s",
    "ai_power_efficiency", "applied",
]

# Lenovo ST550 IPMI raw commands
IPMI_MANUAL_ON = ["ipmitool", "raw", "0x32", "0x9b", "0x01"]
IPMI_MANUAL_OFF = ["ipmitool", "raw", "0x32", "0x9b", "0x00"]

# Fan duty cycle for IPMI manual mode (zone 0, duty 0x00-0x64)
def _ipmi_set_duty_cmd(duty_pct: int) -> List[str]:
    duty_pct = max(0, min(100, duty_pct))
    return ["ipmitool", "raw", "0x32", "0x69", "0x00", hex(duty_pct)]


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------
class Phase:
    BASELINE = "BASELINE"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"


def _day_phase(day: int) -> str:
    if day <= 3:
        return Phase.BASELINE
    if day == 4:
        return Phase.SHADOW
    return Phase.ACTIVE


# ---------------------------------------------------------------------------
# P2000 thermal curve → target fan duty %
# ---------------------------------------------------------------------------
# The Quadro P2000 throttles at ~97 °C but we aim to keep GPUs well under
# the 85 °C watchdog limit.  The curve is intentionally conservative during
# ACTIVE mode so that CooledAI earns trust in the first deployment.

def p2000_target_duty(gpu_max_c: float) -> int:
    """Map peak GPU temp to optimal fan duty-cycle percentage."""
    if gpu_max_c < 40:
        return 25
    if gpu_max_c < 50:
        return 35
    if gpu_max_c < 60:
        return 45
    if gpu_max_c < 70:
        return 60
    if gpu_max_c < 78:
        return 75
    if gpu_max_c < 83:
        return 90
    return 100


# ---------------------------------------------------------------------------
# 60-second temperature predictor (simple linear extrapolation)
# ---------------------------------------------------------------------------
class TempPredictor60s:
    """Predict temperature 60 seconds into the future using a sliding window
    of recent readings and linear-regression slope."""

    def __init__(self, window: int = 24):
        self._history: deque[Tuple[float, float]] = deque(maxlen=window)

    def add(self, ts: float, temp_c: float) -> None:
        self._history.append((ts, temp_c))

    def predict(self, horizon_s: float = 60.0) -> Optional[float]:
        if len(self._history) < 4:
            return None
        times = [t for t, _ in self._history]
        temps = [v for _, v in self._history]
        n = len(times)
        t0 = times[0]
        xs = [t - t0 for t in times]
        x_mean = sum(xs) / n
        y_mean = sum(temps) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, temps))
        den = sum((x - x_mean) ** 2 for x in xs)
        if abs(den) < 1e-9:
            return temps[-1]
        slope = num / den
        last_x = xs[-1]
        return temps[-1] + slope * horizon_s


# ---------------------------------------------------------------------------
# Hardware telemetry readers
# ---------------------------------------------------------------------------

def read_gpu_temps(dry_run: bool = False) -> List[float]:
    """Return list of GPU temps (°C) via nvidia-smi."""
    if dry_run:
        t = time.time()
        return [
            round(52.0 + 15.0 * math.sin(t * 0.01) + 3.0 * math.sin(t * 0.07), 1),
            round(54.0 + 14.0 * math.sin(t * 0.01 + 0.5) + 2.5 * math.cos(t * 0.05), 1),
        ]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return [float(line.strip()) for line in out.splitlines() if line.strip()]
    except Exception as exc:
        _log.warning("nvidia-smi failed: %s", exc)
        return []


def read_cpu_temp(dry_run: bool = False) -> Optional[float]:
    """Return highest CPU package temp (°C) from lm-sensors."""
    if dry_run:
        t = time.time()
        return round(42.0 + 10.0 * math.sin(t * 0.008) + 2.0 * math.cos(t * 0.03), 1)
    try:
        out = subprocess.check_output(
            ["sensors"], timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        temps: List[float] = []
        for line in out.splitlines():
            # Match lines like "Core 0:  +45.0°C" or "Package id 0: +52.0°C"
            m = re.search(r"[+]?([\d.]+)\s*°?C", line)
            if m and any(k in line.lower() for k in ("core", "package", "tctl", "tdie")):
                temps.append(float(m.group(1)))
        return max(temps) if temps else None
    except Exception as exc:
        _log.warning("sensors failed: %s", exc)
        return None


def read_fan_speeds(dry_run: bool = False) -> Dict[str, int]:
    """Return dict of fan_name → RPM from ipmitool SDR."""
    if dry_run:
        t = time.time()
        base = 3200 + int(800 * math.sin(t * 0.005))
        return {"Fan1A": base, "Fan1B": base + 50, "Fan2A": base - 30, "Fan2B": base + 20}
    try:
        out = subprocess.check_output(
            ["ipmitool", "sdr"], timeout=5, stderr=subprocess.DEVNULL,
        ).decode()
        fans: Dict[str, int] = {}
        for line in out.splitlines():
            if "fan" not in line.lower():
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            m = re.search(r"(\d+)\s*RPM", parts[1], re.IGNORECASE)
            if m:
                fans[name] = int(m.group(1))
        return fans
    except Exception as exc:
        _log.warning("ipmitool sdr failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# IPMI fan control
# ---------------------------------------------------------------------------

def ipmi_enable_manual(dry_run: bool = False) -> bool:
    if dry_run:
        _log.info("[DRY-RUN] Would run: %s", " ".join(IPMI_MANUAL_ON))
        return True
    try:
        subprocess.check_call(IPMI_MANUAL_ON, timeout=5,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log.info("IPMI: Manual fan control ENABLED.")
        return True
    except Exception as exc:
        _log.error("IPMI manual-on failed: %s", exc)
        return False


def ipmi_revert_auto(dry_run: bool = False) -> bool:
    if dry_run:
        _log.info("[DRY-RUN] Would run: %s", " ".join(IPMI_MANUAL_OFF))
        return True
    try:
        subprocess.check_call(IPMI_MANUAL_OFF, timeout=5,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log.info("IPMI: Reverted to BIOS Auto.")
        return True
    except Exception as exc:
        _log.error("IPMI revert-auto failed: %s", exc)
        return False


def ipmi_set_fan_duty(duty_pct: int, dry_run: bool = False) -> bool:
    cmd = _ipmi_set_duty_cmd(duty_pct)
    if dry_run:
        _log.debug("[DRY-RUN] Would run: %s", " ".join(cmd))
        return True
    try:
        subprocess.check_call(cmd, timeout=5,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        _log.warning("IPMI set-duty %d%% failed: %s", duty_pct, exc)
        return False


# ---------------------------------------------------------------------------
# CooledAI API helper
# ---------------------------------------------------------------------------

def call_optimize_thermal(
    api_url: str,
    cpu_temp: float,
    gpu_max: float,
    fan_rpm: int,
    api_key: str = "",
) -> Optional[Dict[str, Any]]:
    """POST telemetry to /optimize/thermal."""
    url = f"{api_url.rstrip('/')}/optimize/thermal"
    payload = json.dumps({
        "cpu_temp": max(cpu_temp, gpu_max),
        "fan_rpm": fan_rpm,
        "power_kw": 0.15,   # 2× P2000 @ 75 W each
        "utilization": min(1.0, gpu_max / 90.0),
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        _log.debug("API call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CSV + report logger
# ---------------------------------------------------------------------------

class PilotLogger:
    def __init__(self, csv_path: Path = CSV_PATH, report_path: Path = REPORT_PATH):
        self.csv_path = csv_path
        self.report_path = report_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def open(self) -> None:
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if write_header:
            self._writer.writeheader()

    def log_row(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self.open()
        assert self._writer is not None
        self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def append_report(self, text: str) -> None:
        with open(self.report_path, "a") as f:
            f.write(text)


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

@dataclass
class DayAccumulator:
    day: int = 0
    phase: str = ""
    samples: int = 0
    gpu_max_readings: List[float] = field(default_factory=list)
    cpu_readings: List[float] = field(default_factory=list)
    fan_rpm_readings: List[int] = field(default_factory=list)
    duty_pct_readings: List[int] = field(default_factory=list)
    predicted_temps: List[float] = field(default_factory=list)
    watchdog_trips: int = 0
    start_ts: str = ""
    end_ts: str = ""

    def add(
        self,
        gpu_max: float,
        cpu_temp: Optional[float],
        fan_rpm_avg: int,
        duty_pct: int,
        predicted_60s: Optional[float],
    ) -> None:
        self.samples += 1
        self.gpu_max_readings.append(gpu_max)
        if cpu_temp is not None:
            self.cpu_readings.append(cpu_temp)
        if fan_rpm_avg > 0:
            self.fan_rpm_readings.append(fan_rpm_avg)
        if duty_pct >= 0:
            self.duty_pct_readings.append(duty_pct)
        if predicted_60s is not None:
            self.predicted_temps.append(predicted_60s)

    def summary(self) -> str:
        def _stats(vals: list) -> str:
            if not vals:
                return "n/a"
            mn, mx, avg = min(vals), max(vals), sum(vals) / len(vals)
            return f"min={mn:.1f}  avg={avg:.1f}  max={mx:.1f}"

        lines = [
            "",
            "=" * 68,
            f"  Day {self.day} Summary — {self.phase}",
            "=" * 68,
            f"  Period       : {self.start_ts}  →  {self.end_ts}",
            f"  Samples      : {self.samples:,}",
            f"  GPU max (°C) : {_stats(self.gpu_max_readings)}",
            f"  CPU     (°C) : {_stats(self.cpu_readings)}",
            f"  Fan RPM      : {_stats([float(r) for r in self.fan_rpm_readings])}",
        ]
        if self.duty_pct_readings:
            lines.append(
                f"  AI duty (%)  : {_stats([float(d) for d in self.duty_pct_readings])}"
            )
        if self.predicted_temps:
            lines.append(
                f"  Pred T+60s   : {_stats(self.predicted_temps)}"
            )
        if self.watchdog_trips:
            lines.append(f"  WATCHDOG     : {self.watchdog_trips} trip(s)!")
        lines.append("=" * 68)
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_manual_control_active = False
_dry_run_global = False


def _cleanup(*_args: Any) -> None:
    """Revert fans on exit, regardless of reason."""
    global _manual_control_active
    if _manual_control_active:
        _log.warning("Cleanup: reverting fans to BIOS Auto.")
        ipmi_revert_auto(dry_run=_dry_run_global)
        _manual_control_active = False


def run_seven_day_pilot(
    days: int = 7,
    api_url: str = DEFAULT_API_URL,
    api_key: str = "",
    dry_run: bool = False,
) -> Path:
    global _manual_control_active, _dry_run_global
    _dry_run_global = dry_run

    # Safety: always revert on exit
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(143)))

    logger = PilotLogger()
    logger.open()
    predictor = TempPredictor60s(window=24)

    steps_per_day = SECONDS_PER_DAY // SAMPLE_INTERVAL_S

    _log.info(
        "Starting %d-day pilot on Lenovo ST550 (dry_run=%s, API=%s)",
        days, dry_run, api_url,
    )

    for day_idx in range(days):
        day_num = day_idx + 1
        phase = _day_phase(day_num)
        acc = DayAccumulator(day=day_num, phase=phase)
        acc.start_ts = datetime.now(timezone.utc).isoformat()

        _log.info("━━━ Day %d/%d — %s ━━━", day_num, days, phase)

        # ACTIVE mode: take IPMI manual control
        if phase == Phase.ACTIVE:
            if ipmi_enable_manual(dry_run):
                _manual_control_active = True
            else:
                _log.error("Could not enable manual fan control — falling back to BASELINE.")
                phase = Phase.BASELINE

        for step in range(steps_per_day):
            now_utc = datetime.now(timezone.utc)
            mono = time.monotonic()
            elapsed_s = day_idx * SECONDS_PER_DAY + step * SAMPLE_INTERVAL_S

            # --- Collect telemetry ---
            gpu_temps = read_gpu_temps(dry_run)
            cpu_temp = read_cpu_temp(dry_run)
            fans = read_fan_speeds(dry_run)

            gpu0 = gpu_temps[0] if len(gpu_temps) > 0 else None
            gpu1 = gpu_temps[1] if len(gpu_temps) > 1 else None
            gpu_max = max(gpu_temps) if gpu_temps else 0.0
            fan_rpm_avg = int(sum(fans.values()) / max(len(fans), 1)) if fans else 0
            fan_str = "; ".join(f"{k}={v}" for k, v in sorted(fans.items()))

            # Feed predictor
            predictor.add(mono, gpu_max)
            predicted_60s = predictor.predict(60.0)

            # ---  Safety watchdog ---
            if gpu_max >= GPU_WATCHDOG_LIMIT_C:
                _log.critical(
                    "WATCHDOG: GPU at %.1f°C >= %.0f°C limit! "
                    "Reverting to BIOS Auto and exiting.",
                    gpu_max, GPU_WATCHDOG_LIMIT_C,
                )
                acc.watchdog_trips += 1
                ipmi_revert_auto(dry_run)
                _manual_control_active = False
                logger.log_row({
                    "timestamp": now_utc.isoformat(),
                    "day": day_num, "phase": f"{phase}/WATCHDOG",
                    "elapsed_s": elapsed_s,
                    "gpu0_temp_c": gpu0, "gpu1_temp_c": gpu1,
                    "gpu_max_c": gpu_max, "cpu_temp_c": cpu_temp,
                    "fan_speeds_rpm": fan_str,
                    "ai_target_duty_pct": 100,
                    "ai_predicted_temp_60s": predicted_60s,
                    "ai_power_efficiency": 0,
                    "applied": "WATCHDOG_REVERT",
                })
                acc.end_ts = now_utc.isoformat()
                logger.append_report(acc.summary())
                logger.close()
                sys.exit(1)

            # --- Decide fan duty ---
            target_duty = -1
            ai_efficiency = 0.0
            applied = "no"

            if phase == Phase.BASELINE:
                applied = "bios_auto"

            elif phase == Phase.SHADOW:
                target_duty = p2000_target_duty(gpu_max)
                api_resp = call_optimize_thermal(
                    api_url, cpu_temp or 0.0, gpu_max, fan_rpm_avg, api_key,
                )
                if api_resp:
                    ai_efficiency = api_resp.get("power_efficiency_score", 0.0)
                applied = "shadow_only"

            elif phase == Phase.ACTIVE:
                target_duty = p2000_target_duty(gpu_max)
                api_resp = call_optimize_thermal(
                    api_url, cpu_temp or 0.0, gpu_max, fan_rpm_avg, api_key,
                )
                if api_resp:
                    ai_efficiency = api_resp.get("power_efficiency_score", 0.0)
                    raw_rpm = api_resp.get("target_rpm")
                    if raw_rpm is not None:
                        ai_duty = int(round((raw_rpm / 3000.0) * 100))
                        target_duty = max(target_duty, ai_duty)

                target_duty = max(20, min(100, target_duty))
                ipmi_set_fan_duty(target_duty, dry_run)
                applied = "yes"

            # --- Log ---
            logger.log_row({
                "timestamp": now_utc.isoformat(),
                "day": day_num, "phase": phase,
                "elapsed_s": elapsed_s,
                "gpu0_temp_c": gpu0, "gpu1_temp_c": gpu1,
                "gpu_max_c": round(gpu_max, 1),
                "cpu_temp_c": cpu_temp,
                "fan_speeds_rpm": fan_str,
                "ai_target_duty_pct": target_duty if target_duty >= 0 else "",
                "ai_predicted_temp_60s": round(predicted_60s, 1) if predicted_60s else "",
                "ai_power_efficiency": round(ai_efficiency, 1),
                "applied": applied,
            })
            acc.add(gpu_max, cpu_temp, fan_rpm_avg, target_duty, predicted_60s)

            # Progress
            if step % 360 == 0 or step == steps_per_day - 1:
                _log.info(
                    "  Day %d | %s | step %d/%d | GPU=%.1f°C | CPU=%s°C | "
                    "fans=%d RPM | duty=%s%%",
                    day_num, phase, step + 1, steps_per_day,
                    gpu_max,
                    f"{cpu_temp:.1f}" if cpu_temp else "?",
                    fan_rpm_avg,
                    str(target_duty) if target_duty >= 0 else "auto",
                )

            time.sleep(SAMPLE_INTERVAL_S)

        # --- End of day ---
        # Revert to Auto between days so BIOS resumes overnight
        if _manual_control_active:
            ipmi_revert_auto(dry_run)
            _manual_control_active = False

        acc.end_ts = datetime.now(timezone.utc).isoformat()
        report_text = acc.summary()
        logger.append_report(report_text)
        _log.info(report_text)

    logger.close()
    _log.info("7-day pilot complete → %s", logger.csv_path)
    return logger.csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI 7-Day Hardware Pilot — Lenovo ST550 + P2000",
    )
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to run (default: 7).")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL,
                        help="CooledAI API base URL.")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("COOLEDAI_API_KEY", ""),
                        help="X-API-Key (default: $COOLEDAI_API_KEY).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fake sensors, skip IPMI writes (for testing).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    csv_out = run_seven_day_pilot(
        days=args.days,
        api_url=args.api_url,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )
    print(f"\nResults: {csv_out}")
    print(f"Report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
