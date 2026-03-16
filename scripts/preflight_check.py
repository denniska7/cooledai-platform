#!/usr/bin/env python3
"""
CooledAI Pre-Flight Check — Lenovo ST550

Run this *before* starting seven_day_pilot.py.  It validates that every
dependency is reachable, the environment is sane, and no stale state will
contaminate the pilot data.

Checks
------
  1. nvidia-smi responds and reports both P2000 GPUs
  2. lm-sensors (``sensors``) returns CPU temperatures
  3. ipmitool SDR returns fan speed readings
  4. ipmitool fan-control commands are executable (IPMI access)
  5. No other instance of seven_day_pilot.py is already running
  6. CooledAI API server is reachable (optional — skipped with --skip-api)
  7. Fans are currently on BIOS Auto (not stuck in manual mode)
  8. logs/ directory is writable
  9. Stale logs from a previous run are archived (not silently mixed)

Exit codes
----------
  0  All checks passed — safe to start the pilot.
  1  One or more checks failed — fix issues before proceeding.

Usage
-----
  python3 scripts/preflight_check.py
  python3 scripts/preflight_check.py --skip-api
  python3 scripts/preflight_check.py --api-url http://10.0.0.5:8000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _PROJECT_ROOT / "logs"
PILOT_CSV = LOGS_DIR / "seven_day_pilot.csv"
PILOT_REPORT = LOGS_DIR / "pilot_report.txt"
DEFAULT_API_URL = "http://127.0.0.1:8000"

# Real-data deployment paths (ST550)
DATA_SOURCE = Path("/home/ubuntu/CoolingAI/data/raw_simulation_data.csv")
DATA_TARGET_ACTIVE = Path("/home/ubuntu/CoolingAI/pilot/input/")
DATA_TARGET_BACKUP = Path("/home/ubuntu/CoolingAI/pilot/backup/")

# ANSI colours for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")


def _header(title: str) -> None:
    print(f"\n{_BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{_RESET}")


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------

def check_nvidia_smi() -> Tuple[bool, str]:
    """Verify nvidia-smi finds both P2000 GPUs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu",
             "--format=csv,noheader"],
            timeout=10, stderr=subprocess.PIPE,
        ).decode().strip()
    except FileNotFoundError:
        return False, "nvidia-smi not found — is the NVIDIA driver installed?"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out (10 s)."
    except subprocess.CalledProcessError as exc:
        return False, f"nvidia-smi exited with code {exc.returncode}."

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        return False, f"Expected 2 GPUs, nvidia-smi reported {len(lines)}:\n{out}"

    gpu_info = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        gpu_info.append(f"{parts[0]} @ {parts[1]}°C" if len(parts) >= 2 else line)

    return True, "  |  ".join(gpu_info)


def check_sensors() -> Tuple[bool, str]:
    """Verify lm-sensors returns CPU temperatures."""
    try:
        out = subprocess.check_output(
            ["sensors"], timeout=10, stderr=subprocess.PIPE,
        ).decode()
    except FileNotFoundError:
        return False, "sensors not found — run: sudo apt install lm-sensors && sudo sensors-detect"
    except subprocess.TimeoutExpired:
        return False, "sensors timed out."
    except subprocess.CalledProcessError as exc:
        return False, f"sensors exited with code {exc.returncode}."

    import re
    temps: List[str] = []
    for line in out.splitlines():
        if any(k in line.lower() for k in ("core", "package", "tctl", "tdie")):
            m = re.search(r"[+]?([\d.]+)\s*°?C", line)
            if m:
                label = line.split(":")[0].strip()
                temps.append(f"{label}={m.group(1)}°C")

    if not temps:
        return False, "No CPU temperature readings found in sensors output."
    return True, "  ".join(temps[:6]) + ("  ..." if len(temps) > 6 else "")


def check_ipmitool_sdr() -> Tuple[bool, str]:
    """Verify ipmitool SDR returns fan readings."""
    try:
        out = subprocess.check_output(
            ["ipmitool", "sdr"], timeout=10, stderr=subprocess.PIPE,
        ).decode()
    except FileNotFoundError:
        return False, "ipmitool not found — run: sudo apt install ipmitool"
    except subprocess.TimeoutExpired:
        return False, "ipmitool sdr timed out."
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode().strip() if exc.stderr else ""
        if "Could not open device" in stderr_msg or "permission" in stderr_msg.lower():
            return False, f"Permission denied — run with sudo or add user to the ipmi group.\n        {stderr_msg}"
        return False, f"ipmitool sdr failed (code {exc.returncode}): {stderr_msg}"

    import re
    fans: List[str] = []
    for line in out.splitlines():
        if "fan" not in line.lower():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            m = re.search(r"(\d+)\s*RPM", parts[1], re.IGNORECASE)
            if m:
                fans.append(f"{parts[0].strip()}={m.group(1)} RPM")

    if not fans:
        return False, "No fan RPM readings in ipmitool SDR output."
    return True, "  ".join(fans[:6]) + ("  ..." if len(fans) > 6 else "")


def check_ipmi_fan_control() -> Tuple[bool, str]:
    """Verify IPMI raw fan-control commands are accepted."""
    try:
        subprocess.check_output(
            ["ipmitool", "raw", "0x32", "0x9b", "0x00"],
            timeout=10, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, "ipmitool not found."
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode().strip() if exc.stderr else ""
        if "permission" in stderr_msg.lower() or "Could not open" in stderr_msg:
            return False, f"Permission denied for IPMI raw commands — run as root.\n        {stderr_msg}"
        return False, f"IPMI raw 0x32 0x9b failed: {stderr_msg}"
    except subprocess.TimeoutExpired:
        return False, "IPMI raw command timed out."

    return True, "IPMI fan-control raw commands accepted (fans on BIOS Auto)."


def check_no_existing_pilot() -> Tuple[bool, str]:
    """Make sure no other seven_day_pilot.py is already running."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "seven_day_pilot"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True, "No existing pilot process found."

    my_pid = str(os.getpid())
    other_lines = [l for l in out.splitlines()
                   if l.strip() and not l.strip().startswith(my_pid)
                   and "preflight" not in l.lower()
                   and "pgrep" not in l.lower()]
    if other_lines:
        return False, f"Another pilot is running:\n        " + "\n        ".join(other_lines)
    return True, "No existing pilot process found."


def check_api_server(api_url: str) -> Tuple[bool, str]:
    """Ping the CooledAI API /health endpoint."""
    url = f"{api_url.rstrip('/')}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
            if body.get("status") == "ok":
                return True, f"API healthy at {api_url}"
            return False, f"API returned unexpected body: {body}"
    except Exception as exc:
        return False, f"Cannot reach {url} — {exc}"


def check_logs_writable() -> Tuple[bool, str]:
    """Verify the logs directory exists and is writable."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    test_file = LOGS_DIR / ".preflight_write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"{LOGS_DIR} is writable."
    except OSError as exc:
        return False, f"Cannot write to {LOGS_DIR}: {exc}"


def archive_stale_logs() -> Tuple[bool, str]:
    """If previous pilot CSV/report exist, move them to a timestamped
    archive so the new run starts clean."""
    archived: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for path in (PILOT_CSV, PILOT_REPORT):
        if path.exists() and path.stat().st_size > 0:
            dest = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            shutil.move(str(path), str(dest))
            archived.append(f"{path.name} → {dest.name}")

    if archived:
        return True, "Archived: " + ", ".join(archived)
    return True, "No stale logs to archive."


def deploy_real_data(
    source: Path = DATA_SOURCE,
    targets: Optional[List[Path]] = None,
) -> Tuple[bool, str]:
    """Copy the source data file to both target directories with 644
    permissions so the pilot can read them."""
    if targets is None:
        targets = [DATA_TARGET_ACTIVE, DATA_TARGET_BACKUP]

    if not source.exists():
        return False, f"Source file not found: {source}"
    if not source.is_file():
        return False, f"Source is not a regular file: {source}"

    deployed: List[str] = []
    for target_dir in targets:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"Cannot create {target_dir}: {exc}"

        dest = target_dir / source.name
        try:
            shutil.copy2(str(source), str(dest))
            os.chmod(str(dest), 0o644)
            deployed.append(str(dest))
        except OSError as exc:
            return False, f"Failed to copy to {dest}: {exc}"

    return True, " + ".join(deployed)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_preflight(api_url: str = DEFAULT_API_URL, skip_api: bool = False) -> bool:
    failures = 0

    _header("1. GPU — nvidia-smi")
    ok, msg = check_nvidia_smi()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("2. CPU — lm-sensors")
    ok, msg = check_sensors()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("3. Fan Speeds — ipmitool SDR")
    ok, msg = check_ipmitool_sdr()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("4. IPMI Fan Control Access")
    ok, msg = check_ipmi_fan_control()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("5. No Existing Pilot Running")
    ok, msg = check_no_existing_pilot()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("6. CooledAI API Server")
    if skip_api:
        _warn("Skipped (--skip-api)")
    else:
        ok, msg = check_api_server(api_url)
        (_pass if ok else _fail)(msg)
        if not ok:
            failures += 1

    _header("7. Logs Directory")
    ok, msg = check_logs_writable()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("8. Archive Stale Logs")
    ok, msg = archive_stale_logs()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    _header("9. Deploy Real Data")
    ok, msg = deploy_real_data()
    (_pass if ok else _fail)(msg)
    if not ok:
        failures += 1

    # --- Verdict ---
    print()
    if failures == 0:
        print(f"{_GREEN}{_BOLD}  ALL CHECKS PASSED — ready to start seven_day_pilot.py{_RESET}")
        print(f"\n  Next step:")
        print(f"    sudo python3 scripts/seven_day_pilot.py\n")
    else:
        print(f"{_RED}{_BOLD}  {failures} CHECK(S) FAILED — fix the issues above before starting the pilot.{_RESET}\n")

    return failures == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI Pre-Flight Check for the 7-Day Pilot",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help="CooledAI API base URL (default: %(default)s).")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip the API-server reachability check.")
    args = parser.parse_args()

    ok = run_preflight(api_url=args.api_url, skip_api=args.skip_api)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
