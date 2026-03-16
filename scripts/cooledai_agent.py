#!/usr/bin/env python3
"""
CooledAI Unified Agent — Hardware-Agnostic Thermal Controller

Auto-discovers sensors, reports telemetry to the CooledAI portal, pulls
remote config, and controls cooling via IPMI / PWM / GPU-fan fallback.

Designed to run on any Linux server (Dell, HP, Lenovo, SuperMicro, or
consumer boards).  Zero external dependencies — Python 3.8+ stdlib only.

Dual-mode loop
--------------
  Report  (every 10 s)  POST telemetry  ->  /api/v1/telemetry
  Listen  (every 60 s)  GET config      <-  /api/v1/config

Offline resilience: if the portal is unreachable the agent keeps cooling
with the last-known config cached to disk.

Safety watchdog: any sensor > CRITICAL_TEMP_C reverts all fans to
hardware auto and logs the event.

Usage
-----
  # Typical (systemd runs this via install.sh)
  sudo cooledai-agent --api-key sk-... --node-id rack-12-node-03

  # Dry-run (fake sensors, no fan writes)
  python3 scripts/cooledai_agent.py --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import logging
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger("cooledai-agent")

# ===================================================================
# Configuration (overridden by CLI / env / remote config)
# ===================================================================
DEFAULT_API_URL = "https://api.cooledai.com"
REPORT_INTERVAL_S = 10
CONFIG_INTERVAL_S = 60
DEFAULT_TARGET_TEMP = 65.0
DEFAULT_CRITICAL_TEMP = 90.0
SAFE_CONFIG_DIR = Path("/var/lib/cooledai")
SAFE_CONFIG_FILE = SAFE_CONFIG_DIR / "last_config.json"

BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
WARN_AFTER_FAILURES = 3

_running = True


# ===================================================================
# Sensor Discovery
# ===================================================================

@dataclass
class SensorCapabilities:
    cpu_sysfs_zones: List[str] = field(default_factory=list)
    cpu_lmsensors: bool = False
    nvidia_gpu: bool = False
    amd_gpu: bool = False
    ipmi: bool = False
    pwm_paths: List[str] = field(default_factory=list)

    # IPMI command variant discovered at startup
    ipmi_variant: str = ""  # "dell" or "lenovo"

    def summary(self) -> str:
        parts: List[str] = []
        if self.cpu_sysfs_zones:
            parts.append(f"CPU(sysfs:{len(self.cpu_sysfs_zones)}zones)")
        if self.cpu_lmsensors:
            parts.append("CPU(lm-sensors)")
        if self.nvidia_gpu:
            parts.append("GPU(nvidia)")
        if self.amd_gpu:
            parts.append("GPU(amd)")
        if self.ipmi:
            parts.append(f"IPMI({self.ipmi_variant or 'ok'})")
        if self.pwm_paths:
            parts.append(f"PWM({len(self.pwm_paths)})")
        return "Discovered: " + ("  ".join(parts) if parts else "NONE")


def _cmd_available(cmd: List[str], timeout: float = 5) -> bool:
    try:
        subprocess.check_output(cmd, timeout=timeout, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return True  # binary exists but returned non-zero (still usable)
    except Exception:
        return False


def discover_sensors(dry_run: bool = False) -> SensorCapabilities:
    caps = SensorCapabilities()

    if dry_run:
        caps.cpu_sysfs_zones = ["thermal_zone0", "thermal_zone1"]
        caps.nvidia_gpu = True
        caps.ipmi = True
        caps.ipmi_variant = "dell"
        return caps

    # CPU sysfs
    for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            val = int(Path(zone_path).read_text().strip())
            if val > 0:
                caps.cpu_sysfs_zones.append(
                    Path(zone_path).parent.name
                )
        except Exception:
            pass

    # lm-sensors
    caps.cpu_lmsensors = _cmd_available(["sensors", "--version"])

    # NVIDIA GPU
    caps.nvidia_gpu = _cmd_available(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
    )

    # AMD GPU
    caps.amd_gpu = _cmd_available(["rocm-smi", "--showtemp", "--csv"])

    # IPMI (probe Dell/generic first, then Lenovo)
    if _cmd_available(["ipmitool", "sdr"]):
        caps.ipmi = True
        try:
            subprocess.check_output(
                ["ipmitool", "raw", "0x30", "0x30", "0x01", "0x00"],
                timeout=5, stderr=subprocess.DEVNULL,
            )
            caps.ipmi_variant = "dell"
        except Exception:
            try:
                subprocess.check_output(
                    ["ipmitool", "raw", "0x32", "0x9b", "0x00"],
                    timeout=5, stderr=subprocess.DEVNULL,
                )
                caps.ipmi_variant = "lenovo"
            except Exception:
                caps.ipmi_variant = "generic"

    # PWM
    for pwm_path in sorted(glob.glob("/sys/class/hwmon/hwmon*/pwm[0-9]*")):
        if not pwm_path.endswith("_enable") and not pwm_path.endswith("_mode"):
            caps.pwm_paths.append(pwm_path)

    return caps


# ===================================================================
# Sensor Readers
# ===================================================================

@dataclass
class ThermalSnapshot:
    """All temperatures from a single read cycle."""
    cpu_temps: List[float] = field(default_factory=list)
    gpu_temps: List[float] = field(default_factory=list)
    gpu_fan_pcts: List[float] = field(default_factory=list)
    gpu_util_pcts: List[float] = field(default_factory=list)
    gpu_power_w: List[float] = field(default_factory=list)
    chassis_temps: List[float] = field(default_factory=list)
    fan_rpms: Dict[str, int] = field(default_factory=dict)
    max_temp_c: float = 0.0
    source_of_max: str = ""

    def compute_max(self) -> None:
        best = 0.0
        src = "none"
        for t in self.cpu_temps:
            if t > best:
                best, src = t, "cpu"
        for t in self.gpu_temps:
            if t > best:
                best, src = t, "gpu"
        for t in self.chassis_temps:
            if t > best:
                best, src = t, "chassis"
        self.max_temp_c = best
        self.source_of_max = src


def read_sensors(caps: SensorCapabilities, dry_run: bool = False) -> ThermalSnapshot:
    snap = ThermalSnapshot()
    t = time.time()

    if dry_run:
        snap.cpu_temps = [
            round(48 + 18 * math.sin(t * 0.008), 1),
            round(50 + 16 * math.sin(t * 0.009 + 0.3), 1),
        ]
        snap.gpu_temps = [
            round(54 + 15 * math.sin(t * 0.01), 1),
            round(56 + 14 * math.sin(t * 0.01 + 0.5), 1),
        ]
        snap.gpu_fan_pcts = [42.0, 44.0]
        snap.gpu_util_pcts = [
            round(50 + 30 * math.sin(t * 0.005), 1),
            round(45 + 28 * math.sin(t * 0.005 + 0.7), 1),
        ]
        snap.gpu_power_w = [65.0, 62.0]
        snap.chassis_temps = [round(30 + 5 * math.sin(t * 0.003), 1)]
        snap.fan_rpms = {"Fan1": 3200, "Fan2": 3150}
        snap.compute_max()
        return snap

    # CPU — sysfs
    for zone_name in caps.cpu_sysfs_zones:
        try:
            raw = Path(f"/sys/class/thermal/{zone_name}/temp").read_text().strip()
            snap.cpu_temps.append(int(raw) / 1000.0)
        except Exception:
            pass

    # CPU — lm-sensors (JSON)
    if caps.cpu_lmsensors:
        try:
            out = subprocess.check_output(
                ["sensors", "-j"], timeout=5, stderr=subprocess.DEVNULL
            ).decode()
            data = json.loads(out)
            for chip in data.values():
                if not isinstance(chip, dict):
                    continue
                for label, readings in chip.items():
                    if not isinstance(readings, dict):
                        continue
                    for key, val in readings.items():
                        if "input" in key and isinstance(val, (int, float)) and val > 0:
                            snap.cpu_temps.append(float(val))
        except Exception as exc:
            _log.debug("lm-sensors JSON parse failed: %s", exc)

    # NVIDIA GPU
    if caps.nvidia_gpu:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu,fan.speed,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        snap.gpu_temps.append(float(parts[0]))
                        snap.gpu_fan_pcts.append(float(parts[1]))
                        snap.gpu_util_pcts.append(float(parts[2]))
                        snap.gpu_power_w.append(float(parts[3]))
                    except ValueError:
                        pass
        except Exception as exc:
            _log.debug("nvidia-smi failed: %s", exc)

    # AMD GPU
    if caps.amd_gpu:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showtemp", "--csv"],
                timeout=10, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.splitlines()[1:]:
                m = re.search(r"([\d.]+)", line)
                if m:
                    snap.gpu_temps.append(float(m.group(1)))
        except Exception as exc:
            _log.debug("rocm-smi failed: %s", exc)

    # IPMI chassis temps + fan RPMs
    if caps.ipmi:
        try:
            out = subprocess.check_output(
                ["ipmitool", "sdr"], timeout=10, stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    continue
                name_lower = parts[0].lower()
                # Temps (ambient, inlet, exhaust)
                if any(k in name_lower for k in ("ambient", "inlet", "exhaust", "temp")):
                    m = re.search(r"([\d.]+)\s*degrees", parts[1], re.IGNORECASE)
                    if m:
                        snap.chassis_temps.append(float(m.group(1)))
                # Fans
                if "fan" in name_lower:
                    m = re.search(r"(\d+)\s*RPM", parts[1], re.IGNORECASE)
                    if m:
                        snap.fan_rpms[parts[0].strip()] = int(m.group(1))
        except Exception as exc:
            _log.debug("ipmitool sdr failed: %s", exc)

    snap.compute_max()
    return snap


# ===================================================================
# Control Layer — IPMI > PWM > GPU fan
# ===================================================================

def target_duty_pct(max_temp_c: float, target_temp: float) -> int:
    """Conservative thermal curve: map temp delta to fan duty %."""
    delta = max_temp_c - target_temp
    if delta < -15:
        return 25
    if delta < -5:
        return 40
    if delta < 0:
        return 55
    if delta < 8:
        return 70
    if delta < 15:
        return 85
    return 100


def _try_ipmi_set_duty(caps: SensorCapabilities, duty: int, dry_run: bool) -> bool:
    duty = max(0, min(100, duty))
    hex_duty = hex(duty)

    if dry_run:
        _log.debug("[DRY-RUN] IPMI set duty %d%%", duty)
        return True

    if caps.ipmi_variant == "dell":
        try:
            subprocess.check_call(
                ["ipmitool", "raw", "0x30", "0x30", "0x01", "0x01"],
                timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["ipmitool", "raw", "0x30", "0x30", "0x02", "0xff", hex_duty],
                timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            _log.debug("Dell IPMI set-duty failed: %s", exc)
            return False

    if caps.ipmi_variant == "lenovo":
        try:
            subprocess.check_call(
                ["ipmitool", "raw", "0x32", "0x9b", "0x01"],
                timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["ipmitool", "raw", "0x32", "0x69", "0x00", hex_duty],
                timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:
            _log.debug("Lenovo IPMI set-duty failed: %s", exc)
            return False

    return False


def _try_pwm_set_duty(caps: SensorCapabilities, duty: int, dry_run: bool) -> bool:
    if not caps.pwm_paths:
        return False
    pwm_value = int(duty * 255 / 100)
    pwm_value = max(0, min(255, pwm_value))
    if dry_run:
        _log.debug("[DRY-RUN] PWM set %d/255", pwm_value)
        return True
    success = False
    for pwm_path in caps.pwm_paths:
        enable_path = pwm_path + "_enable"
        try:
            Path(enable_path).write_text("1")
            Path(pwm_path).write_text(str(pwm_value))
            success = True
        except Exception as exc:
            _log.debug("PWM write %s failed: %s", pwm_path, exc)
    return success


def _try_gpu_fan_set(duty: int, dry_run: bool) -> bool:
    if dry_run:
        _log.debug("[DRY-RUN] GPU fan set %d%%", duty)
        return True
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.splitlines():
            idx = line.strip()
            subprocess.check_call(
                ["nvidia-settings", "-a",
                 f"[gpu:{idx}]/GPUFanControlState=1",
                 "-a", f"[fan:{idx}]/GPUTargetFanSpeed={duty}"],
                timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return True
    except Exception as exc:
        _log.debug("GPU fan set failed: %s", exc)
        return False


def set_fan_duty(
    caps: SensorCapabilities,
    duty: int,
    dry_run: bool = False,
) -> str:
    """Try IPMI -> PWM -> GPU fan.  Returns the method used or 'none'."""
    if caps.ipmi and _try_ipmi_set_duty(caps, duty, dry_run):
        return "ipmi"
    if caps.pwm_paths and _try_pwm_set_duty(caps, duty, dry_run):
        return "pwm"
    if (caps.nvidia_gpu or caps.amd_gpu) and _try_gpu_fan_set(duty, dry_run):
        return "gpu_fan"
    return "none"


def revert_fans_to_auto(caps: SensorCapabilities, dry_run: bool = False) -> None:
    """Best-effort revert to hardware-default fan control."""
    if dry_run:
        _log.info("[DRY-RUN] Reverting fans to auto.")
        return
    if caps.ipmi:
        try:
            if caps.ipmi_variant == "dell":
                subprocess.check_call(
                    ["ipmitool", "raw", "0x30", "0x30", "0x01", "0x00"],
                    timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.check_call(
                    ["ipmitool", "raw", "0x32", "0x9b", "0x00"],
                    timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            _log.info("Reverted IPMI fans to auto.")
        except Exception as exc:
            _log.warning("IPMI revert-auto failed: %s", exc)
    for pwm_path in caps.pwm_paths:
        enable_path = pwm_path + "_enable"
        try:
            Path(enable_path).write_text("2")  # 2 = automatic
        except Exception:
            pass


# ===================================================================
# HTTP helpers (stdlib only, same retry pattern as bridge_to_cloud)
# ===================================================================

def _http_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 10.0,
) -> Optional[Dict[str, Any]]:
    headers = {"X-API-Key": token}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        _log.debug("HTTP %d from %s", exc.code, url)
    except Exception as exc:
        _log.debug("Request failed (%s): %s", url, exc)
    return None


def post_with_backoff(
    url: str,
    payload: Dict[str, Any],
    token: str,
    max_attempts: int = 4,
) -> bool:
    """POST with limited retries (non-blocking — returns after max_attempts)."""
    backoff = BACKOFF_BASE_S
    for attempt in range(max_attempts):
        result = _http_request(url, token, "POST", payload)
        if result is not None:
            if attempt > 0:
                _log.info("Portal reconnected after %d retries.", attempt)
            return True
        if attempt < max_attempts - 1:
            jitter = random.uniform(0, backoff * 0.3)
            time.sleep(min(backoff + jitter, BACKOFF_CAP_S))
            backoff = min(backoff * 2, BACKOFF_CAP_S)
    return False


# ===================================================================
# Config cache (offline resilience)
# ===================================================================

@dataclass
class AgentConfig:
    target_temp: float = DEFAULT_TARGET_TEMP
    control_enabled: bool = True
    poll_interval_s: float = REPORT_INTERVAL_S
    critical_temp_c: float = DEFAULT_CRITICAL_TEMP

    def to_dict(self) -> dict:
        return {
            "target_temp": self.target_temp,
            "control_enabled": self.control_enabled,
            "poll_interval_s": self.poll_interval_s,
            "critical_temp_c": self.critical_temp_c,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        return cls(
            target_temp=float(d.get("target_temp", DEFAULT_TARGET_TEMP)),
            control_enabled=bool(d.get("control_enabled", True)),
            poll_interval_s=float(d.get("poll_interval_s", REPORT_INTERVAL_S)),
            critical_temp_c=float(d.get("critical_temp_c", DEFAULT_CRITICAL_TEMP)),
        )


def save_config(cfg: AgentConfig) -> None:
    try:
        SAFE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SAFE_CONFIG_FILE.write_text(json.dumps(cfg.to_dict(), indent=2))
    except Exception as exc:
        _log.debug("Could not save config cache: %s", exc)


def load_cached_config() -> AgentConfig:
    try:
        if SAFE_CONFIG_FILE.exists():
            data = json.loads(SAFE_CONFIG_FILE.read_text())
            _log.info("Loaded cached config from %s", SAFE_CONFIG_FILE)
            return AgentConfig.from_dict(data)
    except Exception as exc:
        _log.debug("Could not load cached config: %s", exc)
    return AgentConfig()


def fetch_remote_config(
    api_url: str,
    token: str,
    node_id: str,
) -> Optional[AgentConfig]:
    url = f"{api_url.rstrip('/')}/api/v1/config?node_id={node_id}"
    data = _http_request(url, token, "GET")
    if data is not None:
        cfg = AgentConfig.from_dict(data)
        save_config(cfg)
        return cfg
    return None


# ===================================================================
# Main loop
# ===================================================================

_manual_control_active = False
_dry_run_flag = False


def _cleanup(*_args: Any) -> None:
    global _manual_control_active
    if _manual_control_active:
        _log.warning("Cleanup: reverting fans to hardware auto.")
        caps = getattr(_cleanup, "_caps", SensorCapabilities())
        revert_fans_to_auto(caps, dry_run=_dry_run_flag)
        _manual_control_active = False


def run_agent(
    api_url: str,
    token: str,
    node_id: str,
    dry_run: bool = False,
) -> None:
    global _running, _manual_control_active, _dry_run_flag
    _dry_run_flag = dry_run

    # --- Discover ---
    caps = discover_sensors(dry_run)
    _cleanup._caps = caps  # type: ignore[attr-defined]
    _log.info(caps.summary())

    has_control = caps.ipmi or bool(caps.pwm_paths) or caps.nvidia_gpu
    if not has_control:
        _log.warning("No fan control method available — running in REPORT-ONLY mode.")

    # --- Safety hooks ---
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(143)))

    # --- Load config ---
    cfg = load_cached_config()
    telemetry_url = f"{api_url.rstrip('/')}/api/v1/telemetry"

    _log.info(
        "Agent started — node=%s  api=%s  target=%.0f°C  critical=%.0f°C  "
        "control=%s  dry_run=%s",
        node_id, api_url, cfg.target_temp, cfg.critical_temp_c,
        has_control, dry_run,
    )
    if not token:
        _log.warning("API key is empty — telemetry POSTs will likely be rejected.")

    last_config_fetch = 0.0
    consecutive_post_failures = 0

    while _running:
        loop_start = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Read all sensors (Max-Voter) ---
        snap = read_sensors(caps, dry_run)

        # --- Safety watchdog ---
        if snap.max_temp_c >= cfg.critical_temp_c:
            _log.critical(
                "WATCHDOG: %s at %.1f°C >= critical %.0f°C! "
                "Reverting fans and exiting.",
                snap.source_of_max, snap.max_temp_c, cfg.critical_temp_c,
            )
            revert_fans_to_auto(caps, dry_run)
            _manual_control_active = False
            sys.exit(1)

        # --- Control ---
        if has_control and cfg.control_enabled:
            duty = target_duty_pct(snap.max_temp_c, cfg.target_temp)
            method = set_fan_duty(caps, duty, dry_run)
            if method != "none":
                _manual_control_active = True
            else:
                _log.debug("Fan set returned 'none' — no method succeeded this cycle.")
        else:
            duty = -1
            method = "report_only"

        # --- Report telemetry ---
        records = []
        if snap.cpu_temps:
            records.append({
                "node_id": f"{node_id}/cpu",
                "timestamp": now_iso,
                "temperature_c": round(max(snap.cpu_temps), 1),
                "sensor_count": len(snap.cpu_temps),
            })
        for i, gt in enumerate(snap.gpu_temps):
            rec: Dict[str, Any] = {
                "node_id": f"{node_id}/gpu{i}",
                "timestamp": now_iso,
                "temperature_c": round(gt, 1),
            }
            if i < len(snap.gpu_fan_pcts):
                rec["fan_speed_pct"] = snap.gpu_fan_pcts[i]
            if i < len(snap.gpu_util_pcts):
                rec["utilization_pct"] = snap.gpu_util_pcts[i]
            if i < len(snap.gpu_power_w):
                rec["power_draw_w"] = snap.gpu_power_w[i]
            records.append(rec)
        if snap.chassis_temps:
            records.append({
                "node_id": f"{node_id}/chassis",
                "timestamp": now_iso,
                "temperature_c": round(max(snap.chassis_temps), 1),
            })
        if snap.fan_rpms:
            records.append({
                "node_id": f"{node_id}/fans",
                "timestamp": now_iso,
                "fan_rpms": snap.fan_rpms,
            })

        payload = {"agent_id": node_id, "telemetry": records}
        ok = post_with_backoff(telemetry_url, payload, token, max_attempts=3)
        if ok:
            consecutive_post_failures = 0
        else:
            consecutive_post_failures += 1
            if consecutive_post_failures == WARN_AFTER_FAILURES:
                _log.warning(
                    "Portal unreachable for %d cycles — cooling continues "
                    "with cached config (target=%.0f°C).",
                    consecutive_post_failures, cfg.target_temp,
                )

        # --- Fetch config (every CONFIG_INTERVAL_S) ---
        now_mono = time.monotonic()
        if now_mono - last_config_fetch >= CONFIG_INTERVAL_S:
            remote_cfg = fetch_remote_config(api_url, token, node_id)
            if remote_cfg is not None:
                if remote_cfg.target_temp != cfg.target_temp:
                    _log.info(
                        "Config updated: target_temp %.0f -> %.0f",
                        cfg.target_temp, remote_cfg.target_temp,
                    )
                cfg = remote_cfg
            last_config_fetch = now_mono

        # --- Pace ---
        elapsed = time.monotonic() - loop_start
        sleep_s = max(0, cfg.poll_interval_s - elapsed)
        if sleep_s > 0 and _running:
            time.sleep(sleep_s)


def _shutdown_signal(signum: int, _frame: Any) -> None:
    global _running
    _running = False


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CooledAI Unified Agent — Hardware-Agnostic Thermal Controller",
    )
    parser.add_argument("--api-url", default=os.environ.get("COOLEDAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key", default=os.environ.get("COOLEDAI_API_KEY", ""))
    parser.add_argument("--node-id", default=os.environ.get("COOLEDAI_NODE_ID", "node-01"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Fake sensors, no fan writes.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    signal.signal(signal.SIGINT, _shutdown_signal)
    signal.signal(signal.SIGTERM, _shutdown_signal)

    run_agent(
        api_url=args.api_url,
        token=args.api_key,
        node_id=args.node_id,
        dry_run=args.dry_run,
    )
    _log.info("Agent stopped.")


if __name__ == "__main__":
    main()
