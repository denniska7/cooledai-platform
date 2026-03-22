#!/usr/bin/env python3
"""
CooledAI Unified Agent — Hardware-Agnostic Thermal Controller

Auto-discovers sensors, reports telemetry to the CooledAI portal, pulls
remote config, and controls cooling via IPMI / PWM / GPU-fan fallback.

Optional Redfish/XCC enrichment: ``--redfish-url`` / ``COOLEDAI_REDFISH_URL``
reads fan percentages and ambient/exhaust/CPU temperatures from the Lenovo
XCC BMC over the USB-ethernet management interface for richer telemetry.

Optional GPU dynamic TDP: ``--gpu-power-management`` / ``COOLEDAI_GPU_POWER_MGMT``
uses ``nvidia-smi -pl`` with GPU temperature bands; see
``docs/GPU_AND_CPU_POWER_PHASE2.md``.

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
import base64
import glob
import json
import logging
import math
import os
import random
import re
import signal
import socket
import ssl
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
REPORT_INTERVAL_S = 3  # Optimize + telemetry every 3s; fallback to local curve when offline
CONFIG_INTERVAL_S = 60
OPTIMIZE_TIMEOUT_S = 2.0  # Don't block cooling loop waiting for API
DEFAULT_TARGET_TEMP = 65.0
DEFAULT_CRITICAL_TEMP = 90.0
SAFE_CONFIG_DIR = Path("/var/lib/cooledai")
SAFE_CONFIG_FILE = SAFE_CONFIG_DIR / "last_config.json"

BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
WARN_AFTER_FAILURES = 3
# Map nvidia-smi GPU fan % (0–100) to chart RPM scale (chassis fans are real RPM; this aligns scales)
GPU_FAN_PCT_TO_RPM_MAX = float(os.environ.get("COOLEDAI_GPU_FAN_PCT_RPM_MAX", "4200"))

_running = True

# GPU dynamic power limit (Phase 2) — optional; requires repo `core/` on sys.path
_GPU_GPG_MOD: Any = None
_GPU_PL_ENVELOPES: List[Any] = []
_GPU_PL_STATE: Dict[str, Any] = {}

# GPU Power Governor (Phase 2 class-based) — replaces free-function wiring
_GPU_GOVERNOR: Any = None

# Thermal calibrator — optional; available when running from repo checkout
_THERMAL_CALIBRATOR: Any = None

# XCC Fan Controller — replaces IPMI bypass for Lenovo ThinkSystem servers
_XCC_FAN_CONTROLLER: Any = None


def _try_load_xcc_fan_controller() -> Any:
    """Import XCCFanController if agent runs from repo checkout."""
    p = Path(__file__).resolve()
    for root in (p.parent.parent, p.parent):
        if (root / "core" / "hardware" / "xcc_fan_controller.py").is_file():
            rs = str(root)
            if rs not in sys.path:
                sys.path.insert(0, rs)
            try:
                from core.hardware.xcc_fan_controller import XCCFanController
                return XCCFanController
            except Exception as exc:
                _log.debug("XCCFanController import failed: %s", exc)
                return None
    return None


def _try_load_gpu_power_governor() -> Any:
    """Import core.optimization.gpu_power_governor if agent runs from repo checkout."""
    p = Path(__file__).resolve()
    for root in (p.parent.parent, p.parent):
        if (root / "core" / "optimization" / "gpu_power_governor.py").is_file():
            rs = str(root)
            if rs not in sys.path:
                sys.path.insert(0, rs)
            try:
                from core.optimization import gpu_power_governor as gpg

                return gpg
            except Exception as exc:
                _log.debug("gpu_power_governor import failed: %s", exc)
                return None
    return None


def _try_load_thermal_calibrator() -> Any:
    """Import ThermalCalibrator if agent runs from repo checkout."""
    p = Path(__file__).resolve()
    for root in (p.parent.parent, p.parent):
        if (root / "core" / "optimization" / "thermal_calibrator.py").is_file():
            rs = str(root)
            if rs not in sys.path:
                sys.path.insert(0, rs)
            try:
                from core.optimization.thermal_calibrator import ThermalCalibrator
                return ThermalCalibrator
            except Exception as exc:
                _log.debug("ThermalCalibrator import failed: %s", exc)
                return None
    return None


def _gpu_power_limits_cleanup() -> None:
    """Restore default nvidia-smi power limits on shutdown / watchdog."""
    global _GPU_GPG_MOD, _GPU_PL_ENVELOPES, _GPU_GOVERNOR
    # Phase 2 governor (preferred)
    if _GPU_GOVERNOR is not None:
        try:
            _GPU_GOVERNOR.reset_to_defaults()
        except Exception as exc:
            _log.debug("GPU governor cleanup failed: %s", exc)
        return
    # Legacy free-function path
    gpg = _GPU_GPG_MOD
    if not gpg or not _GPU_PL_ENVELOPES or _dry_run_flag:
        return
    try:
        gpg.reset_all_gpus_to_default(_GPU_PL_ENVELOPES, dry_run=False)
        _log.info("GPU power limits restored to driver defaults.")
    except Exception as exc:
        _log.debug("GPU power cleanup failed: %s", exc)


# ===================================================================
# systemd Watchdog (sd_notify)
# ===================================================================
# When WatchdogSec is set, systemd provides NOTIFY_SOCKET.  The agent
# must send WATCHDOG=1 periodically or systemd will restart it.
# We only arm the watchdog AFTER the first successful telemetry POST,
# so slow NVML/driver init during boot does not trigger a false restart.


def _sd_notify_watchdog() -> bool:
    """Send WATCHDOG=1 to systemd.  No-op if NOTIFY_SOCKET unset."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            addr = ("\0" + sock_path[1:]) if sock_path.startswith("@") else sock_path
            s.sendto(b"WATCHDOG=1", addr)
        return True
    except Exception:
        return False


# ===================================================================
# Redfish / XCC Client (Lenovo ThinkSystem BMC)
# ===================================================================
# The Lenovo XCC exposes a Redfish API over the USB-ethernet management
# interface (typically 169.254.95.118).  This gives us:
#   - Fan speeds as percent (more granular than IPMI SDR RPM)
#   - Ambient, exhaust, CPU, DTS temperatures
#   - BIOS operating-mode visibility
# Direct fan-speed PATCH is not supported (405), but reading enriches
# telemetry and enables smarter optimization decisions.

_REDFISH_SSL_CTX: Optional[ssl.SSLContext] = None


def _redfish_ssl_ctx() -> ssl.SSLContext:
    """Lazy-init a permissive SSL context for self-signed XCC certs."""
    global _REDFISH_SSL_CTX
    if _REDFISH_SSL_CTX is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _REDFISH_SSL_CTX = ctx
    return _REDFISH_SSL_CTX


@dataclass
class RedfishClient:
    """Lightweight Redfish client for Lenovo XCC (stdlib only)."""

    base_url: str  # e.g. "https://169.254.95.118"
    username: str = "USERID"
    password: str = "***REDACTED_BMC_PASS***"
    timeout_s: float = 5.0
    _auth_header: str = ""

    def __post_init__(self) -> None:
        cred = f"{self.username}:{self.password}".encode()
        self._auth_header = f"Basic {base64.b64encode(cred).decode()}"

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        """GET a Redfish resource.  Returns parsed JSON or None on error."""
        url = self.base_url.rstrip("/") + path
        req = urllib.request.Request(
            url, headers={"Authorization": self._auth_header}
        )
        try:
            with urllib.request.urlopen(
                req, context=_redfish_ssl_ctx(), timeout=self.timeout_s
            ) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            _log.debug("Redfish GET %s failed: %s", path, exc)
            return None

    def is_reachable(self) -> bool:
        """Quick connectivity check via service root."""
        return self.get("/redfish/v1/") is not None

    def read_thermal(self) -> Optional[Dict[str, Any]]:
        """Read /redfish/v1/Chassis/1/Thermal/ — fans + temperatures."""
        return self.get("/redfish/v1/Chassis/1/Thermal/")

    def read_bios_attrs(self) -> Optional[Dict[str, Any]]:
        """Read current BIOS attributes."""
        data = self.get("/redfish/v1/Systems/1/Bios/")
        if data:
            return data.get("Attributes", {})
        return None


def _discover_redfish(
    redfish_url: str,
    redfish_user: str,
    redfish_pass: str,
) -> Optional[RedfishClient]:
    """Probe the XCC Redfish endpoint.  Returns client if reachable."""
    if not redfish_url:
        return None
    client = RedfishClient(
        base_url=redfish_url,
        username=redfish_user,
        password=redfish_pass,
    )
    try:
        if client.is_reachable():
            _log.info("Redfish XCC reachable at %s", redfish_url)
            # Log BIOS operating mode for operator visibility
            attrs = client.read_bios_attrs()
            if attrs:
                mode = attrs.get("OperatingModes_ChooseOperatingMode", "?")
                _log.info("BIOS OperatingMode: %s", mode)
            return client
    except Exception as exc:
        _log.debug("Redfish discovery failed: %s", exc)
    _log.info("Redfish XCC not reachable at %s — continuing without it", redfish_url)
    return None


@dataclass
class RedfishThermalSnapshot:
    """Parsed Redfish Thermal data."""

    fan_pcts: Dict[str, float] = field(default_factory=dict)  # "Fan 1" -> 56.0
    temps: Dict[str, float] = field(default_factory=dict)  # "Ambient" -> 25.0
    ambient_temp: Optional[float] = None
    exhaust_temp: Optional[float] = None
    cpu_temps: List[float] = field(default_factory=list)


def _read_redfish_thermal(client: RedfishClient) -> Optional[RedfishThermalSnapshot]:
    """Read and parse XCC thermal data."""
    data = client.read_thermal()
    if data is None:
        return None
    snap = RedfishThermalSnapshot()

    # Fans — XCC returns Reading as percent (0-100)
    for fan in data.get("Fans", []):
        name = fan.get("FanName") or fan.get("Name") or fan.get("MemberId", "?")
        reading = fan.get("Reading")
        if reading is not None:
            snap.fan_pcts[name] = float(reading)

    # Temperatures
    for temp_entry in data.get("Temperatures", []):
        reading = temp_entry.get("ReadingCelsius")
        if reading is None or reading < -50:
            continue
        name = temp_entry.get("Name", "")
        snap.temps[name] = float(reading)
        name_lower = name.lower()
        if "ambient" in name_lower:
            snap.ambient_temp = float(reading)
        elif "exhaust" in name_lower:
            snap.exhaust_temp = float(reading)
        elif "cpu" in name_lower and "dts" not in name_lower:
            snap.cpu_temps.append(float(reading))

    return snap


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

    # Redfish / XCC client (Lenovo ThinkSystem BMC)
    redfish: Optional[RedfishClient] = None

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
        if self.redfish:
            parts.append(f"Redfish({self.redfish.base_url})")
        if _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available:
            parts.append("XCC_FAN(active)")
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


def discover_sensors(dry_run: bool = False, ipmi_variant: Optional[str] = None) -> SensorCapabilities:
    caps = SensorCapabilities()

    if dry_run:
        caps.cpu_sysfs_zones = ["thermal_zone0", "thermal_zone1"]
        caps.nvidia_gpu = True
        caps.ipmi = True
        caps.ipmi_variant = ipmi_variant or "dell"
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

    # IPMI (forced variant, or probe Dell then Lenovo)
    if _cmd_available(["ipmitool", "sdr"]):
        caps.ipmi = True
        if ipmi_variant:
            caps.ipmi_variant = ipmi_variant
        else:
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
    fan_power_w: Optional[float] = None
    max_temp_c: float = 0.0
    source_of_max: str = ""
    # FIX A/E: Peak GPU power within polling interval (raw max, not EWMA)
    peak_gpu_power_w: Optional[float] = None

    # Redfish enrichment (XCC BMC data)
    redfish_fan_pcts: Dict[str, float] = field(default_factory=dict)
    redfish_temps: Dict[str, float] = field(default_factory=dict)
    redfish_ambient_temp: Optional[float] = None
    redfish_exhaust_temp: Optional[float] = None

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
        # FIX A/E: Peak GPU power
        if self.gpu_power_w:
            self.peak_gpu_power_w = max(self.gpu_power_w)


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
                        temp = float(parts[0])
                        util = float(parts[2])
                        raw_power = (parts[3].strip() if len(parts) > 3 else "").upper()
                        # nvidia-smi can return "N/A", "[N/A]", "[Not Available]", or empty
                        if raw_power and "N/A" not in raw_power and "NOT" not in raw_power:
                            power = float(parts[3].strip())
                        else:
                            power = 0.0
                        # nvidia-smi often reports 0W when idle; avoid ruining telemetry
                        if power <= 0:
                            power = max(10.0, util * 1.2) if util > 0 else 10.0
                        snap.gpu_temps.append(temp)
                        snap.gpu_fan_pcts.append(float(parts[1]))
                        snap.gpu_util_pcts.append(util)
                        snap.gpu_power_w.append(power)
                    except (ValueError, IndexError):
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

    # IPMI chassis temps + fan RPMs + fan power
    # ST550 labels: "Fan 1 Tach", "Fan 2 Tach", ..., "Sys Fan Pwr"
    if caps.ipmi:
        try:
            out = subprocess.check_output(
                ["ipmitool", "sdr"], timeout=10, stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    continue
                name = parts[0].strip()
                name_lower = name.lower()
                value_str = parts[1].strip()

                # Temps (ambient, inlet, exhaust)
                if any(k in name_lower for k in ("ambient", "inlet", "exhaust", "temp")):
                    m = re.search(r"([\d.]+)\s*degrees", value_str, re.IGNORECASE)
                    if m:
                        snap.chassis_temps.append(float(m.group(1)))

                # Sys Fan Pwr → wattage (not RPM)
                if "sys fan pwr" in name_lower:
                    m = re.search(r"([\d.]+)\s*[Ww]", value_str)
                    if m:
                        snap.fan_power_w = float(m.group(1))
                    continue

                # Fan tach (IPMI): "Fan 1 Tach", "Fan1", "System Fan", etc.
                if "fan" in name_lower or "tach" in name_lower:
                    m = re.search(r"(\d+)\s*RPM", value_str, re.IGNORECASE)
                    if m:
                        snap.fan_rpms[name] = int(m.group(1))
                    else:
                        # Some boards report raw RPM without the word "RPM"
                        m2 = re.search(r"\b(\d{3,5})\b", value_str)
                        if m2 and "na" not in value_str.lower():
                            v = int(m2.group(1))
                            if 200 <= v <= 20000:
                                snap.fan_rpms[name] = v
        except Exception as exc:
            _log.debug("ipmitool sdr failed: %s", exc)

    # Redfish XCC — enriched thermal data (ambient, exhaust, CPU, fan %)
    if caps.redfish is not None:
        rf = _read_redfish_thermal(caps.redfish)
        if rf is not None:
            snap.redfish_fan_pcts = rf.fan_pcts
            snap.redfish_temps = rf.temps
            snap.redfish_ambient_temp = rf.ambient_temp
            snap.redfish_exhaust_temp = rf.exhaust_temp
            # Merge Redfish CPU temps if sysfs/lm-sensors missed them
            if not snap.cpu_temps and rf.cpu_temps:
                snap.cpu_temps = rf.cpu_temps
            # Merge Redfish ambient/exhaust into chassis temps if IPMI missed them
            if not snap.chassis_temps:
                if rf.ambient_temp is not None:
                    snap.chassis_temps.append(rf.ambient_temp)
                if rf.exhaust_temp is not None:
                    snap.chassis_temps.append(rf.exhaust_temp)

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


def _get_xcc_override_threshold_rpm() -> float:
    """Profile-driven XCC override threshold: fan_idle_rpm + 40% of fan_range_rpm.

    Falls back to 3500 RPM when calibration profile is not yet available.
    """
    try:
        if (_THERMAL_CALIBRATOR is not None
                and _THERMAL_CALIBRATOR.is_calibrated
                and hasattr(_THERMAL_CALIBRATOR, "profile")
                and _THERMAL_CALIBRATOR.profile is not None):
            p = _THERMAL_CALIBRATOR.profile
            return p.fan_idle_rpm + (p.fan_range_rpm * 0.40)
    except Exception:
        pass
    return 3500.0


def _try_xcc_fan_set(duty: int, rated_max_rpm: float, hw_rpm: float, dry_run: bool) -> bool:
    """XCC Redfish fan control: set fan speed via BMC when hardware ignores normal commands.

    Replaces the old IPMI bypass (ipmitool raw commands) which does not work on
    Lenovo ST550 servers (no /dev/ipmi0 exposed to OS).

    Fires when target > profile-driven threshold AND hardware is not responding (< 85% of target).
    """
    global _XCC_FAN_CONTROLLER
    if _XCC_FAN_CONTROLLER is None or not _XCC_FAN_CONTROLLER.available:
        return False

    target_rpm = (duty / 100.0) * rated_max_rpm
    threshold_rpm = _get_xcc_override_threshold_rpm()
    if target_rpm <= threshold_rpm:
        return False
    if hw_rpm > 0 and hw_rpm >= target_rpm * 0.85:
        return False  # hardware is responding adequately

    fan_pct = min(100, max(0, int(round((target_rpm / rated_max_rpm) * 100))))

    if dry_run:
        _log.debug("[DRY-RUN] XCC_FAN: would send %d%% (target_rpm=%.0f, hw_rpm=%.0f)",
                   fan_pct, target_rpm, hw_rpm)
        return True

    _log.info(
        "XCC_FAN_OVERRIDE: target_rpm=%.0f actual_hw_rpm=%.0f "
        "— setting XCC fan to %d%% via Redfish",
        target_rpm, hw_rpm, fan_pct,
    )
    return _XCC_FAN_CONTROLLER.set_manual_fan_percent(fan_pct)


def _probe_xcc_health() -> bool:
    """Lightweight XCC health check — returns True if BMC is responsive."""
    if _XCC_FAN_CONTROLLER is None or not _XCC_FAN_CONTROLLER.available:
        return False
    try:
        fans = _XCC_FAN_CONTROLLER.read_fan_speeds()
        return bool(fans)
    except Exception:
        return False


def _xcc_failsafe_watchdog(dry_run: bool, current_duty_pct: int = 50) -> None:
    """30-second XCC failsafe watchdog — verifies connectivity, activates failsafe or reconnects."""
    global _optimization_owns_control, _last_successful_xcc_command_ts
    global _xcc_watchdog_last_check

    now = time.monotonic()
    if now - _xcc_watchdog_last_check < _XCC_WATCHDOG_INTERVAL_S:
        return
    _xcc_watchdog_last_check = now

    if _XCC_FAN_CONTROLLER is None:
        return

    if _optimization_owns_control:
        # Check for command gap
        gap = now - _last_successful_xcc_command_ts
        if gap > _XCC_FAILSAFE_TIMEOUT_S:
            _log.warning(
                "[XCC_FAN] WARNING: command_gap=%.0fs — verifying connectivity",
                gap,
            )
            if _probe_xcc_health():
                _last_successful_xcc_command_ts = now
                _log.info("[XCC_FAN] Probe succeeded — connectivity OK")
            else:
                _log.warning(
                    "[XCC_FAN] FAILSAFE_ACTIVATED — XCC auto mode restored "
                    "after connectivity loss"
                )
                _restore_xcc_auto(dry_run)
                _optimization_owns_control = False
    else:
        # Reconnect loop: try to retake control
        if _XCC_FAN_CONTROLLER.available and _probe_xcc_health():
            if not dry_run:
                pct = max(30, min(100, current_duty_pct))
                if _XCC_FAN_CONTROLLER.set_manual_fan_percent(pct):
                    _optimization_owns_control = True
                    _last_successful_xcc_command_ts = now
                    _log.info(
                        "[XCC_FAN] RECONNECTED — optimization_owns_control=True "
                        "(fan_pct=%d)",
                        pct,
                    )


def _restore_xcc_auto(dry_run: bool) -> None:
    """Restore automatic fan control via XCC after override is no longer needed."""
    global _XCC_FAN_CONTROLLER
    if _XCC_FAN_CONTROLLER is None or not _XCC_FAN_CONTROLLER.available:
        return
    if dry_run:
        _log.debug("[DRY-RUN] XCC auto restore")
        return
    if _XCC_FAN_CONTROLLER.restore_auto_control():
        _log.info("XCC_FAN_RESTORED: automatic fan control re-enabled via Redfish.")
    else:
        _log.warning("XCC_FAN_RESTORE_FAILED: could not restore auto — BMC may need manual intervention.")


# Track XCC fan override state (replaces _ipmi_bypass_active)
_xcc_override_active = False
_xcc_override_activated_at: float = 0.0  # monotonic time when override was activated

# XCC failsafe architecture (FIX 2)
_optimization_owns_control: bool = False
_last_successful_xcc_command_ts: float = 0.0
_XCC_FAILSAFE_TIMEOUT_S = float(os.environ.get("COOLEDAI_XCC_FAILSAFE_TIMEOUT_S", "120"))
_XCC_WATCHDOG_INTERVAL_S = 30.0
_xcc_watchdog_last_check: float = 0.0

# Calibration persistence path (FIX 3)
CALIB_PROFILE_FILE = SAFE_CONFIG_DIR / "calibration_profile.json"


def _auto_release_xcc_if_spike_over(dry_run: bool) -> None:
    """Auto-release XCC override after spike_hold_duration when GPU power drops below trigger.

    Prevents the override from being held indefinitely after a transient spike.
    """
    global _xcc_override_active, _xcc_override_activated_at

    if not _xcc_override_active or _xcc_override_activated_at <= 0:
        return

    try:
        if (_THERMAL_CALIBRATOR is None
                or not _THERMAL_CALIBRATOR.is_calibrated
                or _THERMAL_CALIBRATOR.profile is None):
            return
        p = _THERMAL_CALIBRATOR.profile
        elapsed = time.monotonic() - _xcc_override_activated_at
        if elapsed < p.spike_hold_duration_s:
            return  # Still within hold window

        # Check if GPU power has dropped below active_compute_trigger
        # Use the most recent GPU power from the last sensor snapshot
        gpu_power_below_trigger = False
        if hasattr(p, "active_compute_trigger_w") and p.active_compute_trigger_w > 0:
            # We read GPU power from the last snapshot stored in the global
            last_gpu_power = getattr(_auto_release_xcc_if_spike_over, "_last_gpu_power", None)
            if last_gpu_power is not None and last_gpu_power < p.active_compute_trigger_w:
                gpu_power_below_trigger = True

        if gpu_power_below_trigger:
            _log.info(
                "XCC_FAN_AUTO_RELEASE: override held %.0fs (spike_hold=%.0fs), "
                "GPU power below trigger — restoring auto fan control.",
                elapsed, p.spike_hold_duration_s,
            )
            _restore_xcc_auto(dry_run)
            _xcc_override_active = False
            _xcc_override_activated_at = 0.0
    except Exception as exc:
        _log.debug("XCC auto-release check failed: %s", exc)


def set_fan_duty(
    caps: SensorCapabilities,
    duty: int,
    dry_run: bool = False,
    hw_rpm: float = 0.0,
    rated_max_rpm: float = 7000.0,
) -> str:
    """Try XCC -> IPMI -> PWM -> GPU fan.  Returns the method used or 'none'.

    XCC Redfish fan control is attempted first for Lenovo ThinkSystem servers
    (where /dev/ipmi0 is not available). Falls back to IPMI, PWM, GPU fan.

    When target_rpm exceeds the profile-driven threshold AND hardware RPM is < 85%
    of target (not responding), attempts XCC Redfish override.

    Auto-releases override after spike_hold_duration when GPU power drops below
    active_compute_trigger.
    """
    global _xcc_override_active, _xcc_override_activated_at, _last_successful_xcc_command_ts

    target_rpm = (duty / 100.0) * rated_max_rpm

    # --- XCC Redfish fan control (primary for Lenovo ST550) ---
    if _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available:
        threshold_rpm = _get_xcc_override_threshold_rpm()
        if target_rpm > threshold_rpm and (hw_rpm <= 0 or hw_rpm < target_rpm * 0.85):
            if _try_xcc_fan_set(duty, rated_max_rpm, hw_rpm, dry_run):
                if not _xcc_override_active:
                    _xcc_override_activated_at = time.monotonic()
                _xcc_override_active = True
                _last_successful_xcc_command_ts = time.monotonic()
                return "xcc_redfish"
        elif _xcc_override_active and target_rpm < 2000:
            # Target dropped low enough — restore auto
            _restore_xcc_auto(dry_run)
            _xcc_override_active = False
            _xcc_override_activated_at = 0.0

        # Auto-release: after spike_hold_duration, if GPU power dropped below trigger
        if _xcc_override_active and _xcc_override_activated_at > 0:
            _auto_release_xcc_if_spike_over(dry_run)

    # --- Standard IPMI ---
    if caps.ipmi and _try_ipmi_set_duty(caps, duty, dry_run):
        return "ipmi"

    # --- PWM ---
    if caps.pwm_paths and _try_pwm_set_duty(caps, duty, dry_run):
        return "pwm"

    # --- GPU fan ---
    if (caps.nvidia_gpu or caps.amd_gpu) and _try_gpu_fan_set(duty, dry_run):
        return "gpu_fan"

    return "none"


def revert_fans_to_auto(caps: SensorCapabilities, dry_run: bool = False) -> None:
    """Best-effort revert to hardware-default fan control."""
    if dry_run:
        _log.info("[DRY-RUN] Reverting fans to auto.")
        return
    # XCC Redfish restore (Lenovo ThinkSystem)
    if _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available:
        _restore_xcc_auto(dry_run=False)
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


def fetch_optimize_control(
    api_url: str,
    token: str,
    snap: "ThermalSnapshot",
    node_id: str,
) -> Optional[int]:
    """
    Call full optimization engine for target_duty (0-100).
    Returns None on any failure — caller must fall back to local curve.
    """
    fan_rpms = list(snap.fan_rpms.values()) if snap.fan_rpms else []
    fan_rpm = int(sum(fan_rpms) / len(fan_rpms)) if fan_rpms else 2000
    gpu_power = sum(snap.gpu_power_w) if snap.gpu_power_w else 50.0
    cpu_temp = max(snap.cpu_temps) if snap.cpu_temps else None
    payload: Dict[str, Any] = {
        "temp_c": snap.max_temp_c,
        "fan_rpm": float(fan_rpm),
        "gpu_power_w": gpu_power,
        "cpu_temp_c": cpu_temp,
        "node_id": node_id,
        "max_fan_rpm": 7000.0,
    }
    # FIX E/G: Include peak power for spike detection on server side
    if snap.peak_gpu_power_w is not None:
        payload["peak_power_w"] = snap.peak_gpu_power_w
    # Redfish enrichment for optimization engine
    if snap.redfish_ambient_temp is not None:
        payload["ambient_temp_c"] = snap.redfish_ambient_temp
    if snap.redfish_exhaust_temp is not None:
        payload["exhaust_temp_c"] = snap.redfish_exhaust_temp
    if snap.redfish_fan_pcts:
        avg_rf = sum(snap.redfish_fan_pcts.values()) / len(snap.redfish_fan_pcts)
        payload["bmc_fan_pct_avg"] = round(avg_rf, 1)
    # Send calibration profile to API so predictor can recalibrate
    if (_THERMAL_CALIBRATOR is not None
            and _THERMAL_CALIBRATOR.is_calibrated
            and hasattr(_THERMAL_CALIBRATOR, "profile")
            and _THERMAL_CALIBRATOR.profile is not None):
        try:
            payload["calibration_profile"] = _THERMAL_CALIBRATOR.profile.to_dict()
        except Exception:
            pass
    lcd = getattr(fetch_optimize_control, "_last_applied_duty", None)
    if lcd is not None:
        payload["last_commanded_duty"] = float(lcd)
    url_opt = f"{api_url.rstrip('/')}/api/v1/optimize/control"
    result = _http_request(url_opt, token, "POST", payload, timeout_s=OPTIMIZE_TIMEOUT_S)
    if result is not None and isinstance(result.get("target_duty"), (int, float)):
        # Throttled INFO so operators can confirm deployed API exposes policy fields
        # (soft floor vs hardware/IPMI cap) without flooding logs every 3s.
        now_m = time.monotonic()
        last_m = getattr(fetch_optimize_control, "_last_policy_log_mono", 0.0)
        if now_m - last_m >= 45.0:
            sf = result.get("policy_soft_floor_rpm")
            ff = result.get("policy_floor_forced_after_layers")
            cap = result.get("policy_capacity_rpm")
            if sf is not None or ff is not None or cap is not None:
                _log.info(
                    "optimize/control policy: soft_floor_rpm=%s floor_forced_after_layers=%s "
                    "capacity_rpm=%s target_duty=%s",
                    sf,
                    ff,
                    cap,
                    result.get("target_duty"),
                )
                fetch_optimize_control._last_policy_log_mono = now_m  # type: ignore[attr-defined]
        return int(max(0, min(100, result["target_duty"])))
    return None


# ===================================================================
# Main loop
# ===================================================================

_manual_control_active = False
_dry_run_flag = False


def _cleanup(*_args: Any) -> None:
    global _manual_control_active, _xcc_override_active, _optimization_owns_control
    # 1. Save calibration profile before anything else
    if _THERMAL_CALIBRATOR is not None and _THERMAL_CALIBRATOR.is_calibrated:
        try:
            _THERMAL_CALIBRATOR.save_profile(str(CALIB_PROFILE_FILE))
        except Exception as exc:
            _log.debug("Calibration profile save on cleanup failed: %s", exc)
    # 2. Reset GPU power governor
    _gpu_power_limits_cleanup()
    # 3. Revert standard fan control (IPMI/PWM/GPU)
    if _manual_control_active:
        _log.warning("Cleanup: reverting fans to hardware auto.")
        caps = getattr(_cleanup, "_caps", SensorCapabilities())
        revert_fans_to_auto(caps, dry_run=_dry_run_flag)
        _manual_control_active = False
    # 4. Restore XCC auto LAST (BMC takes over after all other resets)
    if _xcc_override_active or _optimization_owns_control:
        _restore_xcc_auto(_dry_run_flag)
        _xcc_override_active = False
        _optimization_owns_control = False
        _log.info("[XCC_FAN] clean_shutdown — XCC auto mode restored")


def run_agent(
    api_url: str,
    token: str,
    node_id: str,
    dry_run: bool = False,
    ipmi_variant: Optional[str] = None,
    shutdown_on_critical: bool = False,
    gpu_power_management: bool = False,
    redfish_url: str = "",
    redfish_user: str = "USERID",
    redfish_pass: str = "***REDACTED_BMC_PASS***",
) -> None:
    global _running, _manual_control_active, _dry_run_flag
    global _GPU_GPG_MOD, _GPU_PL_ENVELOPES, _GPU_PL_STATE
    _dry_run_flag = dry_run
    _GPU_GPG_MOD = None
    _GPU_PL_ENVELOPES = []
    _GPU_PL_STATE = {"last_limits": {}, "last_tick": 0.0}

    # --- Discover ---
    caps = discover_sensors(dry_run, ipmi_variant=ipmi_variant)

    # --- Redfish / XCC discovery ---
    if redfish_url and not dry_run:
        caps.redfish = _discover_redfish(redfish_url, redfish_user, redfish_pass)
    elif redfish_url and dry_run:
        _log.info("[DRY-RUN] Redfish URL provided but skipped in dry-run mode.")

    # --- XCC Fan Controller discovery (Lenovo ThinkSystem BMC) ---
    global _XCC_FAN_CONTROLLER
    global _optimization_owns_control, _last_successful_xcc_command_ts
    XCCFanControllerCls = _try_load_xcc_fan_controller()
    if XCCFanControllerCls is not None:
        _XCC_FAN_CONTROLLER = XCCFanControllerCls()
        if _XCC_FAN_CONTROLLER.available:
            _log.info(
                "XCC fan controller AVAILABLE — Redfish fan control enabled "
                "(replaces IPMI bypass for Lenovo ThinkSystem)"
            )
            # FIX 2: Immediately take ownership of fan control from BMC
            if not dry_run:
                startup_pct = 50  # safe bootstrap fan %
                if _XCC_FAN_CONTROLLER.set_manual_fan_percent(startup_pct):
                    _optimization_owns_control = True
                    _last_successful_xcc_command_ts = time.monotonic()
                    _log.info(
                        "[XCC_FAN] optimization_owns_control=True — "
                        "XCC auto mode suspended at startup (fan_pct=%d)",
                        startup_pct,
                    )
                else:
                    _log.warning(
                        "[XCC_FAN] Failed to take control at startup — "
                        "BMC auto mode still active"
                    )
        else:
            _log.info("XCC fan controller loaded but BMC not reachable — XCC fan control disabled.")
            _XCC_FAN_CONTROLLER = None
    else:
        _XCC_FAN_CONTROLLER = None
        _log.debug("XCCFanController not available — IPMI/PWM/GPU fan fallback only.")

    _cleanup._caps = caps  # type: ignore[attr-defined]
    _log.info(caps.summary())

    xcc_available = _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available
    has_control = caps.ipmi or bool(caps.pwm_paths) or caps.nvidia_gpu or xcc_available
    if not has_control:
        _log.warning("No fan control method available — running in REPORT-ONLY mode.")

    # --- Safety hooks ---
    atexit.register(_cleanup)
    atexit.register(_gpu_power_limits_cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(143)))

    global _GPU_GOVERNOR
    gpg_mod = _try_load_gpu_power_governor() if gpu_power_management else None
    if gpu_power_management:
        if gpg_mod is None:
            _log.warning(
                "GPU power management requested but gpu_power_governor is unavailable "
                "(run agent from repo root so core/optimization is importable)."
            )
        elif not caps.nvidia_gpu:
            _log.warning("GPU power management skipped — no NVIDIA GPU detected.")
        else:
            envs = gpg_mod.query_nvidia_power_envelopes()
            if not envs:
                _log.warning(
                    "GPU power management skipped — nvidia-smi did not return power envelopes "
                    "(driver / permissions)."
                )
            else:
                _GPU_GPG_MOD = gpg_mod
                _GPU_PL_ENVELOPES = list(envs)
                # Create Phase 2 GPUPowerGovernor class
                try:
                    _GPU_GOVERNOR = gpg_mod.GPUPowerGovernor.from_env(
                        envelopes=envs, dry_run=dry_run,
                    )
                    _log.info(
                        "GPU power governor ENABLED (Phase 2) — %d GPU(s), "
                        "performance_mode=%s, calibration_window=%.0fs",
                        len(envs),
                        _GPU_GOVERNOR.performance_mode if _GPU_GOVERNOR else "?",
                        _GPU_GOVERNOR._calibration_window_s if _GPU_GOVERNOR else 0,
                    )
                except Exception as exc:
                    _log.warning("GPUPowerGovernor init failed: %s — falling back to free-function path", exc)
                    _GPU_GOVERNOR = None
                    full, soft, hard, _, _ = gpg_mod.load_governor_config_from_env()
                    _log.info(
                        "GPU power management ENABLED (legacy) — %d GPU(s), "
                        "temp bands full≤%.0f soft≤%.0f hard≤%.0f °C",
                        len(envs), full, soft, hard,
                    )

    # --- Thermal calibrator (auto-discover thresholds) ---
    global _THERMAL_CALIBRATOR
    ThermalCalibratorCls = _try_load_thermal_calibrator()
    if ThermalCalibratorCls is not None:
        _THERMAL_CALIBRATOR = ThermalCalibratorCls()
        # FIX 3: Try loading saved calibration profile
        calib_path = os.environ.get("COOLEDAI_CALIBRATION_PATH", str(CALIB_PROFILE_FILE))
        if _THERMAL_CALIBRATOR.load_profile(calib_path):
            _log.info(
                "CooledAI loaded cached calibration — "
                "active_floor=%.0f RPM | spike_hold=%.0f RPM",
                _THERMAL_CALIBRATOR.profile.active_compute_fan_floor_rpm,
                _THERMAL_CALIBRATOR.profile.spike_hold_fan_floor_rpm,
            )
        else:
            _log.info(
                "CooledAI calibrating — using bootstrap defaults for %.0fs. "
                "System is protected but not yet optimized for this hardware.",
                _THERMAL_CALIBRATOR._window_s,
            )
    else:
        _THERMAL_CALIBRATOR = None
        _log.debug("ThermalCalibrator not available — using static thresholds.")

    # --- Load config ---
    cfg = load_cached_config()
    # Env override: COOLEDAI_CONTROL_ENABLED=false disables fan control (report-only)
    _ctrl_env = os.environ.get("COOLEDAI_CONTROL_ENABLED", "").strip().lower()
    if _ctrl_env in ("0", "false", "no", "off"):
        cfg.control_enabled = False
        _log.info("Control DISABLED via COOLEDAI_CONTROL_ENABLED env — report-only mode.")
    telemetry_url = f"{api_url.rstrip('/')}/api/v1/telemetry"

    _log.info(
        "Agent started — node=%s  api=%s  target=%.0f°C  critical=%.0f°C  "
        "control=%s  dry_run=%s  shutdown_on_critical=%s  gpu_power_mgmt=%s  "
        "redfish=%s",
        node_id, api_url, cfg.target_temp, cfg.critical_temp_c,
        has_control, dry_run, shutdown_on_critical,
        bool(_GPU_PL_ENVELOPES),
        bool(caps.redfish),
    )
    if not token:
        _log.warning("API key is empty — telemetry POSTs will likely be rejected.")

    last_config_fetch = 0.0
    consecutive_post_failures = 0
    watchdog_armed = False  # Only after first successful telemetry
    _in_calibration_window = (_THERMAL_CALIBRATOR is not None and not _THERMAL_CALIBRATOR.is_calibrated)

    while _running:
        loop_start = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Read all sensors (Max-Voter) ---
        snap = read_sensors(caps, dry_run)

        # --- Feed thermal calibrator ---
        if _THERMAL_CALIBRATOR is not None:
            fan_rpms_list = list(snap.fan_rpms.values()) if snap.fan_rpms else []
            was_calibrated = _THERMAL_CALIBRATOR.is_calibrated
            _THERMAL_CALIBRATOR.update(
                fan_rpms=fan_rpms_list or None,
                gpu_power_w=snap.gpu_power_w or None,
                gpu_temp_c=snap.gpu_temps or None,
                cpu_temp_c=snap.cpu_temps or None,
            )
            if not was_calibrated and _THERMAL_CALIBRATOR.is_calibrated:
                _in_calibration_window = False  # calibration done — release idle stepping
                p = _THERMAL_CALIBRATOR.profile
                _log.info(
                    "CooledAI calibration complete — thresholds set from observation: "
                    "active_floor=%.0f RPM | spike_hold=%.0f RPM | trigger=%.1f°C | "
                    "hysteresis=%.0f RPM | slew_up=%.0f RPM/cycle | slew_down=%.0f RPM/cycle | "
                    "min_response=%.0f RPM | hold_duration=%.0fs",
                    p.active_compute_fan_floor_rpm,
                    p.spike_hold_fan_floor_rpm,
                    p.spike_trigger_temp_c,
                    p.hysteresis_rpm,
                    p.slew_rate_up_rpm_per_cycle,
                    p.slew_rate_down_rpm_per_cycle,
                    p.min_response_quantum_rpm,
                    p.spike_hold_duration_s,
                )
                # FIX 3: Persist calibration profile
                _THERMAL_CALIBRATOR.save_profile(str(CALIB_PROFILE_FILE))
            # FIX 3: Save after EWMA recalibration too
            elif _THERMAL_CALIBRATOR.is_calibrated and getattr(_THERMAL_CALIBRATOR, "_recalibrated_this_tick", False):
                _THERMAL_CALIBRATOR.save_profile(str(CALIB_PROFILE_FILE))

        # --- GPU dynamic power cap (nvidia-smi -pl), independent of fan duty ---
        if _GPU_GOVERNOR is not None and snap.gpu_temps:
            # Feed profile from thermal calibrator when it becomes available
            if (_THERMAL_CALIBRATOR is not None
                    and _THERMAL_CALIBRATOR.is_calibrated
                    and hasattr(_THERMAL_CALIBRATOR, "profile")
                    and _THERMAL_CALIBRATOR.profile is not None):
                _GPU_GOVERNOR.update_from_profile(_THERMAL_CALIBRATOR.profile)
            # Phase 2 governor handles calibration, parity guard, hysteresis, logging
            gpu_power_list = list(snap.gpu_power_w) if snap.gpu_power_w else None
            changes = _GPU_GOVERNOR.tick(snap.gpu_temps, gpu_power_list)
            for idx, w, reason in changes:
                gt = snap.gpu_temps[idx] if idx < len(snap.gpu_temps) else float("nan")
                _log.info(
                    "GPU %d power limit -> %d W (GPU temp %.1f°C, reason=%s)",
                    idx, int(w), gt, reason,
                )
        elif _GPU_GPG_MOD and _GPU_PL_ENVELOPES and snap.gpu_temps:
            # Legacy free-function fallback (no GPUPowerGovernor class available)
            gpg_live = _GPU_GPG_MOD
            full, soft, hard, min_dw, min_int = gpg_live.load_governor_config_from_env()
            nowm = time.monotonic()
            if nowm - float(_GPU_PL_STATE["last_tick"]) >= min_int:
                _GPU_PL_STATE["last_tick"] = nowm
                targets = gpg_live.compute_all_targets(
                    snap.gpu_temps,
                    _GPU_PL_ENVELOPES,
                    temp_full_power_c=full,
                    temp_soft_start_c=soft,
                    temp_hard_c=hard,
                )
                for idx, w in targets:
                    prev = _GPU_PL_STATE["last_limits"].get(idx)
                    if prev is not None and abs(w - prev) < min_dw:
                        continue
                    if gpg_live.set_gpu_power_limit_w(idx, int(w), dry_run=dry_run):
                        _GPU_PL_STATE["last_limits"][idx] = w
                        gt = snap.gpu_temps[idx] if idx < len(snap.gpu_temps) else float("nan")
                        _log.info(
                            "GPU %d power limit -> %d W (GPU temp %.1f°C)",
                            idx, int(w), gt,
                        )

        # --- Safety watchdog ---
        if snap.max_temp_c >= cfg.critical_temp_c:
            _log.critical(
                "WATCHDOG: %s at %.1f°C >= critical %.0f°C! "
                "Reverting fans and exiting.",
                snap.source_of_max, snap.max_temp_c, cfg.critical_temp_c,
            )
            revert_fans_to_auto(caps, dry_run)
            _manual_control_active = False
            _gpu_power_limits_cleanup()
            if shutdown_on_critical and not dry_run:
                _log.critical(
                    "WATCHDOG: shutdown_on_critical enabled — initiating host shutdown."
                )
                _emergency_shutdown(
                    f"CooledAI critical temp {snap.max_temp_c:.1f}C >= {cfg.critical_temp_c:.0f}C"
                )
            sys.exit(1)

        # --- Control ---
        if has_control and cfg.control_enabled:
            api_duty = fetch_optimize_control(api_url, token, snap, node_id)
            if api_duty is not None:
                duty = api_duty
            else:
                duty = target_duty_pct(snap.max_temp_c, cfg.target_temp)
                if consecutive_post_failures == 0:
                    _log.debug("Optimize API unreachable — using local curve (hardware protected)")
            _fan_rpms_for_duty = list(snap.fan_rpms.values()) if snap.fan_rpms else []
            _hw_rpm_for_duty = float(sum(_fan_rpms_for_duty) / len(_fan_rpms_for_duty)) if _fan_rpms_for_duty else 0.0
            # Enrich hw_rpm from XCC Redfish if IPMI tach is unavailable
            if _hw_rpm_for_duty <= 0 and _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available:
                xcc_fans = _XCC_FAN_CONTROLLER.read_fan_speeds()
                if xcc_fans:
                    xcc_pcts = [f["reading_pct"] for f in xcc_fans if f.get("reading_pct") is not None]
                    if xcc_pcts:
                        avg_pct = sum(xcc_pcts) / len(xcc_pcts)
                        # Convert XCC fan % to RPM estimate using rated max
                        _hw_rpm_for_duty = (avg_pct / 100.0) * 7000.0
                        _log.debug("[XCC_FAN] hw_rpm estimated from XCC: %.0f RPM (%.1f%%)", _hw_rpm_for_duty, avg_pct)
            # Feed GPU power for XCC auto-release check
            if snap.gpu_power_w:
                _auto_release_xcc_if_spike_over._last_gpu_power = max(snap.gpu_power_w)  # type: ignore[attr-defined]

            # --- Calibration idle stepping (circular reference fix) ---
            # During the observation window, if GPU is idle the optimizer would
            # command ~3500 RPM — the calibrator would measure that as
            # fan_idle_rpm, inflating the anchor.  Instead, force fans to the
            # bootstrap floor so the calibrator sees true hardware idle speed.
            if _in_calibration_window and _THERMAL_CALIBRATOR is not None:
                _cal_trigger = _THERMAL_CALIBRATOR.profile.active_compute_trigger_w
                _cal_gpu_power = max(snap.gpu_power_w) if snap.gpu_power_w else 0.0
                if _cal_gpu_power < _cal_trigger:
                    _bootstrap_rpm = getattr(_THERMAL_CALIBRATOR, "_bootstrap_floor", None)
                    if _bootstrap_rpm is not None and _bootstrap_rpm > 0:
                        _bootstrap_duty = int(round((_bootstrap_rpm / 7000.0) * 100.0))
                        _bootstrap_duty = max(15, min(100, _bootstrap_duty))
                        _log.info(
                            "[THERMAL_CAL] calibration_idle_step: gpu_power=%.1fW "
                            "fan_forced_to=%d%% (%.0f RPM) — letting calibrator "
                            "observe true idle",
                            _cal_gpu_power, _bootstrap_duty, _bootstrap_rpm,
                        )
                        duty = _bootstrap_duty

            method = set_fan_duty(caps, duty, dry_run, hw_rpm=_hw_rpm_for_duty)
            if method != "none":
                _manual_control_active = True
                fetch_optimize_control._last_applied_duty = int(duty)  # type: ignore[attr-defined]
            else:
                _log.debug("Fan set returned 'none' — no method succeeded this cycle.")
        else:
            duty = -1
            method = "report_only"

        # --- XCC failsafe watchdog (FIX 2) ---
        if _XCC_FAN_CONTROLLER is not None:
            _xcc_failsafe_watchdog(dry_run, current_duty_pct=max(30, duty) if duty >= 0 else 50)

        # --- [FAN_DIAG] Periodic fan diagnostic (every 10s) ---
        _fan_diag_interval = 10.0
        _fan_diag_last_t = getattr(run_agent, "_fan_diag_last_t", 0.0)
        if loop_start - _fan_diag_last_t >= _fan_diag_interval:
            run_agent._fan_diag_last_t = loop_start  # type: ignore[attr-defined]
            fan_rpms_list = list(snap.fan_rpms.values()) if snap.fan_rpms else []
            hw_rpm = int(sum(fan_rpms_list) / len(fan_rpms_list)) if fan_rpms_list else 0
            # Enrich hw_rpm from XCC Redfish if IPMI tach unavailable
            xcc_rpm_est = 0
            if hw_rpm <= 0 and _XCC_FAN_CONTROLLER is not None and _XCC_FAN_CONTROLLER.available:
                _xcc_diag_fans = _XCC_FAN_CONTROLLER.read_fan_speeds()
                if _xcc_diag_fans:
                    _xcc_pcts = [f["reading_pct"] for f in _xcc_diag_fans if f.get("reading_pct") is not None]
                    if _xcc_pcts:
                        xcc_rpm_est = int((sum(_xcc_pcts) / len(_xcc_pcts) / 100.0) * 7000.0)
            # Reconstruct target_rpm from duty + max
            target_rpm = int(round((duty / 100.0) * 7000.0)) if duty >= 0 else -1
            _log.info(
                "[FAN_DIAG] target_duty=%d%% target_rpm=%d hw_rpm=%d "
                "xcc_rpm_est=%d gpu_power_w=%.1f peak_power_w=%s method=%s "
                "control_enabled=%s xcc_available=%s",
                duty, target_rpm, hw_rpm, xcc_rpm_est,
                sum(snap.gpu_power_w) if snap.gpu_power_w else 0.0,
                snap.peak_gpu_power_w,
                method,
                cfg.control_enabled,
                bool(_XCC_FAN_CONTROLLER and _XCC_FAN_CONTROLLER.available),
            )

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
            # FIX E/G: Include peak power alongside EWMA-smoothed value
            if snap.peak_gpu_power_w is not None:
                rec["peak_power_w"] = snap.peak_gpu_power_w
            records.append(rec)
        if snap.chassis_temps:
            records.append({
                "node_id": f"{node_id}/chassis",
                "timestamp": now_iso,
                "temperature_c": round(max(snap.chassis_temps), 1),
            })
        if snap.fan_rpms or snap.fan_power_w is not None or snap.gpu_fan_pcts:
            tach_rpms = list(snap.fan_rpms.values())
            avg_rpm = int(sum(tach_rpms) / len(tach_rpms)) if tach_rpms else 0
            # Graph needs a fan series: prefer chassis tach; else GPU fan % → scaled RPM
            if avg_rpm <= 0 and snap.gpu_fan_pcts:
                avg_pct = sum(snap.gpu_fan_pcts) / len(snap.gpu_fan_pcts)
                avg_rpm = int(round((avg_pct / 100.0) * GPU_FAN_PCT_TO_RPM_MAX))
            fan_rec: Dict[str, Any] = {
                "node_id": f"{node_id}/fans",
                "timestamp": now_iso,
                "fan_rpms": snap.fan_rpms,
                "fan_rpm": max(0, avg_rpm),
            }
            if snap.gpu_fan_pcts:
                fan_rec["gpu_fan_speed_pct_avg"] = round(
                    sum(snap.gpu_fan_pcts) / len(snap.gpu_fan_pcts), 1
                )
            if snap.fan_power_w is not None:
                fan_rec["raw_fan_wattage"] = round(snap.fan_power_w, 1)
            # Enrich fan record with Redfish fan percentages
            if snap.redfish_fan_pcts:
                fan_rec["redfish_fan_pcts"] = {
                    k: round(v, 1) for k, v in snap.redfish_fan_pcts.items()
                }
                avg_rf_pct = sum(snap.redfish_fan_pcts.values()) / len(snap.redfish_fan_pcts)
                fan_rec["redfish_fan_pct_avg"] = round(avg_rf_pct, 1)
            records.append(fan_rec)

        # Redfish enrichment record (ambient, exhaust, all XCC temps)
        if snap.redfish_temps:
            rf_rec: Dict[str, Any] = {
                "node_id": f"{node_id}/redfish",
                "timestamp": now_iso,
                "redfish_temps": {k: round(v, 1) for k, v in snap.redfish_temps.items()},
            }
            if snap.redfish_ambient_temp is not None:
                rf_rec["ambient_temp_c"] = round(snap.redfish_ambient_temp, 1)
            if snap.redfish_exhaust_temp is not None:
                rf_rec["exhaust_temp_c"] = round(snap.redfish_exhaust_temp, 1)
            if snap.redfish_fan_pcts:
                rf_rec["fan_pcts"] = {
                    k: round(v, 1) for k, v in snap.redfish_fan_pcts.items()
                }
            records.append(rf_rec)

        payload = {"agent_id": node_id, "telemetry": records}
        ok = post_with_backoff(telemetry_url, payload, token, max_attempts=3)
        if ok:
            consecutive_post_failures = 0
            if not watchdog_armed:
                watchdog_armed = True
                _log.info("First telemetry sent — systemd watchdog armed")
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
                # Preserve env override for control_enabled
                if _ctrl_env in ("0", "false", "no", "off"):
                    remote_cfg.control_enabled = False
                cfg = remote_cfg
            last_config_fetch = now_mono

        # --- systemd watchdog heartbeat (only after first successful POST) ---
        if watchdog_armed:
            _sd_notify_watchdog()

        # --- Pace ---
        elapsed = time.monotonic() - loop_start
        sleep_s = max(0, cfg.poll_interval_s - elapsed)
        if sleep_s > 0 and _running:
            time.sleep(sleep_s)


def _shutdown_signal(signum: int, _frame: Any) -> None:
    global _running
    _running = False


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emergency_shutdown(reason: str) -> None:
    """Best-effort immediate host shutdown (requires root)."""
    for cmd in (
        ["/sbin/shutdown", "-h", "now", reason],
        ["shutdown", "-h", "now", reason],
    ):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            continue


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
    parser.add_argument("--ipmi-variant", default=os.environ.get("COOLEDAI_IPMI_VARIANT", ""),
                        choices=["", "dell", "lenovo", "generic"],
                        help="Force IPMI fan variant (dell/lenovo/generic). Default: auto-detect.")
    parser.add_argument(
        "--shutdown-on-critical",
        action="store_true",
        default=_bool_from_env("COOLEDAI_SHUTDOWN_ON_CRITICAL", False),
        help="Shutdown host if critical temp is reached.",
    )
    parser.add_argument(
        "--gpu-power-management",
        action="store_true",
        default=_bool_from_env("COOLEDAI_GPU_POWER_MGMT", False),
        help="Dynamic NVIDIA power cap via nvidia-smi -pl (requires root + repo core/).",
    )
    parser.add_argument(
        "--redfish-url",
        default=os.environ.get("COOLEDAI_REDFISH_URL", ""),
        help="Redfish/XCC base URL (e.g. https://169.254.95.118). "
        "Enables enriched thermal telemetry from Lenovo XCC BMC.",
    )
    parser.add_argument(
        "--redfish-user",
        default=os.environ.get("COOLEDAI_REDFISH_USER", "USERID"),
        help="Redfish username (default: USERID).",
    )
    parser.add_argument(
        "--redfish-pass",
        default=os.environ.get("COOLEDAI_REDFISH_PASS", "***REDACTED_BMC_PASS***"),
        help="Redfish password.",
    )
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
        ipmi_variant=args.ipmi_variant.strip() or None,
        shutdown_on_critical=args.shutdown_on_critical,
        gpu_power_management=bool(args.gpu_power_management),
        redfish_url=args.redfish_url.strip(),
        redfish_user=args.redfish_user.strip(),
        redfish_pass=args.redfish_pass,
    )
    _log.info("Agent stopped.")


if __name__ == "__main__":
    main()
