#!/usr/bin/env python3
"""
ST550 GPU Telemetry Agent — lightweight, single-purpose.

Reads GPU temperatures via pynvml (nvidia-ml-py3) and POSTs them to the
CooledAI API every POLL_INTERVAL seconds.  Also attaches **CPU temp**
(sysfs thermal zones + IPMI) and **chassis fan RPM** (ipmitool sdr) so
control nodes without the full cooledai_agent still populate portal series
(`control_cpu_temp_c`, `control_fan_rpm`). Run **start_telemetry.sh under sudo**
so ipmitool can read the BMC.

Network topology
----------------
The ST550 has two NICs:
  eno1np0        → Router / Internet  (192.168.12.x, gw 192.168.12.1)
  enx9a29a6faed33 → Laptop / Wrench  (direct link, no internet)

All API traffic MUST egress via eno1np0.  The script binds outbound
connections to eno1np0's IP using a requests SourceAddressAdapter so the
"Wrench" port can never hijack telemetry traffic.

Usage:
    python3 scripts/st550_telemetry.py                     # defaults
    python3 scripts/st550_telemetry.py --node-id MyNode    # custom node
"""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration (env-overridable) ─────────────────────────────────
API_URL = os.environ.get(
    "COOLEDAI_API_URL",
    "https://proactive-creativity-production.up.railway.app",
)
API_KEY = os.environ.get(
    "COOLEDAI_API_KEY",
    "***REDACTED_API_KEY***",
)
NODE_ID = os.environ.get("COOLEDAI_NODE_ID", "ST550-CooledAI-Predictive")
# 5s heartbeat to lighten network load (was 10s)
POLL_INTERVAL = int(os.environ.get("COOLEDAI_POLL_S", "5"))
# Railway / cold starts: longer timeouts + more attempts than LAN defaults.
MAX_RETRIES = int(os.environ.get("COOLEDAI_TELEMETRY_RETRIES", "5"))
RETRY_BACKOFF = float(os.environ.get("COOLEDAI_TELEMETRY_BACKOFF_S", "2"))
REQUEST_TIMEOUT = float(os.environ.get("COOLEDAI_TELEMETRY_TIMEOUT_S", "35"))
# Chassis CPU + fan tach (IPMI) — same signals the full cooledai_agent posts.
# Control nodes often run only this script; without IPMI they would lack cpu/fan in the portal.
SKIP_IPMI = os.environ.get("COOLEDAI_TELEMETRY_SKIP_IPMI", "").lower() in ("1", "true", "yes")

DATA_IFACE = "eno1np0"
ROUTER_GW = "192.168.12.1"

# ── Dependency bootstrap ────────────────────────────────────────────
REQUIRED_PACKAGES = {"pynvml": "nvidia-ml-py3", "requests": "requests"}


def _ensure_deps() -> None:
    """Install missing packages using the running interpreter's pip."""
    missing = {}
    for mod, pkg in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing[mod] = pkg

    if not missing:
        return

    # Skip pip if it's not available (e.g. minimal Python install, no pip module)
    try:
        importlib.import_module("pip")
    except ImportError:
        print(
            "[telemetry] ERROR: Missing packages {} and pip is not available. "
            "Install manually: sudo apt install python3-pip && pip3 install {}"
            .format(list(missing.values()), " ".join(missing.values()))
        )
        sys.exit(1)

    print(f"[telemetry] Installing missing packages: {list(missing.values())}")
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--break-system-packages",
        *missing.values(),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(
            f"[telemetry] ERROR: pip install failed ({e}). "
            f"Install manually: pip3 install {' '.join(missing.values())}"
        )
        sys.exit(1)
    # Re-import after install so the rest of the script can use them.
    for mod in missing:
        importlib.import_module(mod)


_ensure_deps()

import pynvml  # noqa: E402  (imported after bootstrap)
import requests  # noqa: E402
import requests.adapters  # noqa: E402
import urllib3  # noqa: E402  (bundled with requests)


# ── Network helpers ─────────────────────────────────────────────────
def _get_iface_ip(iface: str) -> str | None:
    """Return the IPv4 address assigned to *iface*, or None."""
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", iface],
            text=True, stderr=subprocess.DEVNULL,
        )
        # e.g. "2: eno1np0  inet 192.168.12.100/24 ..."
        for token in out.split():
            if "/" in token and token[0].isdigit():
                return token.split("/")[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _check_data_port() -> str:
    """Verify eno1np0 is up and can reach the internet.

    Returns the interface's IPv4 address on success, or calls sys.exit
    with a CRITICAL log on failure.
    """
    ip = _get_iface_ip(DATA_IFACE)
    if not ip:
        print(
            f"[network] CRITICAL: Main Data Port ({DATA_IFACE}) is "
            f"unreachable — no IPv4 address assigned."
        )
        sys.exit(1)

    # Ping 8.8.8.8 through the data interface (best-effort; some networks block ICMP).
    try:
        subprocess.check_call(
            ["ping", "-c", "1", "-W", "3", "-I", DATA_IFACE, "8.8.8.8"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[network] {DATA_IFACE} OK — IP {ip}, internet reachable")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"[network] WARN: ping 8.8.8.8 via {DATA_IFACE} failed "
            "(ICMP may be blocked). Continuing — first POST will verify connectivity."
        )
    return ip


class _SourceAddressAdapter(requests.adapters.HTTPAdapter):
    """Force all connections to bind to a specific source IP.

    This guarantees traffic leaves via eno1np0 even if the Wrench port
    (enx…) has a default route with a lower metric.
    """

    def __init__(self, source_address: str, **kwargs):
        self._src = source_address
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["source_address"] = (self._src, 0)
        super().init_poolmanager(*args, **kwargs)


def _build_session(source_ip: str) -> requests.Session:
    """Return a Session whose connections are pinned to *source_ip*."""
    s = requests.Session()
    adapter = _SourceAddressAdapter(source_ip)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ── NVML helpers ────────────────────────────────────────────────────
def _init_nvml(retries: int = 5, wait: float = 3.0) -> int:
    """Initialise NVML with retries.

    After a power failure the NVIDIA driver may report "driver busy" for
    a few seconds.  We retry instead of crashing.

    Returns the GPU device count.
    """
    for attempt in range(1, retries + 1):
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"[telemetry] NVML ready — {count} GPU(s) detected")
            return count
        except pynvml.NVMLError as exc:
            print(
                f"[telemetry] NVML init attempt {attempt}/{retries} "
                f"failed: {exc}"
            )
            if attempt < retries:
                time.sleep(wait)
    print("[telemetry] FATAL: could not initialise NVML after retries")
    sys.exit(1)


def _read_gpu_temps(count: int) -> list[dict]:
    """Return one telemetry record per GPU (temp + power).

    NVML/pynvml often reports 0W when GPU is idle or between sampling windows.
    We apply a floor (10W) or utilization-based estimate to avoid 0W ruining charts.

    FIX E: Each GPU record includes ``peak_power_w`` — the raw max power across
    all GPUs in this polling cycle — so the optimization brain can detect
    sub-interval spikes even when EWMA smoothing masks them.
    """
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    raw_powers: list[float] = []
    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp_c = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            rec: dict = {
                "node_id": f"{NODE_ID}/gpu{i}",  # Pilot vs Control
                "timestamp": now,
                "temperature_c": float(temp_c),
            }
            power_w = 0.0
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = float(power_mw) / 1000.0
            except pynvml.NVMLError:
                pass  # Power not supported on some GPUs

            raw_powers.append(power_w)

            # NVML often reports 0W when idle; avoid ruining telemetry charts
            if power_w <= 0:
                util_pct = 0.0
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_pct = float(util.gpu)  # 0–100
                except pynvml.NVMLError:
                    pass
                if util_pct > 0:
                    power_w = max(10.0, util_pct * 1.2)
                else:
                    power_w = 10.0
            rec["power_draw_w"] = round(power_w, 2)
            try:
                fan_pct = pynvml.nvmlDeviceGetFanSpeed(handle)
                if fan_pct is not None and fan_pct >= 0:
                    rec["fan_speed_pct"] = float(fan_pct)
            except pynvml.NVMLError:
                pass  # Passive / OEM boards may not expose GPU fan
            records.append(rec)
        except pynvml.NVMLError as exc:
            print(f"[telemetry] GPU {i} read error: {exc}")

    # FIX E: Attach peak_power_w (max across all GPUs this cycle) to each record
    if raw_powers:
        peak = round(max(raw_powers), 2)
        for rec in records:
            rec["peak_power_w"] = peak

    return records


def _read_cpu_temps_sysfs() -> list[float]:
    """CPU-ish thermal zones from sysfs (no extra packages)."""
    out: list[float] = []
    base = Path("/sys/class/thermal")
    if not base.is_dir():
        return out
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            typ = (zone / "type").read_text().strip().lower()
            if not any(
                k in typ
                for k in ("cpu", "x86", "pkg", "core", "k10temp", "zen", "processor")
            ):
                continue
            raw = int((zone / "temp").read_text().strip()) / 1000.0
            if 15.0 < raw < 125.0:
                out.append(raw)
        except (OSError, ValueError):
            continue
    return out


def _read_ipmi_cpu_fan() -> tuple[list[float], dict[str, int], float | None]:
    """
    Parse `ipmitool sdr` for CPU temperatures and fan tach (ST550 / Lenovo-style labels).
    Requires root or ipmi group; skipped if command fails.
    """
    cpu_list: list[float] = []
    fan_rpms: dict[str, int] = {}
    fan_power: float | None = None

    try:
        out = subprocess.check_output(
            ["ipmitool", "sdr"],
            timeout=12,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return [], {}, None

    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        name_lower = name.lower()
        value_str = parts[1].strip()

        if "sys fan pwr" in name_lower:
            m = re.search(r"([\d.]+)\s*[Ww]", value_str)
            if m:
                fan_power = float(m.group(1))
            continue

        # CPU / package temperature (avoid DIMM)
        if (
            ("cpu" in name_lower or "processor" in name_lower)
            and "dimm" not in name_lower
        ):
            m = re.search(r"([\d.]+)\s*degrees", value_str, re.IGNORECASE)
            if m:
                t = float(m.group(1))
                if 15.0 < t < 125.0:
                    cpu_list.append(t)
            continue

        # Chassis fan RPM
        if "fan" in name_lower or "tach" in name_lower:
            m = re.search(r"(\d+)\s*RPM", value_str, re.IGNORECASE)
            if m:
                fan_rpms[name] = int(m.group(1))
            else:
                m2 = re.search(r"\b(\d{3,5})\b", value_str)
                if m2 and "na" not in value_str.lower():
                    v = int(m2.group(1))
                    if 200 <= v <= 20000:
                        fan_rpms[name] = v

    return cpu_list, fan_rpms, fan_power


def _build_cpu_fan_records() -> list[dict]:
    """Extra telemetry rows so API fills control_cpu_temp_c / control_fan_rpm like the pilot agent."""
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    cpu_sysfs = _read_cpu_temps_sysfs()
    cpu_ipmi: list[float] = []
    fan_rpms: dict[str, int] = {}
    fan_power: float | None = None

    if not SKIP_IPMI:
        cpu_ipmi, fan_rpms, fan_power = _read_ipmi_cpu_fan()

    # Prefer max of sysfs + IPMI CPU readings
    cpu_all = cpu_sysfs + cpu_ipmi
    if cpu_all:
        records.append(
            {
                "node_id": f"{NODE_ID}/cpu",
                "timestamp": now,
                "temperature_c": round(max(cpu_all), 1),
                "sensor_count": len(cpu_all),
            }
        )

    if fan_rpms:
        tach = list(fan_rpms.values())
        avg_rpm = int(sum(tach) / len(tach)) if tach else 0
        fan_rec: dict = {
            "node_id": f"{NODE_ID}/fans",
            "timestamp": now,
            "fan_rpms": fan_rpms,
            "fan_rpm": max(0, avg_rpm),
        }
        if fan_power is not None:
            fan_rec["raw_fan_wattage"] = round(fan_power, 1)
        records.append(fan_rec)

    return records


def _record_breakdown(records: list[dict]) -> dict[str, int]:
    """Count rows by suffix (gpu/cpu/fans) for clearer logs vs API received=N."""
    counts: dict[str, int] = {}
    for r in records:
        nid = str(r.get("node_id", ""))
        if "/gpu" in nid:
            counts["gpu"] = counts.get("gpu", 0) + 1
        elif "/cpu" in nid:
            counts["cpu"] = counts.get("cpu", 0) + 1
        elif "/fans" in nid:
            counts["fans"] = counts.get("fans", 0) + 1
        else:
            counts["other"] = counts.get("other", 0) + 1
    return counts


# ── API posting ─────────────────────────────────────────────────────
def _post_telemetry(session: requests.Session, records: list[dict]) -> None:
    """POST the telemetry payload with retry + exponential back-off + jitter."""
    payload = {
        "agent_id": NODE_ID,
        "telemetry": records,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    url = f"{API_URL.rstrip('/')}/api/v1/telemetry"
    breakdown = _record_breakdown(records)
    # (connect timeout, read timeout) — slow TLS / Railway wakeups need headroom
    timeout = (min(20.0, REQUEST_TIMEOUT), REQUEST_TIMEOUT)

    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=timeout)
            print(
                f"[telemetry] POST {url} — "
                f"Status: {resp.status_code} | "
                f"Body: {resp.text.strip()} | "
                f"records={len(records)} {breakdown}"
            )
            if resp.status_code < 300:
                return
        except requests.RequestException as exc:
            print(f"[telemetry] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")

        if attempt < MAX_RETRIES:
            time.sleep(backoff + random.uniform(0, 1.25))
            backoff = min(backoff * 2.0, 90.0)


# ── Main loop ───────────────────────────────────────────────────────
def main() -> None:
    import argparse

    global NODE_ID, POLL_INTERVAL, API_URL, API_KEY  # noqa: PLW0603
    parser = argparse.ArgumentParser(description="ST550 GPU Telemetry Agent")
    parser.add_argument("--node-id", default=NODE_ID, help="Agent / node ID")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    NODE_ID = args.node_id
    POLL_INTERVAL = args.interval
    API_URL = args.api_url
    API_KEY = args.api_key

    print(f"[telemetry] Agent={NODE_ID}  API={API_URL}  Interval={POLL_INTERVAL}s")

    # ── Network pre-flight ──────────────────────────────────────────
    source_ip = _check_data_port()
    session = _build_session(source_ip)
    print(f"[network] Outbound traffic pinned to {source_ip} ({DATA_IFACE})")

    gpu_count = _init_nvml()

    chassis_warned = False
    while True:
        records = _read_gpu_temps(gpu_count)
        records.extend(_build_cpu_fan_records())
        if records:
            if not chassis_warned:
                chassis_warned = True
                has_cpu = any("/cpu" in str(r.get("node_id", "")) for r in records)
                has_fans = any("/fans" in str(r.get("node_id", "")) for r in records)
                if not has_cpu:
                    print(
                        "[telemetry] WARN: no /cpu record — "
                        "install ipmitool, run under sudo, or check /sys/class/thermal zones"
                    )
                if not has_fans:
                    print(
                        "[telemetry] WARN: no /fans record — "
                        "need `sudo ipmitool sdr` fan tach readable by this process"
                    )
            _post_telemetry(session, records)
        else:
            print("[telemetry] No GPU data this cycle")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
