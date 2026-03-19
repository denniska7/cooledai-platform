#!/usr/bin/env python3
"""
ST550 GPU Telemetry Agent — lightweight, single-purpose.

Reads GPU temperatures via pynvml (nvidia-ml-py3) and POSTs them to the
CooledAI API every POLL_INTERVAL seconds.  Designed to survive driver
hiccups, network blips, and cold-start race conditions after a power
failure.

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
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

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
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry

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
    """
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
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
            records.append(rec)
        except pynvml.NVMLError as exc:
            print(f"[telemetry] GPU {i} read error: {exc}")
    return records


# ── API posting ─────────────────────────────────────────────────────
def _post_telemetry(session: requests.Session, records: list[dict]) -> None:
    """POST the telemetry payload with retry + exponential back-off."""
    payload = {
        "agent_id": NODE_ID,
        "telemetry": records,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    url = f"{API_URL.rstrip('/')}/api/v1/telemetry"

    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=10)
            print(
                f"[telemetry] POST {url} — "
                f"Status: {resp.status_code} | "
                f"Body: {resp.text.strip()} | "
                f"GPUs: {len(records)}"
            )
            if resp.status_code < 300:
                return
        except requests.RequestException as exc:
            print(f"[telemetry] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")

        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2


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

    while True:
        records = _read_gpu_temps(gpu_count)
        if records:
            _post_telemetry(session, records)
        else:
            print("[telemetry] No GPU data this cycle")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
