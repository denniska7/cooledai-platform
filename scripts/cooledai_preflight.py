#!/usr/bin/env python3
"""
CooledAI Preflight Check — validate configuration before first run.

Usage:
    python3 scripts/cooledai_preflight.py [--config configs/cooledai.yaml]

Checks:
  1. Validate cooledai.yaml against schema
  2. Test XCC BMC connectivity (Redfish GET /redfish/v1/)
  3. Check API health endpoint (GET /api/v1/health)
  4. Send one test telemetry record and verify 200 response

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check_config(config_path: str) -> bool:
    """Check 1: Validate cooledai.yaml."""
    print(f"\n[1/4] Validating config: {config_path}")
    try:
        from core.config.settings import Settings
        settings = Settings(config_path)
        settings.validate(strict=False)
        print(f"  Client: {settings.client_name or '(not set)'}")
        print(f"  Site:   {settings.site or '(not set)'}")
        print(f"  GPUs:   {settings.gpu_count}")
        print(f"  [{PASS}] Config is valid")
        return True
    except Exception as exc:
        print(f"  [{FAIL}] {exc}")
        return False


def check_xcc(config_path: str) -> bool:
    """Check 2: Test XCC BMC connectivity."""
    print("\n[2/4] Testing XCC BMC connectivity")
    host = os.environ.get("XCC_BMC_HOST", "")
    user = os.environ.get("XCC_BMC_USER", "USERID")
    passwd = os.environ.get("XCC_BMC_PASS", "")

    if not host:
        print(f"  [{FAIL}] XCC_BMC_HOST not set — skipping BMC check")
        print("  Set XCC_BMC_HOST in /etc/cooledai/agent.env or environment")
        return False

    url = f"https://{host}/redfish/v1/"
    print(f"  Connecting to {url} ...")

    try:
        import base64
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        cred = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        req = Request(url, headers={"Authorization": f"Basic {cred}"})
        resp = urlopen(req, timeout=10, context=ctx)
        data = json.loads(resp.read())
        product = data.get("Product", data.get("Name", "Unknown"))
        print(f"  BMC Product: {product}")
        print(f"  [{PASS}] XCC BMC reachable")
        return True
    except Exception as exc:
        print(f"  [{FAIL}] Cannot reach BMC: {exc}")
        return False


def check_api_health(api_url: str, api_key: str) -> bool:
    """Check 3: GET /api/v1/health."""
    print("\n[3/4] Checking API health endpoint")
    url = f"{api_url.rstrip('/')}/api/v1/health"
    print(f"  GET {url}")

    try:
        req = Request(url, headers={"X-API-Key": api_key})
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read())
        status = data.get("status", "unknown")
        version = data.get("version", "unknown")
        print(f"  Status:  {status}")
        print(f"  Version: {version}")
        if status in ("healthy", "degraded"):
            print(f"  [{PASS}] API is responsive")
            return True
        else:
            print(f"  [{FAIL}] API status is '{status}'")
            return False
    except Exception as exc:
        print(f"  [{FAIL}] Cannot reach API: {exc}")
        return False


def check_telemetry(api_url: str, api_key: str, node_id: str) -> bool:
    """Check 4: Send one test telemetry record."""
    print("\n[4/4] Sending test telemetry record")
    url = f"{api_url.rstrip('/')}/api/v1/telemetry"

    payload = json.dumps({
        "node_id": node_id,
        "telemetry": [{
            "node_id": node_id,
            "gpu_temp_c": 42.0,
            "gpu_power_w": 50.0,
            "cpu_temp_c": 38.0,
            "fan_rpm": 3000,
            "timestamp_utc": "2026-01-01T00:00:00Z",
        }],
    }).encode("utf-8")

    try:
        req = Request(
            url,
            data=payload,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        resp = urlopen(req, timeout=10)
        status = resp.getcode()
        body = json.loads(resp.read())
        print(f"  Status: {status}")
        print(f"  Response: {body}")
        if status == 200:
            print(f"  [{PASS}] Telemetry accepted")
            return True
        else:
            print(f"  [{FAIL}] Unexpected status {status}")
            return False
    except Exception as exc:
        print(f"  [{FAIL}] Telemetry POST failed: {exc}")
        return False


def capture_baseline(output_path: str, duration_s: int = 300) -> bool:
    """Check 5: Capture Day 0 baseline profile BEFORE optimization takes control.

    Observes BMC-controlled fan behavior and GPU power at various utilization
    levels for `duration_s` seconds. Writes baseline_profile.json — the permanent
    "before CooledAI" anchor for all savings calculations.

    This file is written ONCE and never overwritten after first write.
    """
    print(f"\n[5/5] Capturing Day 0 baseline ({duration_s}s observation)")

    output = Path(output_path)
    if output.is_file():
        print(f"  Baseline already exists at {output_path}")
        try:
            with open(output) as f:
                bp = json.load(f)
            print(f"  Captured at: {bp.get('baseline_captured_at', 'unknown')}")
            print(f"  Avg util: {bp.get('baseline_avg_util_pct', '?')}%")
            print(f"  [{PASS}] Using existing baseline (not overwritten)")
            return True
        except Exception:
            pass

    # Try to read GPU and fan data
    import subprocess
    import time
    from collections import defaultdict
    from datetime import datetime, timezone

    fan_by_temp: defaultdict = defaultdict(list)  # temp_bucket -> [rpm]
    gpu_by_util: defaultdict = defaultdict(list)  # util_bucket -> [watts]
    all_utils: list = []
    all_watts: list = []
    all_rpms: list = []

    start = time.monotonic()
    samples = 0

    print(f"  Observing for {duration_s}s (Ctrl-C to abort)...")

    while time.monotonic() - start < duration_s:
        try:
            # Read GPU data
            gpu_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()

            gpu_temp = 0.0
            gpu_power = 0.0
            gpu_util = 0.0
            for line in gpu_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    t, p, u = float(parts[0]), float(parts[1]), float(parts[2])
                    gpu_temp = max(gpu_temp, t)
                    gpu_power += p
                    gpu_util = max(gpu_util, u)

            # Read fan RPMs via IPMI
            try:
                fan_out = subprocess.check_output(
                    ["ipmitool", "sdr", "type", "Fan"],
                    timeout=5, stderr=subprocess.DEVNULL,
                ).decode()
                rpms = []
                import re
                for line in fan_out.splitlines():
                    m = re.search(r"(\d+)\s*RPM", line)
                    if m:
                        rpms.append(int(m.group(1)))
                avg_rpm = sum(rpms) / len(rpms) if rpms else 0
            except Exception:
                avg_rpm = 0

            if gpu_temp > 0 and avg_rpm > 0:
                # Bucket by temperature (5°C bands)
                if gpu_temp < 55:
                    bucket = "50-55"
                elif gpu_temp < 60:
                    bucket = "55-60"
                elif gpu_temp < 65:
                    bucket = "60-65"
                elif gpu_temp < 70:
                    bucket = "65-70"
                else:
                    bucket = "70+"
                fan_by_temp[bucket].append(avg_rpm)

                # Bucket by utilization (20% bands)
                if gpu_util < 20:
                    ubucket = "0-20"
                elif gpu_util < 40:
                    ubucket = "20-40"
                elif gpu_util < 60:
                    ubucket = "40-60"
                elif gpu_util < 80:
                    ubucket = "60-80"
                else:
                    ubucket = "80-100"
                gpu_by_util[ubucket].append(gpu_power)

                all_utils.append(gpu_util)
                all_watts.append(gpu_power)
                all_rpms.append(avg_rpm)
                samples += 1

                if samples % 30 == 0:
                    elapsed = int(time.monotonic() - start)
                    print(f"  {elapsed}s/{duration_s}s — {samples} samples, "
                          f"GPU {gpu_temp:.0f}°C {gpu_power:.0f}W {gpu_util:.0f}%, "
                          f"Fan {avg_rpm:.0f} RPM")

        except Exception as exc:
            print(f"  Sample error: {exc}")

        time.sleep(5)  # 5-second intervals

    if samples < 10:
        print(f"  [{FAIL}] Only {samples} samples collected — need at least 10")
        return False

    # Build baseline profile
    baseline = {
        "baseline_captured_at": datetime.now(timezone.utc).isoformat(),
        "baseline_fan_rpm_by_temp": {
            k: round(sum(v) / len(v)) for k, v in sorted(fan_by_temp.items())
        },
        "baseline_gpu_w_by_util": {
            k: round(sum(v) / len(v), 1) for k, v in sorted(gpu_by_util.items())
        },
        "baseline_avg_util_pct": round(sum(all_utils) / len(all_utils), 1),
        "baseline_avg_gpu_w": round(sum(all_watts) / len(all_watts), 1),
        "baseline_avg_fan_rpm": round(sum(all_rpms) / len(all_rpms)),
        "samples": samples,
        "duration_s": duration_s,
    }

    # Write atomically
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(baseline, f, indent=2)
    tmp.replace(output)

    print(f"  Baseline written to {output_path}")
    print(f"  Samples: {samples}")
    print(f"  Avg util: {baseline['baseline_avg_util_pct']}%")
    print(f"  Avg power: {baseline['baseline_avg_gpu_w']}W")
    print(f"  Avg fan RPM: {baseline['baseline_avg_fan_rpm']}")
    print(f"  [{PASS}] Baseline profile captured")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="CooledAI Preflight Check")
    parser.add_argument("--config", default="configs/cooledai.yaml")
    parser.add_argument("--api-url", default=os.environ.get(
        "COOLEDAI_API_URL", "https://proactive-creativity-production.up.railway.app"))
    parser.add_argument("--api-key", default=os.environ.get("COOLEDAI_API_KEY", ""))
    parser.add_argument("--node-id", default=os.environ.get("COOLEDAI_NODE_ID", "preflight-test"))
    parser.add_argument("--baseline-path", default=os.environ.get(
        "COOLEDAI_BASELINE_PROFILE_PATH", "/var/lib/cooledai/baseline_profile.json"))
    parser.add_argument("--baseline-duration", type=int, default=300,
                        help="Baseline observation duration in seconds (default 300)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline capture (checks 1-4 only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CooledAI Preflight Check")
    print("=" * 60)

    results = [
        ("Config validation", check_config(args.config)),
        ("XCC BMC connectivity", check_xcc(args.config)),
        ("API health", check_api_health(args.api_url, args.api_key)),
        ("Telemetry POST", check_telemetry(args.api_url, args.api_key, args.node_id)),
    ]

    if not args.skip_baseline:
        results.append((
            "Day 0 baseline capture",
            capture_baseline(args.baseline_path, args.baseline_duration),
        ))

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        icon = PASS if passed else FAIL
        print(f"  [{icon}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed. CooledAI is ready for deployment.")
        sys.exit(0)
    else:
        print("One or more checks failed. Review errors above before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
